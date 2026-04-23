from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import respx
from httpx import Response

from gateway.errors import QuotaExceededError
from gateway.models.requests import GenerateRequest, Message, WebSearchOptions
from gateway.settings import get_settings
from gateway.tools import cache as cache_mod
from gateway.tools import quota as quota_mod
from gateway.tools.providers import gemini as gemini_provider
from gateway.tools.providers import searxng as searxng_provider
from gateway.tools.web_search import web_search


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings():
    s = get_settings()
    s.search_provider = "gemini"
    s.gemini_api_key = "test-key"
    s.search_cache_ttl_news_seconds = 1800
    s.search_cache_ttl_evergreen_seconds = 86400
    s.search_daily_quota_per_key = 500
    s.search_max_iterations = 2
    s.search_total_budget_ms = 8000
    return s


_GEMINI_RESPONSE = {
    "candidates": [
        {
            "content": {"parts": [{"text": "Paris is the capital of France."}]},
            "groundingMetadata": {
                "groundingChunks": [
                    {"web": {"uri": "https://example.com/paris", "title": "Paris"}},
                ]
            },
        }
    ]
}

_SEARXNG_RESPONSE = {
    "results": [
        {"url": "https://example.com/a", "title": "Result A", "content": "snippet A"},
        {"url": "https://example.com/b", "title": "Result B", "content": "snippet B"},
    ]
}


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_returns_cached_without_calling_provider(settings) -> None:
    """If there is a cache hit, the provider should never be called."""
    cached_payload = {
        "text": "cached answer",
        "sources": [{"url": "https://cached.example.com", "title": "Cached"}],
    }

    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock(
        return_value=json.dumps(cached_payload)
    )

    with patch("gateway.tools.web_search.quota_mod.check_and_increment") as mock_quota, \
         patch("gateway.tools.web_search.gemini_provider.search") as mock_provider:

        result = await web_search(
            query="capital of France",
            days=None,
            settings=settings,
            redis_client=redis_mock,
            api_key_id="key_123",
        )

    assert result["text"] == "cached answer"
    assert result["cache_hit"] is True
    mock_quota.assert_not_called()
    mock_provider.assert_not_called()


@pytest.mark.asyncio
async def test_cache_miss_calls_provider_and_stores(settings) -> None:
    """Cache miss should call provider and store result."""
    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock()

    with patch("gateway.tools.web_search.quota_mod.check_and_increment", new_callable=AsyncMock), \
         patch("gateway.tools.web_search.gemini_provider.search", new_callable=AsyncMock) as mock_gemini:

        mock_gemini.return_value = {
            "text": "live answer",
            "sources": [{"url": "https://live.example.com", "title": "Live"}],
        }

        result = await web_search(
            query="live query",
            days=None,
            settings=settings,
            redis_client=redis_mock,
            api_key_id="key_abc",
        )

    assert result["text"] == "live answer"
    assert result["cache_hit"] is False
    redis_mock.set.assert_called_once()


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_gemini_provider_parses_response() -> None:
    """Gemini provider should extract text and grounding sources correctly."""
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models"
        "/gemini-2.5-flash-preview-04-17:generateContent"
    ).mock(return_value=Response(200, json=_GEMINI_RESPONSE))

    result = await gemini_provider.search("capital of France", api_key="fake-key")

    assert result["text"] == "Paris is the capital of France."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["url"] == "https://example.com/paris"
    assert result["sources"][0]["title"] == "Paris"


@pytest.mark.asyncio
@respx.mock
async def test_gemini_provider_raises_on_non_200() -> None:
    """Non-200 from Gemini should raise RuntimeError."""
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models"
        "/gemini-2.5-flash-preview-04-17:generateContent"
    ).mock(return_value=Response(403, text="forbidden"))

    with pytest.raises(RuntimeError, match="403"):
        await gemini_provider.search("query", api_key="bad-key")


# ---------------------------------------------------------------------------
# SearXNG provider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_searxng_provider_parses_response() -> None:
    """SearXNG provider should join snippets and return structured sources."""
    respx.get("http://searxng:8888/search").mock(
        return_value=Response(200, json=_SEARXNG_RESPONSE)
    )

    result = await searxng_provider.search("test query")

    assert "snippet A" in result["text"]
    assert "snippet B" in result["text"]
    assert len(result["sources"]) == 2
    assert result["sources"][0]["url"] == "https://example.com/a"


# ---------------------------------------------------------------------------
# Quota
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quota_exhaustion_raises_error() -> None:
    """When Redis counter exceeds the limit, QuotaExceededError should be raised."""
    redis_mock = AsyncMock()
    # incr returns limit + 1 — over quota
    redis_mock.incr = AsyncMock(return_value=501)
    redis_mock.expire = AsyncMock()

    with pytest.raises(QuotaExceededError):
        await quota_mod.check_and_increment(redis_mock, "key_over_quota", daily_limit=500)


@pytest.mark.asyncio
async def test_quota_within_limit_does_not_raise() -> None:
    """Counter at exactly the limit should NOT raise."""
    redis_mock = AsyncMock()
    redis_mock.incr = AsyncMock(return_value=500)
    redis_mock.expire = AsyncMock()

    # Should not raise
    await quota_mod.check_and_increment(redis_mock, "key_at_limit", daily_limit=500)


@pytest.mark.asyncio
async def test_quota_fails_open_on_redis_error() -> None:
    """Redis failure should be silently swallowed (fail open)."""
    redis_mock = AsyncMock()
    redis_mock.incr = AsyncMock(side_effect=ConnectionError("redis down"))

    # Should not raise
    await quota_mod.check_and_increment(redis_mock, "key_redis_down", daily_limit=500)


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


def _make_engine_result(content: str, finish_reason: str = "stop", tool_calls=None):
    from gateway.engines.base import EngineGenerateResult
    return EngineGenerateResult(
        content=content,
        tokens_in=10,
        tokens_out=5,
        finish_reason=finish_reason,
        tool_calls=tool_calls,
    )


def _make_generate_request(**kwargs) -> GenerateRequest:
    defaults = dict(
        messages=[Message(role="user", content="What happened today?")],
        stream=False,
        enable_web_search=True,
    )
    defaults.update(kwargs)
    return GenerateRequest(**defaults)


def _make_state(settings_obj):
    state = MagicMock()
    state.settings = settings_obj
    state.redis = None
    return state


@pytest.mark.asyncio
async def test_loop_stops_on_stop_finish_reason(settings) -> None:
    """Loop should return immediately if model returns finish_reason=stop on first call."""
    from gateway.tools.loop import run_with_search

    engine = AsyncMock()
    engine.generate = AsyncMock(return_value=_make_engine_result("Final answer", "stop"))

    body = _make_generate_request()
    state = _make_state(settings)

    result = await run_with_search(body=body, engine=engine, state=state, api_key_id="k1")

    assert result.content == "Final answer"
    assert result.finish_reason == "stop"
    engine.generate.assert_called_once()


@pytest.mark.asyncio
async def test_loop_executes_web_search_tool_call(settings) -> None:
    """Loop should execute a web_search tool call and then call generate again."""
    from gateway.tools.loop import run_with_search

    tool_call_result = _make_engine_result(
        content="",
        finish_reason="tool_calls",
        tool_calls=[{"name": "web_search", "arguments": {"query": "latest news"}}],
    )
    final_result = _make_engine_result("Here is what I found.", "stop")

    engine = AsyncMock()
    engine.generate = AsyncMock(side_effect=[tool_call_result, final_result])

    body = _make_generate_request()
    state = _make_state(settings)

    with patch("gateway.tools.loop.web_search", new_callable=AsyncMock) as mock_ws:
        mock_ws.return_value = {
            "text": "News snippets here",
            "sources": [{"url": "https://news.example.com", "title": "News"}],
            "cache_hit": False,
        }
        result = await run_with_search(body=body, engine=engine, state=state, api_key_id="k1")

    assert result.content == "Here is what I found."
    assert len(result.sources) == 1
    assert engine.generate.call_count == 2
    mock_ws.assert_called_once()


@pytest.mark.asyncio
async def test_loop_stops_after_max_iterations(settings) -> None:
    """Loop should not exceed search_max_iterations."""
    from gateway.tools.loop import run_with_search

    settings.search_max_iterations = 2

    always_tool_call = _make_engine_result(
        content="",
        finish_reason="tool_calls",
        tool_calls=[{"name": "web_search", "arguments": {"query": "keep searching"}}],
    )
    final_answer = _make_engine_result("Done.", "stop")

    # 2 tool-call results + 1 final generate (after loop exhausted)
    engine = AsyncMock()
    engine.generate = AsyncMock(side_effect=[always_tool_call, always_tool_call, final_answer])

    body = _make_generate_request()
    state = _make_state(settings)

    with patch("gateway.tools.loop.web_search", new_callable=AsyncMock) as mock_ws:
        mock_ws.return_value = {"text": "results", "sources": [], "cache_hit": False}
        result = await run_with_search(body=body, engine=engine, state=state, api_key_id="k1")

    # 2 loop iterations + 1 final generate = 3 total calls
    assert engine.generate.call_count == 3
    assert result.content == "Done."


@pytest.mark.asyncio
async def test_enable_web_search_false_skips_loop() -> None:
    """When enable_web_search=False, generate.py uses the direct engine path, not the loop."""
    import respx as respx_mod
    from httpx import ASGITransport, AsyncClient, Response

    from gateway.main import create_app

    _AUTH = {"Authorization": "Bearer gk_test1.secret_for_tests_0123456789"}
    _OLLAMA_JSON = {
        "message": {"role": "assistant", "content": "simple answer"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 3,
        "eval_count": 2,
    }

    with respx_mod.mock:
        respx_mod.get("http://ollama.invalid/api/tags").mock(
            return_value=Response(200, json={"models": [{"name": "gemma4:e4b"}]})
        )
        respx_mod.post("http://ollama.invalid/api/chat").mock(
            return_value=Response(200, json=_OLLAMA_JSON)
        )

        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            async with app.router.lifespan_context(app):
                r = await client.post(
                    "/v1/generate",
                    json={
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                        "enable_web_search": False,
                    },
                    headers=_AUTH,
                )
                assert r.status_code == 200
                body = r.json()
                assert body["content"] == "simple answer"
                assert body["search_used"] is False
