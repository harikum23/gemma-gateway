from __future__ import annotations

import json

import pytest
import respx
from httpx import ASGITransport, AsyncClient, Response

from gateway.main import create_app

_AUTH = {"Authorization": "Bearer gk_test1.secret_for_tests_0123456789"}

_OLLAMA_CHAT_JSON = {
    "message": {"role": "assistant", "content": "hi"},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 5,
    "eval_count": 3,
}

# Streaming response: two token lines then a done line
_OLLAMA_STREAM_BODY = (
    b'{"message":{"role":"assistant","content":"hel"},"done":false}\n'
    b'{"message":{"role":"assistant","content":"lo"},"done":false}\n'
    b'{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":4,"eval_count":2}\n'
)


@pytest.mark.asyncio
@respx.mock
async def test_generate_happy_path() -> None:
    respx.get("http://ollama.invalid/api/tags").mock(
        return_value=Response(200, json={"models": [{"name": "gemma4:e4b"}]})
    )
    respx.post("http://ollama.invalid/api/chat").mock(
        return_value=Response(200, json=_OLLAMA_CHAT_JSON)
    )

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            r = await client.post(
                "/v1/generate",
                json={
                    "model": "gemma4:e4b",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                    "stream": False,
                },
                headers=_AUTH,
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["content"] == "hi"
            assert body["tokens_in"] == 5
            assert body["tokens_out"] == 3


@pytest.mark.asyncio
@respx.mock
async def test_generate_stream_false_returns_json() -> None:
    """stream=false must return a normal JSON response, not SSE."""
    respx.get("http://ollama.invalid/api/tags").mock(
        return_value=Response(200, json={"models": [{"name": "gemma4:e4b"}]})
    )
    respx.post("http://ollama.invalid/api/chat").mock(
        return_value=Response(200, json=_OLLAMA_CHAT_JSON)
    )

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            r = await client.post(
                "/v1/generate",
                json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
                headers=_AUTH,
            )
            assert r.status_code == 200, r.text
            assert "application/json" in r.headers["content-type"]
            body = r.json()
            assert body["content"] == "hi"


@pytest.mark.asyncio
@respx.mock
async def test_generate_stream_true_returns_event_stream() -> None:
    """stream=true (the default) must return text/event-stream content-type."""
    respx.get("http://ollama.invalid/api/tags").mock(
        return_value=Response(200, json={"models": [{"name": "gemma4:e4b"}]})
    )
    respx.post("http://ollama.invalid/api/chat").mock(
        return_value=Response(200, content=_OLLAMA_STREAM_BODY)
    )

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            r = await client.post(
                "/v1/generate",
                json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
                headers=_AUTH,
            )
            assert r.status_code == 200, r.text
            assert "text/event-stream" in r.headers["content-type"]


@pytest.mark.asyncio
@respx.mock
async def test_generate_stream_sse_events_well_formed() -> None:
    """SSE events must be valid JSON with expected shape; final event has done=true and content."""
    respx.get("http://ollama.invalid/api/tags").mock(
        return_value=Response(200, json={"models": [{"name": "gemma4:e4b"}]})
    )
    respx.post("http://ollama.invalid/api/chat").mock(
        return_value=Response(200, content=_OLLAMA_STREAM_BODY)
    )

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            r = await client.post(
                "/v1/generate",
                json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
                headers=_AUTH,
            )
            assert r.status_code == 200, r.text

            raw = r.text
            events = []
            for line in raw.splitlines():
                line = line.strip()
                if line.startswith("data: "):
                    payload = json.loads(line[len("data: "):])
                    events.append(payload)

            assert len(events) >= 1, "expected at least one SSE event"

            # All non-final events have delta and done=false
            for evt in events[:-1]:
                assert "delta" in evt
                assert evt["done"] is False

            # Final event has done=true and full content
            final = events[-1]
            assert final["done"] is True
            assert "content" in final
            assert final["content"] == "hello"  # "hel" + "lo"
            assert final["delta"] == ""


@pytest.mark.asyncio
@respx.mock
async def test_generate_default_is_streaming() -> None:
    """Omitting stream field defaults to streaming (text/event-stream)."""
    respx.get("http://ollama.invalid/api/tags").mock(
        return_value=Response(200, json={"models": [{"name": "gemma4:e4b"}]})
    )
    respx.post("http://ollama.invalid/api/chat").mock(
        return_value=Response(200, content=_OLLAMA_STREAM_BODY)
    )

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            r = await client.post(
                "/v1/generate",
                json={"messages": [{"role": "user", "content": "hello"}]},
                headers=_AUTH,
            )
            assert r.status_code == 200, r.text
            assert "text/event-stream" in r.headers["content-type"]


@pytest.mark.asyncio
@respx.mock
async def test_generate_rejects_without_auth() -> None:
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            r = await client.post(
                "/v1/generate",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            assert r.status_code == 401


@pytest.mark.asyncio
@respx.mock
async def test_generate_validates_max_tokens() -> None:
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            r = await client.post(
                "/v1/generate",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 999_999,
                },
                headers=_AUTH,
            )
            assert r.status_code == 400
            assert r.json()["detail"]["code"] == "invalid_request"
