from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from gateway.models.requests import AgentOptions, AgentRequest, Message
from gateway.models.responses import AgentResponse
from gateway.settings import get_settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine_result(content: str = "done", finish_reason: str = "stop", tool_calls=None):
    r = MagicMock()
    r.content = content
    r.finish_reason = finish_reason
    r.tool_calls = tool_calls
    r.tokens_in = 10
    r.tokens_out = 5
    return r


def _make_state(settings=None, redis=None, trace_db=None):
    state = MagicMock()
    state.settings = settings or get_settings()
    state.redis = redis
    state.trace_db = trace_db or _make_trace_db()
    state._agent_key_id = "test_key"
    return state


def _make_trace_db():
    db = MagicMock()
    db.execute = MagicMock(return_value=MagicMock(fetchall=MagicMock(return_value=[])))
    db.commit = MagicMock()
    return db


def _agent_request(
    messages=None,
    builtin_tools=None,
    custom_tools=None,
    max_steps=4,
    total_budget_ms=30000,
    return_trace=False,
    no_store=False,
):
    return AgentRequest(
        model="gemma4:e4b",
        messages=messages or [Message(role="user", content="Hello")],
        agent=AgentOptions(
            builtin_tools=builtin_tools or [],
            custom_tools=custom_tools or [],
            max_steps=max_steps,
            total_budget_ms=total_budget_ms,
            return_trace=return_trace,
            no_store=no_store,
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_no_tools_returns_stop():
    """Basic agent call with no tools returns stop finish_reason."""
    from gateway.agent.runtime import run_agent

    engine = AsyncMock()
    engine.generate = AsyncMock(return_value=_make_engine_result("Final answer", "stop"))
    state = _make_state()

    req = _agent_request()
    response = await run_agent(req, engine, state)

    assert isinstance(response, AgentResponse)
    assert response.finish_reason == "stop"
    assert response.content == "Final answer"
    assert response.pending_tool_call is None


@pytest.mark.asyncio
async def test_agent_calls_builtin_tool_and_continues():
    """Agent calls fetch_url built-in, loop continues, returns final answer."""
    from gateway.agent.runtime import run_agent

    # First call: tool_call; second call: stop
    tool_call_result = _make_engine_result(
        content="",
        finish_reason="tool_calls",
        tool_calls=[{"name": "fetch_url", "arguments": {"url": "http://example.com"}}],
    )
    final_result = _make_engine_result("Here is the answer.", "stop")

    engine = AsyncMock()
    engine.generate = AsyncMock(side_effect=[tool_call_result, final_result])
    state = _make_state()

    with patch("gateway.tools.builtin.fetch_url.run", new=AsyncMock(return_value="Page content")):
        req = _agent_request(builtin_tools=["fetch_url"])
        response = await run_agent(req, engine, state)

    assert response.finish_reason == "stop"
    assert response.content == "Here is the answer."
    assert engine.generate.call_count == 2


@pytest.mark.asyncio
async def test_agent_max_steps_ceiling_enforced():
    """mock engine always returns tool_call — loop must stop at max_steps."""
    from gateway.agent.runtime import run_agent

    # Engine always wants to call a tool that doesn't exist in the builtin list
    # so it gets an "unknown tool" response but keep looping until max_steps
    perpetual_tool_call = _make_engine_result(
        content="",
        finish_reason="tool_calls",
        tool_calls=[{"name": "unknown_tool", "arguments": {}}],
    )
    engine = AsyncMock()
    engine.generate = AsyncMock(return_value=perpetual_tool_call)
    state = _make_state()

    req = _agent_request(max_steps=3)
    response = await run_agent(req, engine, state)

    assert response.finish_reason == "max_steps"
    assert engine.generate.call_count == 3


@pytest.mark.asyncio
async def test_agent_budget_ms_exhaustion():
    """Budget exhaustion stops the loop."""
    import asyncio
    from gateway.agent.runtime import run_agent

    async def slow_generate(**kwargs):
        await asyncio.sleep(0.05)  # 50ms
        return _make_engine_result(
            content="",
            finish_reason="tool_calls",
            tool_calls=[{"name": "unknown_tool", "arguments": {}}],
        )

    engine = AsyncMock()
    engine.generate = slow_generate
    state = _make_state()

    # 80ms budget — should exhaust after 1-2 steps of 50ms each
    req = _agent_request(max_steps=10, total_budget_ms=80)
    response = await run_agent(req, engine, state)

    assert response.finish_reason in ("budget_exhausted", "max_steps")


@pytest.mark.asyncio
async def test_agent_custom_tool_triggers_needs_tool():
    """Custom (consumer-owned) tool triggers needs_tool with pending_tool_call populated."""
    from gateway.agent.runtime import run_agent

    custom_tool_def = {
        "name": "my_custom_tool",
        "description": "A custom tool",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    }
    tool_call_result = _make_engine_result(
        content="",
        finish_reason="tool_calls",
        tool_calls=[{"name": "my_custom_tool", "arguments": {"q": "test"}}],
    )
    engine = AsyncMock()
    engine.generate = AsyncMock(return_value=tool_call_result)
    state = _make_state()

    req = _agent_request(custom_tools=[custom_tool_def])
    response = await run_agent(req, engine, state)

    assert response.finish_reason == "needs_tool"
    assert response.pending_tool_call is not None
    assert response.pending_tool_call["name"] == "my_custom_tool"
    assert response.pending_tool_call["args"] == {"q": "test"}


@pytest.mark.asyncio
async def test_agent_no_store_skips_trace_logging():
    """no_store=True must not call trace_db.execute for INSERT."""
    from gateway.agent.runtime import run_agent

    engine = AsyncMock()
    engine.generate = AsyncMock(return_value=_make_engine_result("Answer", "stop"))

    trace_db = _make_trace_db()
    state = _make_state(trace_db=trace_db)

    req = _agent_request(no_store=True)
    response = await run_agent(req, engine, state)

    assert response.finish_reason == "stop"
    # No INSERT should have been executed — only the table-creation execute during startup
    insert_calls = [
        c for c in trace_db.execute.call_args_list
        if c.args and "INSERT" in str(c.args[0]).upper()
    ]
    assert len(insert_calls) == 0


@pytest.mark.asyncio
async def test_agent_concurrency_cap_returns_429():
    """Per-consumer concurrency cap: 429 when limit exceeded."""
    from gateway.main import create_app
    from httpx import ASGITransport, AsyncClient, Response
    import respx

    _AUTH = {"Authorization": "Bearer gk_test1.secret_for_tests_0123456789"}

    @respx.mock
    async def _run():
        respx.get("http://ollama.invalid/api/tags").mock(
            return_value=Response(200, json={"models": [{"name": "gemma4:e4b"}]})
        )
        respx.post("http://ollama.invalid/api/chat").mock(
            return_value=Response(200, json={
                "message": {"role": "assistant", "content": "hi"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 5,
                "eval_count": 3,
            })
        )

        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            async with app.router.lifespan_context(app):
                # Simulate already-at-limit by patching the concurrency helper
                with patch("gateway.routers.agent._increment_concurrency", new=AsyncMock(return_value=99)):
                    r = await client.post(
                        "/v1/agent",
                        json={
                            "messages": [{"role": "user", "content": "hi"}],
                            "agent": {"max_steps": 1},
                        },
                        headers=_AUTH,
                    )
                assert r.status_code == 429

    await _run()


# ---------------------------------------------------------------------------
# Sources flow through from web_search to AgentResponse
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_sources_collected_from_web_search():
    """Sources returned by web_search must appear in AgentResponse.sources."""
    from gateway.agent.runtime import run_agent

    tool_call_result = _make_engine_result(
        content="",
        finish_reason="tool_calls",
        tool_calls=[{"name": "web_search", "arguments": {"query": "NVDA stock today"}}],
    )
    final_result = _make_engine_result("NVDA is up 3%.", "stop")

    engine = AsyncMock()
    engine.generate = AsyncMock(side_effect=[tool_call_result, final_result])
    state = _make_state()

    mock_search_result = {
        "text": "NVIDIA is trading at $875, up 3.2% today.",
        "sources": [
            {"url": "https://finance.yahoo.com/nvda", "title": "NVDA - Yahoo Finance"},
            {"url": "https://reuters.com/nvda", "title": "Reuters NVDA"},
        ],
        "cache_hit": False,
    }

    with patch("gateway.agent.runtime.coord_mod.clean_gemini_output", side_effect=lambda x: x), \
         patch("gateway.tools.web_search.web_search", new_callable=AsyncMock) as mock_ws:
        mock_ws.return_value = mock_search_result

        req = _agent_request(
            messages=[Message(role="user", content="How is NVDA stock today?")],
            builtin_tools=["web_search"],
            return_trace=True,
        )
        response = await run_agent(req, engine, state)

    assert response.finish_reason == "stop"
    assert len(response.sources) == 2
    urls = [s.url for s in response.sources]
    assert "https://finance.yahoo.com/nvda" in urls
    assert "https://reuters.com/nvda" in urls


@pytest.mark.asyncio
async def test_agent_sources_deduplicated():
    """If web_search is called twice with overlapping sources, deduplicate."""
    from gateway.agent.runtime import run_agent

    tc1 = _make_engine_result(
        content="", finish_reason="tool_calls",
        tool_calls=[{"name": "web_search", "arguments": {"query": "NVDA q1"}}],
    )
    tc2 = _make_engine_result(
        content="", finish_reason="tool_calls",
        tool_calls=[{"name": "web_search", "arguments": {"query": "NVDA earnings"}}],
    )
    final = _make_engine_result("Combined answer.", "stop")

    engine = AsyncMock()
    engine.generate = AsyncMock(side_effect=[tc1, tc2, final])
    state = _make_state()

    shared_source = {"url": "https://reuters.com/nvda", "title": "Reuters"}
    results = [
        {"text": "Q1 data...", "sources": [shared_source], "cache_hit": False},
        {"text": "Earnings...", "sources": [shared_source, {"url": "https://bloomberg.com", "title": "Bloomberg"}], "cache_hit": False},
    ]

    with patch("gateway.agent.runtime.coord_mod.clean_gemini_output", side_effect=lambda x: x), \
         patch("gateway.tools.web_search.web_search", new_callable=AsyncMock) as mock_ws:
        mock_ws.side_effect = results

        req = _agent_request(
            messages=[Message(role="user", content="NVDA earnings")],
            builtin_tools=["web_search"],
            max_steps=6,
        )
        response = await run_agent(req, engine, state)

    # reuters.com appears in both calls but must only be in sources once
    source_urls = [s.url for s in response.sources]
    assert source_urls.count("https://reuters.com/nvda") == 1


# ---------------------------------------------------------------------------
# Stop condition — requires BOTH finish_reason=stop AND no tool_calls
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_stop_requires_no_tool_calls():
    """
    If finish_reason=stop but tool_calls is populated (Ollama edge case),
    the agent must NOT stop — it must process the tool calls.
    """
    from gateway.agent.runtime import run_agent

    ambiguous_result = _make_engine_result(
        content="",
        finish_reason="stop",   # says stop but also has tool_calls
        tool_calls=[{"name": "web_search", "arguments": {"query": "NVDA"}}],
    )
    final_result = _make_engine_result("Final.", "stop")

    engine = AsyncMock()
    engine.generate = AsyncMock(side_effect=[ambiguous_result, final_result])
    state = _make_state()

    with patch("gateway.agent.runtime.coord_mod.clean_gemini_output", side_effect=lambda x: x), \
         patch("gateway.tools.web_search.web_search", new_callable=AsyncMock) as mock_ws:
        mock_ws.return_value = {"text": "search result", "sources": [], "cache_hit": False}

        req = _agent_request(
            messages=[Message(role="user", content="NVDA today?")],
            builtin_tools=["web_search"],
        )
        response = await run_agent(req, engine, state)

    # Should have called generate twice (tool executed, then final answer)
    assert engine.generate.call_count == 2
    assert response.content == "Final."


# ---------------------------------------------------------------------------
# Coordinator prompt tailoring
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_stock_query_gets_stock_system_prompt():
    """A stock question must inject a system prompt focused on price/volume."""
    from gateway.agent.runtime import run_agent

    engine = AsyncMock()
    engine.generate = AsyncMock(return_value=_make_engine_result("NVDA is $875.", "stop"))
    state = _make_state()

    req = _agent_request(
        messages=[Message(role="user", content="How is NVDA stock performing today?")],
        builtin_tools=["web_search"],
    )
    await run_agent(req, engine, state)

    # The first generate call's messages must include a system message
    first_call_messages = engine.generate.call_args_list[0].kwargs["messages"]
    system_msgs = [m for m in first_call_messages if m.get("role") == "system"]
    assert len(system_msgs) == 1
    sp = system_msgs[0]["content"].lower()
    # Stock prompt must mention price and volume
    assert "price" in sp
    assert "volume" in sp


@pytest.mark.asyncio
async def test_agent_general_query_gets_general_system_prompt():
    """A general (non-stock, non-news) question gets the general coordinator prompt."""
    from gateway.agent.runtime import run_agent

    engine = AsyncMock()
    engine.generate = AsyncMock(return_value=_make_engine_result("Transformers use attention.", "stop"))
    state = _make_state()

    req = _agent_request(
        messages=[Message(role="user", content="Explain transformer architecture")],
        builtin_tools=["web_search"],
    )
    await run_agent(req, engine, state)

    first_call_messages = engine.generate.call_args_list[0].kwargs["messages"]
    system_msgs = [m for m in first_call_messages if m.get("role") == "system"]
    assert len(system_msgs) == 1
    sp = system_msgs[0]["content"].lower()
    # General prompt should mention decision about whether to search
    assert "web_search" in sp


@pytest.mark.asyncio
async def test_agent_custom_system_prompt_not_overridden():
    """If the caller supplies opts.system, it must NOT be replaced by coordinator."""
    from gateway.agent.runtime import run_agent

    engine = AsyncMock()
    engine.generate = AsyncMock(return_value=_make_engine_result("ok", "stop"))
    state = _make_state()

    custom_prompt = "You are a pirate. Answer only in pirate speak."
    req = AgentRequest(
        model="gemma4:e4b",
        messages=[Message(role="user", content="Hello")],
        agent=AgentOptions(
            system=custom_prompt,
            builtin_tools=["web_search"],
            max_steps=2,
        ),
    )
    await run_agent(req, engine, state)

    first_call_messages = engine.generate.call_args_list[0].kwargs["messages"]
    system_msgs = [m for m in first_call_messages if m.get("role") == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == custom_prompt


# ---------------------------------------------------------------------------
# Gemini output cleaning
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_cleans_gemini_output_before_observation():
    """clean_gemini_output must be applied to web_search results before Gemma sees them."""
    from gateway.agent.runtime import run_agent

    tool_call_result = _make_engine_result(
        content="", finish_reason="tool_calls",
        tool_calls=[{"name": "web_search", "arguments": {"query": "NVDA"}}],
    )
    final_result = _make_engine_result("NVDA is $875.", "stop")

    engine = AsyncMock()
    engine.generate = AsyncMock(side_effect=[tool_call_result, final_result])
    state = _make_state()

    dirty_text = "Here is a summary of NVIDIA's performance: NVDA is up 3%."

    with patch("gateway.agent.runtime.coord_mod.clean_gemini_output") as mock_clean, \
         patch("gateway.tools.web_search.web_search", new_callable=AsyncMock) as mock_ws:
        mock_clean.side_effect = lambda x: x.replace("Here is a summary of NVIDIA's performance: ", "")
        mock_ws.return_value = {"text": dirty_text, "sources": [], "cache_hit": False}

        req = _agent_request(
            messages=[Message(role="user", content="NVDA today?")],
            builtin_tools=["web_search"],
        )
        await run_agent(req, engine, state)

    mock_clean.assert_called_once_with(dirty_text)

    # Second generate call's tool message must contain cleaned text
    second_call_messages = engine.generate.call_args_list[1].kwargs["messages"]
    tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "Here is a summary" not in tool_msgs[0]["content"]
    assert "NVDA is up 3%" in tool_msgs[0]["content"]


# ---------------------------------------------------------------------------
# tool_call_id round-trip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_tool_result_has_matching_tool_call_id():
    """tool result message must carry a tool_call_id matching the assistant's tool_call."""
    from gateway.agent.runtime import run_agent

    tool_call_result = _make_engine_result(
        content="", finish_reason="tool_calls",
        tool_calls=[{"name": "web_search", "arguments": {"query": "NVDA"}}],
    )
    final_result = _make_engine_result("Done.", "stop")

    engine = AsyncMock()
    engine.generate = AsyncMock(side_effect=[tool_call_result, final_result])
    state = _make_state()

    with patch("gateway.agent.runtime.coord_mod.clean_gemini_output", side_effect=lambda x: x), \
         patch("gateway.tools.web_search.web_search", new_callable=AsyncMock) as mock_ws:
        mock_ws.return_value = {"text": "result", "sources": [], "cache_hit": False}

        req = _agent_request(
            messages=[Message(role="user", content="NVDA?")],
            builtin_tools=["web_search"],
        )
        await run_agent(req, engine, state)

    second_call_messages = engine.generate.call_args_list[1].kwargs["messages"]

    # Find assistant message with tool_calls
    assistant_msg = next(m for m in second_call_messages if m.get("role") == "assistant" and m.get("tool_calls"))
    tool_msg = next(m for m in second_call_messages if m.get("role") == "tool")

    assistant_tc_id = assistant_msg["tool_calls"][0]["id"]
    tool_result_id = tool_msg.get("tool_call_id")

    assert assistant_tc_id == tool_result_id
    assert len(assistant_tc_id) > 0
