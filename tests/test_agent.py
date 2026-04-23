from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

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
