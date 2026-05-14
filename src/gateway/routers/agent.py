from __future__ import annotations

import time
import uuid
from typing import Any

import orjson
from fastapi import APIRouter, Depends, Request
from loguru import logger
from pydantic import BaseModel

from gateway.agent import runtime as agent_runtime
from gateway.agent import trace as trace_mod
from gateway.auth import ApiKeyRecord, require_api_key
from gateway.errors import CircuitOpenError, EngineUnavailableError, ValidationError
from gateway.models.requests import AgentRequest
from gateway.models.responses import AgentResponse

router = APIRouter(tags=["agent"])

_CONCURRENCY_KEY_PREFIX = "agent_concurrency:"


async def _increment_concurrency(redis_client: Any, key_id: str) -> int:
    """INCR and return new value. Returns 1 if redis unavailable (no cap)."""
    if redis_client is None:
        return 1
    try:
        val = await redis_client.incr(f"{_CONCURRENCY_KEY_PREFIX}{key_id}")
        # Auto-expire the counter after 5 minutes to prevent stale locks
        await redis_client.expire(f"{_CONCURRENCY_KEY_PREFIX}{key_id}", 300)
        return int(val)
    except Exception as exc:
        logger.warning("agent concurrency INCR failed: {}", exc)
        return 1


async def _decrement_concurrency(redis_client: Any, key_id: str) -> None:
    if redis_client is None:
        return
    try:
        val = await redis_client.decr(f"{_CONCURRENCY_KEY_PREFIX}{key_id}")
        if val < 0:
            await redis_client.set(f"{_CONCURRENCY_KEY_PREFIX}{key_id}", 0)
    except Exception as exc:
        logger.warning("agent concurrency DECR failed: {}", exc)


@router.post(
    "/v1/agent",
    response_model=AgentResponse,
    summary="ReAct agent",
    description=(
        "Run a domain-agnostic ReAct agent loop. Built-in tools include "
        "web_search, fetch_url, clean_text, dedupe, validate_schema, summarize, "
        "extract_entities, calculator, date_parse, and translate. "
        "Custom domain tools are supported via a pause/resume round-trip."
    ),
)
async def run_agent(
    request: Request,
    body: AgentRequest,
    principal: ApiKeyRecord = Depends(require_api_key),
) -> Any:
    state = request.app.state
    settings = state.settings

    # Validate max_steps against ceiling
    ceiling = getattr(settings, "agent_max_steps_ceiling", 10)
    if body.agent.max_steps > ceiling:
        raise ValidationError(f"agent.max_steps must be <= {ceiling}")

    agent_model = body.model or settings.default_model
    breaker = state.circuit.for_key(engine=state.engine.name, model=agent_model)
    if not breaker.allow(is_high_priority=False):
        raise CircuitOpenError()

    state.rate_limiter.check(principal.key_id)

    # Per-key concurrency cap
    max_concurrency = getattr(settings, "agent_max_concurrency_per_key", 2)
    redis_client = getattr(state, "redis", None)
    current = await _increment_concurrency(redis_client, principal.key_id)
    if current > max_concurrency:
        await _decrement_concurrency(redis_client, principal.key_id)
        from gateway.errors import GatewayError
        raise GatewayError(429, f"agent concurrency limit ({max_concurrency}) exceeded for this key", code="concurrency_limit")

    # Stash key_id on state temporarily so runtime can read it for traces
    state._agent_key_id = principal.key_id

    t0 = time.monotonic()
    try:
        async def _run():
            return await agent_runtime.run_agent(body, state.engine, state)

        response = await state.queue.submit(_run, priority="normal", max_wait_ms=body.agent.total_budget_ms)
        breaker.record_success()
    except EngineUnavailableError:
        breaker.record_failure()
        raise
    except Exception:
        breaker.record_failure()
        raise
    finally:
        await _decrement_concurrency(redis_client, principal.key_id)

    total_ms = (time.monotonic() - t0) * 1000
    state.metrics.record(
        endpoint="/v1/agent",
        status=200,
        latency_ms=total_ms,
        tokens_in=response.tokens_in,
        tokens_out=response.tokens_out,
        model=body.model or settings.default_model,
        workflow=None,
        api_key_id=principal.key_id,
    )

    return response


class ToolResultBody(BaseModel):
    request_id: str
    tool_name: str
    result: str


@router.post(
    "/v1/agent/tool_result",
    response_model=AgentResponse,
    summary="Resume paused agent with tool result",
    description=(
        "Post back the result of a custom domain tool to resume an agent that "
        "paused waiting for it. Requires Redis. The request_id is obtained from "
        "the paused agent response."
    ),
)
async def submit_tool_result(
    request: Request,
    body: ToolResultBody,
    principal: ApiKeyRecord = Depends(require_api_key),
) -> Any:
    """Consumer posts back the result of a custom tool execution.

    The paused agent state is reconstructed from Redis.  If the state is not
    found (e.g. it expired) a 404 is returned.
    """
    state = request.app.state
    redis_client = getattr(state, "redis", None)

    if redis_client is None:
        from gateway.errors import GatewayError
        raise GatewayError(503, "tool_result endpoint requires Redis", code="redis_unavailable")

    redis_key = f"agent_pause:{body.request_id}"
    raw = await redis_client.get(redis_key)
    if raw is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "paused agent state not found or expired"})

    paused_state = orjson.loads(raw)
    await redis_client.delete(redis_key)

    # Re-hydrate the AgentRequest from stored JSON
    agent_request = AgentRequest.model_validate(paused_state["request"])

    # Inject the consumer tool result into messages and continue
    messages = paused_state["messages"]
    messages.append({
        "role": "tool",
        "name": body.tool_name,
        "content": body.result,
    })

    # Reconstruct a minimal continuation request with updated messages
    from gateway.models.requests import Message
    agent_request.messages = [Message(**m) for m in messages if m.get("role") in ("user", "assistant", "tool", "system")]

    state._agent_key_id = principal.key_id

    resume_model = agent_request.model or state.settings.default_model
    breaker = state.circuit.for_key(engine=state.engine.name, model=resume_model)
    if not breaker.allow(is_high_priority=False):
        raise CircuitOpenError()

    max_concurrency = getattr(state.settings, "agent_max_concurrency_per_key", 2)
    current = await _increment_concurrency(redis_client, principal.key_id)
    if current > max_concurrency:
        await _decrement_concurrency(redis_client, principal.key_id)
        from gateway.errors import GatewayError
        raise GatewayError(429, f"agent concurrency limit ({max_concurrency}) exceeded", code="concurrency_limit")

    try:
        async def _run():
            return await agent_runtime.run_agent(agent_request, state.engine, state)

        response = await state.queue.submit(_run, priority="normal", max_wait_ms=agent_request.agent.total_budget_ms)
        breaker.record_success()
    except EngineUnavailableError:
        breaker.record_failure()
        raise
    except Exception:
        breaker.record_failure()
        raise
    finally:
        await _decrement_concurrency(redis_client, principal.key_id)

    return response
