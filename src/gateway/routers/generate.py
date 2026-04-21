from __future__ import annotations

import time
import uuid
from typing import Any

import orjson
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from gateway.auth import ApiKeyRecord, require_api_key
from gateway.errors import CircuitOpenError, EngineUnavailableError, ValidationError
from gateway.models.requests import GenerateRequest
from gateway.models.responses import GenerateResponse, ToolCall

router = APIRouter(tags=["generate"])


@router.post("/v1/generate")
async def generate(
    request: Request,
    body: GenerateRequest,
    principal: ApiKeyRecord = Depends(require_api_key),
) -> Any:
    state = request.app.state
    settings = state.settings

    if body.timeout_ms is not None and body.timeout_ms > settings.max_timeout_ms:
        raise ValidationError(f"timeout_ms must be <= {settings.max_timeout_ms}")
    if body.max_tokens > settings.max_tokens_hard_limit:
        raise ValidationError(f"max_tokens must be <= {settings.max_tokens_hard_limit}")

    model = body.model or settings.default_model
    req_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
    is_high = body.priority == "high"

    if not state.circuit.allow(is_high_priority=is_high):
        raise CircuitOpenError()

    state.rate_limiter.check(principal.key_id)

    engine = state.engine
    messages_dicts = [m.model_dump(exclude_none=True) for m in body.messages]
    tools_dicts: list[dict[str, Any]] | None = (
        [t.model_dump(exclude_none=True) for t in body.tools] if body.tools else None
    )

    enqueue_start = time.monotonic()

    if body.stream:
        async def _sse():
            try:
                first = True
                async for piece in engine.stream(
                    model=model,
                    messages=messages_dicts,
                    temperature=body.temperature,
                    max_tokens=body.max_tokens,
                    stop=body.stop,
                ):
                    if first:
                        first = False
                    yield b"data: " + orjson.dumps({"delta": piece, "request_id": req_id}) + b"\n\n"
                yield b"data: " + orjson.dumps({"done": True, "request_id": req_id}) + b"\n\n"
                state.circuit.record_success()
            except Exception as exc:
                state.circuit.record_failure()
                logger.exception("stream error: {}", exc)
                yield b"data: " + orjson.dumps({"error": str(exc), "request_id": req_id}) + b"\n\n"

        return StreamingResponse(
            _sse(),
            media_type="text/event-stream",
            headers={"X-Request-Id": req_id, "Cache-Control": "no-cache"},
        )

    async def _call():
        return await engine.generate(
            model=model,
            messages=messages_dicts,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            stop=body.stop,
            tools=tools_dicts,
            response_format=body.response_format,
            json_schema=body.json_schema,
        )

    t0 = time.monotonic()
    try:
        result = await state.queue.submit(
            _call, priority=body.priority, max_wait_ms=body.timeout_ms
        )
    except EngineUnavailableError:
        state.circuit.record_failure()
        raise
    except Exception:
        state.circuit.record_failure()
        raise
    state.circuit.record_success()
    total_ms = (time.monotonic() - t0) * 1000
    queue_wait_ms = (time.monotonic() - enqueue_start - (total_ms / 1000)) * 1000
    if queue_wait_ms < 0:
        queue_wait_ms = 0.0

    state.metrics.record(
        endpoint="/v1/generate",
        status=200,
        latency_ms=total_ms,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        model=model,
        workflow=body.workflow,
        api_key_id=principal.key_id,
    )

    tool_calls = (
        [ToolCall(name=tc["name"], arguments=tc.get("arguments", {})) for tc in result.tool_calls]
        if result.tool_calls
        else None
    )

    response = GenerateResponse(
        model=model,
        content=result.content,
        tool_calls=tool_calls,
        finish_reason=result.finish_reason,  # type: ignore[arg-type]
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        latency_ms=total_ms,
        request_id=req_id,
        queue_wait_ms=queue_wait_ms,
    )
    return response
