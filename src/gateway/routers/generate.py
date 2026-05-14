from __future__ import annotations

import time
import uuid
from typing import Any

import orjson
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from gateway.auth import ApiKeyRecord, require_api_key
from gateway.errors import CircuitOpenError, EngineUnavailableError, QueueFullError, QueueTimeoutError, ValidationError
from gateway.models.requests import GenerateRequest
from gateway.models.responses import GenerateResponse, SearchSource, ToolCall
from gateway.tools import loop as search_loop

router = APIRouter(tags=["generate"])


@router.post(
    "/v1/generate",
    summary="Chat generation",
    description=(
        "Send a list of messages and receive a completion. "
        "Streams SSE tokens by default (stream=true). "
        "Supports optional web search (enable_web_search), function tools, "
        "structured JSON output (response_format / json_schema), "
        "and priority queuing."
    ),
)
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

    # Reject prompts that wouldn't fit in the model context window. Without
    # this, Ollama silently truncates the prompt and produces a confused
    # response with no error surfaced to the caller.
    prompt_chars = sum(len(m.content) for m in body.messages)
    est_prompt_tokens = max(1, prompt_chars // settings.chars_per_token)
    budget = settings.model_num_ctx - body.max_tokens - settings.input_token_headroom
    if est_prompt_tokens > budget:
        raise ValidationError(
            f"estimated prompt tokens ({est_prompt_tokens}) exceed available context "
            f"({budget} = num_ctx {settings.model_num_ctx} - max_tokens {body.max_tokens} "
            f"- headroom {settings.input_token_headroom})"
        )

    model = body.model or settings.default_model
    req_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
    is_high = body.priority == "high"

    breaker = state.circuit.for_key(engine=state.engine.name, model=model)
    if not breaker.allow(is_high_priority=is_high):
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
            collected: list[str] = []
            try:
                async for piece in engine.stream(
                    model=model,
                    messages=messages_dicts,
                    temperature=body.temperature,
                    max_tokens=body.max_tokens,
                    stop=body.stop,
                ):
                    collected.append(piece)
                    yield b"data: " + orjson.dumps({"delta": piece, "done": False, "request_id": req_id}) + b"\n\n"
                full_content = "".join(collected)
                yield b"data: " + orjson.dumps({
                    "delta": "",
                    "done": True,
                    "content": full_content,
                    "model": model,
                    "request_id": req_id,
                }) + b"\n\n"
                breaker.record_success()
            except Exception as exc:
                breaker.record_failure()
                logger.exception("stream error: {}", exc)
                yield b"data: " + orjson.dumps({"error": str(exc), "request_id": req_id}) + b"\n\n"

        return StreamingResponse(
            _sse(),
            media_type="text/event-stream",
            headers={"X-Request-Id": req_id, "Cache-Control": "no-cache"},
        )

    if body.enable_web_search:
        async def _search_call():
            return await search_loop.run_with_search(
                body=body,
                engine=engine,
                state=state,
                api_key_id=principal.key_id,
            )

        t0 = time.monotonic()
        try:
            loop_result = await state.queue.submit(
                _search_call, priority=body.priority, max_wait_ms=body.timeout_ms
            )
        except (QueueFullError, QueueTimeoutError):
            # Queue pressure is not an engine fault — don't penalise the circuit.
            raise
        except EngineUnavailableError:
            breaker.record_failure()
            raise
        except Exception:
            breaker.record_failure()
            raise
        breaker.record_success()
        total_ms = (time.monotonic() - t0) * 1000
        queue_wait_ms = (time.monotonic() - enqueue_start - (total_ms / 1000)) * 1000
        if queue_wait_ms < 0:
            queue_wait_ms = 0.0

        state.metrics.record(
            endpoint="/v1/generate",
            status=200,
            latency_ms=total_ms,
            tokens_in=loop_result.tokens_in,
            tokens_out=loop_result.tokens_out,
            model=model,
            workflow=body.workflow,
            api_key_id=principal.key_id,
        )

        return GenerateResponse(
            model=model,
            content=loop_result.content,
            finish_reason=loop_result.finish_reason,  # type: ignore[arg-type]
            tokens_in=loop_result.tokens_in,
            tokens_out=loop_result.tokens_out,
            latency_ms=total_ms,
            request_id=req_id,
            queue_wait_ms=queue_wait_ms,
            sources=[SearchSource(url=s["url"], title=s["title"]) for s in loop_result.sources],
            search_used=True,
            search_cache_hit=loop_result.cache_hit,
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
    except (QueueFullError, QueueTimeoutError):
        # Queue pressure is not an engine fault — don't penalise the circuit.
        raise
    except EngineUnavailableError:
        breaker.record_failure()
        raise
    except Exception:
        breaker.record_failure()
        raise
    breaker.record_success()
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
