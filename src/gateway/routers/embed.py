from __future__ import annotations

import math
import time
import uuid

from fastapi import APIRouter, Depends, Request

from gateway.auth import ApiKeyRecord, require_api_key
from gateway.errors import EngineUnavailableError, ValidationError
from gateway.models.requests import EmbedRequest
from gateway.models.responses import EmbedResponse

router = APIRouter(tags=["embed"])

_MAX_BATCH = 128
_MAX_INPUT_CHARS = 32_000


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


@router.post(
    "/v1/embed",
    response_model=EmbedResponse,
    summary="Batch embeddings",
    description=(
        "Compute embeddings for a batch of strings (max 128 per request, "
        "each up to 32 000 characters). Returns float vectors and their "
        "dimensionality. Vectors are L2-normalized by default."
    ),
)
async def embed(
    request: Request,
    body: EmbedRequest,
    principal: ApiKeyRecord = Depends(require_api_key),
) -> EmbedResponse:
    state = request.app.state
    settings = state.settings

    if len(body.inputs) > _MAX_BATCH:
        raise ValidationError(f"inputs batch must be <= {_MAX_BATCH}")
    for s in body.inputs:
        if len(s) > _MAX_INPUT_CHARS:
            raise ValidationError(f"each input must be <= {_MAX_INPUT_CHARS} chars")

    model = body.model or settings.default_embed_model
    req_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex

    state.rate_limiter.check(principal.key_id)

    t0 = time.monotonic()
    try:
        result = await state.engine.embed(model=model, inputs=body.inputs)
    except EngineUnavailableError:
        state.circuit.record_failure()
        raise
    state.circuit.record_success()
    latency_ms = (time.monotonic() - t0) * 1000

    vectors = [_normalize(v) for v in result.vectors] if body.normalize else result.vectors

    state.metrics.record(
        endpoint="/v1/embed",
        status=200,
        latency_ms=latency_ms,
        tokens_in=result.tokens_in,
        model=model,
        api_key_id=principal.key_id,
    )

    return EmbedResponse(
        model=model,
        dims=result.dims,
        vectors=vectors,
        tokens_in=result.tokens_in,
        latency_ms=latency_ms,
        request_id=req_id,
    )
