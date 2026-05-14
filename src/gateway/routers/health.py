from __future__ import annotations

from fastapi import APIRouter, Request

from gateway.models.responses import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/v1/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Returns engine readiness, loaded model IDs, admission queue depth, "
        "and circuit breaker state. No authentication required."
    ),
)
async def health(request: Request) -> HealthResponse:
    state = request.app.state
    engine = state.engine
    settings = state.settings
    ready = await engine.health()
    models = [m["id"] for m in (await engine.list_models() if ready else [])]
    # Degraded if the default model isn't loaded yet (e.g. still pulling).
    default_model = settings.default_model
    model_ready = ready and any(m == default_model or m.startswith(default_model.split(":")[0]) for m in models)
    circuit_state = state.circuit.state()
    queue_depth = state.queue.depth()
    circuit_snapshot = state.circuit.snapshot()

    redis_configured = bool(settings.redis_url)
    redis_connected = False
    if redis_configured and state.redis is not None:
        try:
            await state.redis.ping()
            redis_connected = True
        except Exception:
            redis_connected = False

    redis_ok = (not redis_configured) or redis_connected or not getattr(state, "redis_required", False)
    overall = (
        "ok" if model_ready and circuit_state == "closed" and redis_ok
        else ("degraded" if ready else "down")
    )
    return HealthResponse(
        status=overall,
        engine=engine.name,
        engine_url=settings.engine_url,
        engine_ready=ready,
        queue_depth=queue_depth,
        queue_max=state.queue.max_depth,
        circuit=circuit_state,
        models=models,
        redis_configured=redis_configured,
        redis_connected=redis_connected,
        circuits=circuit_snapshot,
    )
