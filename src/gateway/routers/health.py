from __future__ import annotations

from fastapi import APIRouter, Request

from gateway.models.responses import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/v1/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    state = request.app.state
    engine = state.engine
    ready = await engine.health()
    models = [m["id"] for m in (await engine.list_models() if ready else [])]
    circuit_state = state.circuit.state()
    queue_depth = state.queue.depth()
    overall = "ok" if ready and circuit_state == "closed" else ("degraded" if ready else "down")
    return HealthResponse(
        status=overall,
        engine=engine.name,
        engine_url=state.settings.engine_url,
        engine_ready=ready,
        queue_depth=queue_depth,
        queue_max=state.queue.max_depth,
        circuit=circuit_state,
        models=models,
    )
