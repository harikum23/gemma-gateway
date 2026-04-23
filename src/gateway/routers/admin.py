from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from gateway.auth import ApiKeyRecord, require_admin_key, require_api_key
from gateway.models.responses import ModelListResponse

router = APIRouter(tags=["admin"])


class CreateKeyRequest(BaseModel):
    label: str = ""
    is_admin: bool = False


class CreateKeyResponse(BaseModel):
    key: str = Field(..., description="Raw API key — shown only once. Store now.")


class KeyListResponse(BaseModel):
    keys: list[dict[str, Any]]


class RevokeKeyResponse(BaseModel):
    revoked: bool


class MetricsSummaryResponse(BaseModel):
    summary: dict[str, Any]


@router.get(
    "/v1/models",
    response_model=ModelListResponse,
    summary="List models",
    description="Returns all model IDs currently loaded in the inference engine.",
)
async def list_models(
    request: Request,
    _: ApiKeyRecord = Depends(require_api_key),
) -> ModelListResponse:
    return ModelListResponse(models=await request.app.state.engine.list_models())


@router.post(
    "/v1/admin/keys",
    response_model=CreateKeyResponse,
    summary="Create API key",
    description="Generate a new API key. The raw key is returned once — store it immediately. Requires admin key.",
)
async def create_key(
    request: Request,
    body: CreateKeyRequest,
    _: ApiKeyRecord = Depends(require_admin_key),
) -> CreateKeyResponse:
    raw = request.app.state.api_keys.generate(label=body.label, is_admin=body.is_admin)
    return CreateKeyResponse(key=raw)


@router.get(
    "/v1/admin/keys",
    response_model=KeyListResponse,
    summary="List API keys",
    description="List all API keys with metadata (key_id, label, is_admin, dates, revocation status). Hashes are never returned. Requires admin key.",
)
async def list_keys(
    request: Request,
    _: ApiKeyRecord = Depends(require_admin_key),
) -> KeyListResponse:
    return KeyListResponse(keys=request.app.state.api_keys.list_keys())


@router.delete("/v1/admin/keys/{key_id}", response_model=RevokeKeyResponse)
async def revoke_key(
    key_id: str,
    request: Request,
    _: ApiKeyRecord = Depends(require_admin_key),
) -> RevokeKeyResponse:
    ok = request.app.state.api_keys.revoke(key_id)
    return RevokeKeyResponse(revoked=ok)


@router.get("/v1/admin/metrics", response_model=MetricsSummaryResponse)
async def metrics_summary(
    request: Request,
    _: ApiKeyRecord = Depends(require_admin_key),
) -> MetricsSummaryResponse:
    return MetricsSummaryResponse(summary=request.app.state.metrics.summary())
