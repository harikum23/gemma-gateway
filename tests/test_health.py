from __future__ import annotations

import pytest
import respx
from httpx import ASGITransport, AsyncClient, Response

from gateway.main import create_app


@pytest.mark.asyncio
@respx.mock
async def test_health_reports_engine_ready() -> None:
    respx.get("http://ollama.invalid/api/tags").mock(
        return_value=Response(200, json={"models": [{"name": "gemma4:e4b", "size": 1}]})
    )
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            r = await client.get("/v1/health")
            assert r.status_code == 200
            data = r.json()
            assert data["engine"] == "ollama"
            assert data["engine_ready"] is True
            assert "gemma4:e4b" in data["models"]


@pytest.mark.asyncio
@respx.mock
async def test_health_reports_engine_down() -> None:
    respx.get("http://ollama.invalid/api/tags").mock(return_value=Response(500))
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            r = await client.get("/v1/health")
            assert r.status_code == 200
            data = r.json()
            assert data["engine_ready"] is False
            assert data["status"] == "down"
