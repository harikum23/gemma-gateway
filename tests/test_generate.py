from __future__ import annotations

import pytest
import respx
from httpx import ASGITransport, AsyncClient, Response

from gateway.main import create_app

_AUTH = {"Authorization": "Bearer gk_test1.secret_for_tests_0123456789"}


@pytest.mark.asyncio
@respx.mock
async def test_generate_happy_path() -> None:
    respx.get("http://ollama.invalid/api/tags").mock(
        return_value=Response(200, json={"models": [{"name": "gemma4:e4b"}]})
    )
    respx.post("http://ollama.invalid/api/chat").mock(
        return_value=Response(
            200,
            json={
                "message": {"role": "assistant", "content": "hi"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 5,
                "eval_count": 3,
            },
        )
    )

    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            r = await client.post(
                "/v1/generate",
                json={
                    "model": "gemma4:e4b",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                },
                headers=_AUTH,
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["content"] == "hi"
            assert body["tokens_in"] == 5
            assert body["tokens_out"] == 3


@pytest.mark.asyncio
@respx.mock
async def test_generate_rejects_without_auth() -> None:
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            r = await client.post(
                "/v1/generate",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            assert r.status_code == 401


@pytest.mark.asyncio
@respx.mock
async def test_generate_validates_max_tokens() -> None:
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            r = await client.post(
                "/v1/generate",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 999_999,
                },
                headers=_AUTH,
            )
            assert r.status_code == 400
            assert r.json()["detail"]["code"] == "invalid_request"
