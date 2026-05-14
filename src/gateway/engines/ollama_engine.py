from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from loguru import logger

from gateway.engines.base import EngineEmbedResult, EngineGenerateResult
from gateway.errors import EngineUnavailableError


class OllamaEngine:
    name = "ollama"

    def __init__(self, base_url: str, *, timeout_s: float = 120.0, num_ctx: int | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_s)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        try:
            r = await self._client.get("/api/tags", timeout=3.0)
            return r.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            r = await self._client.get("/api/tags", timeout=5.0)
            r.raise_for_status()
            data = r.json()
            return [
                {"id": m["name"], "size": m.get("size"), "modified_at": m.get("modified_at")}
                for m in data.get("models", [])
            ]
        except httpx.HTTPError as e:
            logger.warning("ollama list_models failed: {}", e)
            return []

    def _payload(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
        response_format: str,
        json_schema: dict[str, Any] | None,
        stream: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if stop:
            options["stop"] = stop
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "options": options,
            "stream": stream,
        }
        if response_format == "json" and json_schema is not None:
            payload["format"] = json_schema
        elif response_format == "json":
            payload["format"] = "json"
        if tools:
            payload["tools"] = [
                {"type": "function", "function": t} if "type" not in t else t for t in tools
            ]
        return payload

    async def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
        tools: list[dict[str, Any]] | None,
        response_format: str,
        json_schema: dict[str, Any] | None,
    ) -> EngineGenerateResult:
        payload = self._payload(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            response_format=response_format,
            json_schema=json_schema,
            stream=False,
            tools=tools,
        )
        try:
            r = await self._client.post("/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as e:
            raise EngineUnavailableError(f"ollama request failed: {e}") from e

        msg = data.get("message", {}) or {}
        content = msg.get("content", "")
        tool_calls_raw = msg.get("tool_calls") or []
        tool_calls = None
        if tool_calls_raw:
            tool_calls = [
                {
                    "name": tc.get("function", {}).get("name") or tc.get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments") or tc.get("arguments", {}),
                }
                for tc in tool_calls_raw
            ]
        return EngineGenerateResult(
            content=content,
            tokens_in=int(data.get("prompt_eval_count", 0) or 0),
            tokens_out=int(data.get("eval_count", 0) or 0),
            finish_reason="tool_calls" if tool_calls else ("length" if data.get("done_reason") == "length" else "stop"),
            tool_calls=tool_calls,
        )

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
    ) -> AsyncIterator[str]:
        payload = self._payload(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            response_format="text",
            json_schema=None,
            stream=True,
        )
        async with self._client.stream("POST", "/api/chat", json=payload) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode(errors="replace")
                raise EngineUnavailableError(f"ollama stream failed {r.status_code}: {body[:200]}")
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = (evt.get("message") or {}).get("content", "")
                if piece:
                    yield piece
                if evt.get("done"):
                    break

    async def _embed_one(self, model: str, text: str) -> tuple[list[float], int]:
        try:
            r = await self._client.post("/api/embeddings", json={"model": model, "prompt": text})
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise EngineUnavailableError(f"ollama embed failed: {e}") from e
        data = r.json()
        vec = data.get("embedding") or []
        if not vec:
            raise EngineUnavailableError(f"ollama returned empty embedding for model '{model}'")
        return [float(x) for x in vec], int(data.get("prompt_eval_count", 0) or 0)

    async def embed(self, *, model: str, inputs: list[str]) -> EngineEmbedResult:
        import asyncio
        results = await asyncio.gather(*[self._embed_one(model, t) for t in inputs])
        vectors = [r[0] for r in results]
        total_in = sum(r[1] for r in results)
        return EngineEmbedResult(vectors=vectors, dims=len(vectors[0]) if vectors else 0, tokens_in=total_in)
