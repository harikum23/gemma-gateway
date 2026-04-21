from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class EngineGenerateResult:
    content: str
    tokens_in: int
    tokens_out: int
    finish_reason: str
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class EngineEmbedResult:
    vectors: list[list[float]]
    dims: int
    tokens_in: int


@runtime_checkable
class Engine(Protocol):
    name: str

    async def health(self) -> bool: ...

    async def list_models(self) -> list[dict[str, Any]]: ...

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
    ) -> EngineGenerateResult: ...

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
    ) -> AsyncIterator[str]: ...

    async def embed(
        self,
        *,
        model: str,
        inputs: list[str],
    ) -> EngineEmbedResult: ...

    async def aclose(self) -> None: ...
