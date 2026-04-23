from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class SearchSource(BaseModel):
    url: str
    title: str


class GenerateResponse(BaseModel):
    model: str
    content: str
    tool_calls: list[ToolCall] | None = None
    finish_reason: Literal["stop", "length", "abstain", "error", "tool_calls"] = "stop"
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    request_id: str
    queue_wait_ms: float = 0.0
    sources: list[SearchSource] = Field(default_factory=list)
    search_used: bool = False
    search_cache_hit: bool = False


class AgentStep(BaseModel):
    step: int
    tool_name: str
    tool_args: dict
    observation: str
    latency_ms: int


class AgentResponse(BaseModel):
    request_id: str
    content: str
    finish_reason: str  # "stop"|"max_steps"|"budget_exhausted"|"needs_tool"|"error"
    sources: list[SearchSource] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    budget_used_ms: int = 0
    pending_tool_call: dict | None = None


class EmbedResponse(BaseModel):
    model: str
    dims: int
    vectors: list[list[float]]
    tokens_in: int = 0
    latency_ms: float = 0.0
    request_id: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    engine: str
    engine_url: str
    engine_ready: bool
    queue_depth: int
    queue_max: int
    circuit: str
    models: list[str] = Field(default_factory=list)


class ModelListResponse(BaseModel):
    models: list[dict[str, Any]]
