from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


Role = Literal["system", "user", "assistant", "tool"]


class WebSearchOptions(BaseModel):
    max_results: int = 5
    days: int | None = None
    domains_allow: list[str] = []


class Message(BaseModel):
    role: Role
    content: str
    name: str | None = None


class ToolParameterSchema(BaseModel):
    type: str = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class ToolDef(BaseModel):
    name: str
    description: str = ""
    parameters: ToolParameterSchema | dict[str, Any] = Field(default_factory=dict)


class GenerateRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    temperature: float = 0.2
    top_p: float | None = None
    max_tokens: int = 1024
    stop: list[str] | None = None
    response_format: Literal["text", "json", "tool_call"] = "text"
    json_schema: dict[str, Any] | None = None
    tools: list[ToolDef] | None = None
    tool_choice: str | dict[str, Any] | None = None
    workflow: str | None = None
    priority: Literal["normal", "high"] = "normal"
    stream: bool = True
    timeout_ms: int | None = None
    enable_web_search: bool = False
    web_search: WebSearchOptions | None = None

    @field_validator("messages")
    @classmethod
    def _at_least_one_message(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("messages must contain at least one entry")
        return v

    @field_validator("max_tokens")
    @classmethod
    def _max_tokens_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class AgentOptions(BaseModel):
    system: str = ""
    builtin_tools: list[str] = Field(default_factory=list)
    custom_tools: list[dict[str, Any]] = Field(default_factory=list)
    max_steps: int = 4
    total_budget_ms: int = 30_000
    return_trace: bool = False
    no_store: bool = False


class AgentRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    agent: AgentOptions = Field(default_factory=AgentOptions)

    @field_validator("messages")
    @classmethod
    def _at_least_one_message(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("messages must contain at least one entry")
        return v


class EmbedRequest(BaseModel):
    model: str | None = None
    inputs: list[str]
    normalize: bool = True

    @field_validator("inputs")
    @classmethod
    def _not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("inputs must not be empty")
        if any(not isinstance(s, str) for s in v):
            raise ValueError("inputs must all be strings")
        return v
