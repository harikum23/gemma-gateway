from __future__ import annotations

import time
from typing import Any

from loguru import logger

from gateway.models.requests import GenerateRequest
from gateway.tools.web_search import web_search


# Tool definition injected into Ollama when web search is enabled
_WEB_SEARCH_TOOL_DEF: dict[str, Any] = {
    "name": "web_search",
    "description": (
        "Search the web for current information. "
        "Use this when the user asks about recent events, live data, or anything "
        "that requires up-to-date knowledge."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
        },
        "required": ["query"],
    },
}


class LoopResult:
    __slots__ = ("content", "sources", "cache_hit", "tokens_in", "tokens_out", "finish_reason")

    def __init__(
        self,
        content: str,
        sources: list[dict],
        cache_hit: bool,
        tokens_in: int,
        tokens_out: int,
        finish_reason: str,
    ) -> None:
        self.content = content
        self.sources = sources
        self.cache_hit = cache_hit
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.finish_reason = finish_reason


async def run_with_search(
    body: GenerateRequest,
    engine: Any,
    state: Any,
    api_key_id: str = "unknown",
) -> LoopResult:
    """Bounded tool-call loop that handles web_search tool calls from the model.

    Injects the web_search tool definition, then iterates until:
    - finish_reason is 'stop'
    - max_iterations exhausted
    - total budget (ms) exceeded
    """
    settings = state.settings
    redis_client = getattr(state, "redis", None)
    principal_key_id = api_key_id

    ws_opts = body.web_search
    days: int | None = ws_opts.days if ws_opts else None

    # Merge caller-supplied tools with the web_search tool
    extra_tools: list[dict[str, Any]] = [_WEB_SEARCH_TOOL_DEF]
    existing_tools: list[dict[str, Any]] = (
        [t.model_dump(exclude_none=True) for t in body.tools] if body.tools else []
    )
    all_tools = existing_tools + extra_tools

    messages: list[dict[str, Any]] = [m.model_dump(exclude_none=True) for m in body.messages]
    model = body.model or settings.default_model

    accumulated_sources: list[dict] = []
    cache_hit = False
    total_tokens_in = 0
    total_tokens_out = 0

    budget_ms = settings.search_total_budget_ms
    max_iter = settings.search_max_iterations
    deadline = time.monotonic() + budget_ms / 1000.0

    for iteration in range(max_iter):
        if time.monotonic() >= deadline:
            logger.warning("web-search loop: budget exceeded before iteration {}", iteration)
            break

        result = await engine.generate(
            model=model,
            messages=messages,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            stop=body.stop,
            tools=all_tools,
            response_format=body.response_format,
            json_schema=body.json_schema,
        )

        total_tokens_in += result.tokens_in
        total_tokens_out += result.tokens_out

        if result.finish_reason != "tool_calls" or not result.tool_calls:
            # Model is done — return final answer
            return LoopResult(
                content=result.content,
                sources=accumulated_sources,
                cache_hit=cache_hit,
                tokens_in=total_tokens_in,
                tokens_out=total_tokens_out,
                finish_reason=result.finish_reason,
            )

        # Append the assistant turn with its tool calls so the conversation is coherent
        messages.append({
            "role": "assistant",
            "content": result.content or "",
            "tool_calls": [
                {
                    "function": {
                        "name": tc["name"],
                        "arguments": tc.get("arguments", {}),
                    }
                }
                for tc in result.tool_calls
            ],
        })

        # Execute each requested tool call
        for tc in result.tool_calls:
            tool_name = tc.get("name", "")
            if tool_name != "web_search":
                # Unknown tool — append empty result and continue
                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": "",
                })
                continue

            args = tc.get("arguments") or {}
            query: str = args.get("query", "")
            if not query:
                messages.append({"role": "tool", "name": "web_search", "content": "no query provided"})
                continue

            if time.monotonic() >= deadline:
                logger.warning("web-search loop: budget exceeded during tool call")
                messages.append({"role": "tool", "name": "web_search", "content": "search skipped: budget exceeded"})
                break

            try:
                search_result = await web_search(
                    query=query,
                    days=days,
                    settings=settings,
                    redis_client=redis_client,
                    api_key_id=principal_key_id,
                )
            except Exception as exc:
                logger.warning("web_search tool call failed: {}", exc)
                messages.append({"role": "tool", "name": "web_search", "content": f"search failed: {exc}"})
                continue

            if search_result.get("cache_hit"):
                cache_hit = True
            accumulated_sources.extend(search_result.get("sources") or [])

            # Provide synthesized text back to the model
            messages.append({
                "role": "tool",
                "name": "web_search",
                "content": search_result.get("text") or "",
            })

    # Budget or iteration limit hit — do a final non-tool generate to get a coherent answer
    if time.monotonic() < deadline:
        try:
            final = await engine.generate(
                model=model,
                messages=messages,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                stop=body.stop,
                tools=None,  # No tools on final pass to force a text response
                response_format=body.response_format,
                json_schema=body.json_schema,
            )
            total_tokens_in += final.tokens_in
            total_tokens_out += final.tokens_out
            return LoopResult(
                content=final.content,
                sources=accumulated_sources,
                cache_hit=cache_hit,
                tokens_in=total_tokens_in,
                tokens_out=total_tokens_out,
                finish_reason=final.finish_reason,
            )
        except Exception as exc:
            logger.warning("web-search loop final generate failed: {}", exc)

    # Fallback: return whatever content accumulated
    return LoopResult(
        content="",
        sources=accumulated_sources,
        cache_hit=cache_hit,
        tokens_in=total_tokens_in,
        tokens_out=total_tokens_out,
        finish_reason="stop",
    )
