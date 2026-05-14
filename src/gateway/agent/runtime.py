from __future__ import annotations

import time
import uuid
from typing import Any

from loguru import logger

from gateway.agent import coordinator as coord_mod
from gateway.agent import memory as memory_mod
from gateway.agent import trace as trace_mod
from gateway.models.requests import AgentRequest
from gateway.models.responses import AgentResponse, AgentStep

# Map of builtin tool names to their module paths (imported lazily)
_BUILTIN_TOOL_MODULES: dict[str, str] = {
    "web_search": "gateway.tools.web_search",  # special wrapper
    "fetch_url": "gateway.tools.builtin.fetch_url",
    "clean_text": "gateway.tools.builtin.clean_text",
    "dedupe": "gateway.tools.builtin.dedupe",
    "validate_schema": "gateway.tools.builtin.validate_schema",
    "summarize": "gateway.tools.builtin.summarize",
    "extract_entities": "gateway.tools.builtin.extract_entities",
    "calculator": "gateway.tools.builtin.calculator",
    "date_parse": "gateway.tools.builtin.date_parse",
    "translate": "gateway.tools.builtin.translate",
}

# Tool definitions injected into the model for built-ins
_BUILTIN_TOOL_DEFS: dict[str, dict] = {
    "web_search": {
        "name": "web_search",
        "description": (
            "Search the web for current, real-time, or up-to-date information. "
            "You MUST call this tool whenever the user asks about: stock prices, "
            "market data, news, current events, weather, sports scores, or anything "
            "that may have changed since your training cutoff. "
            "Never answer time-sensitive questions from memory — always search first."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Specific search query"}},
            "required": ["query"],
        },
    },
    "fetch_url": {
        "name": "fetch_url",
        "description": "Fetch and extract text content from a URL.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    "clean_text": {
        "name": "clean_text",
        "description": "Strip HTML tags, collapse whitespace and normalize unicode.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    "dedupe": {
        "name": "dedupe",
        "description": "Remove near-duplicate strings from a list.",
        "parameters": {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "string"}}},
            "required": ["items"],
        },
    },
    "validate_schema": {
        "name": "validate_schema",
        "description": "Validate data against a JSON schema.",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"description": "Data to validate"},
                "schema": {"type": "object", "description": "JSON Schema"},
            },
            "required": ["data", "schema"],
        },
    },
    "summarize": {
        "name": "summarize",
        "description": "Summarize a piece of text.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "max_sentences": {"type": "integer", "default": 5},
            },
            "required": ["text"],
        },
    },
    "extract_entities": {
        "name": "extract_entities",
        "description": "Extract named entities from text.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "types": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["text"],
        },
    },
    "calculator": {
        "name": "calculator",
        "description": "Evaluate a safe arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    "date_parse": {
        "name": "date_parse",
        "description": "Parse a natural language date string to ISO format.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "relative_to": {"type": "string"},
            },
            "required": ["text"],
        },
    },
    "translate": {
        "name": "translate",
        "description": "Translate text to a target language.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "target_language": {"type": "string"},
            },
            "required": ["text", "target_language"],
        },
    },
}


async def _execute_builtin(
    tool_name: str,
    tool_args: dict,
    settings: Any,
    redis_client: Any,
    engine: Any,
) -> tuple[str, list[dict]]:
    """Dispatch to the correct builtin tool module.

    Returns (observation_text, sources) where sources is a list of
    {"url": str, "title": str} dicts (non-empty only for web_search).
    """
    import importlib

    # web_search has a different signature — handle specially
    if tool_name == "web_search":
        from gateway.tools.web_search import web_search
        try:
            result = await web_search(
                query=tool_args.get("query", ""),
                days=tool_args.get("days"),
                settings=settings,
                redis_client=redis_client,
                api_key_id="agent",
            )
            raw = result.get("text") or ""
            cleaned = coord_mod.clean_gemini_output(raw)
            return cleaned, result.get("sources") or []
        except Exception as exc:
            return f"web_search error: {exc}", []

    module_path = _BUILTIN_TOOL_MODULES.get(tool_name)
    if not module_path:
        return f"error: unknown builtin tool '{tool_name}'", []

    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        return f"error: could not import tool '{tool_name}': {exc}", []

    run_fn = getattr(mod, "run", None)
    if run_fn is None:
        return f"error: tool '{tool_name}' has no run() function", []

    # Engine-using tools need the engine passed through
    import inspect
    sig = inspect.signature(run_fn)
    try:
        if "engine" in sig.parameters:
            obs = await run_fn(tool_args, settings, redis_client, engine=engine)
        else:
            obs = await run_fn(tool_args, settings, redis_client)
    except Exception as exc:
        return f"error running tool '{tool_name}': {exc}", []
    return obs, []


async def run_agent(
    request: AgentRequest,
    engine: Any,
    app_state: Any,
) -> AgentResponse:
    settings = app_state.settings
    redis_client = getattr(app_state, "redis", None)
    trace_conn = getattr(app_state, "trace_db", None)

    req_id = uuid.uuid4().hex
    opts = request.agent

    # Clamp max_steps to ceiling
    max_steps = min(opts.max_steps, getattr(settings, "agent_max_steps_ceiling", 10))
    budget_ms = opts.total_budget_ms
    deadline = time.monotonic() + budget_ms / 1000.0

    # Build tool definitions first (needed for system prompt)
    builtin_names: set[str] = set(opts.builtin_tools)
    custom_tool_names: set[str] = {t.get("name", "") for t in opts.custom_tools}

    all_tool_defs: list[dict] = []
    for name in opts.builtin_tools:
        if name in _BUILTIN_TOOL_DEFS:
            all_tool_defs.append(_BUILTIN_TOOL_DEFS[name])
    all_tool_defs.extend(opts.custom_tools)

    # Classify the user's intent from the last user message so the coordinator
    # prompt can be tailored to the query type (stock / news / factual / general).
    last_user_msg = next(
        (m.content for m in reversed(request.messages) if m.role == "user"),
        "",
    )
    query_type = coord_mod.classify_query(last_user_msg)
    logger.debug("agent query_type={} last_user={!r}", query_type, last_user_msg[:80])

    # Build initial message list
    messages: list[dict] = []
    system_prompt = opts.system
    if not system_prompt and all_tool_defs:
        tool_names = [t["name"] for t in all_tool_defs]
        system_prompt = coord_mod.coordinator_prompt(query_type, tool_names)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(m.model_dump(exclude_none=True) for m in request.messages)

    # State accumulators
    steps: list[AgentStep] = []
    all_sources: list[dict] = []
    total_tokens_in = 0
    total_tokens_out = 0
    finish_reason = "max_steps"
    final_content = ""
    pending_tool_call: dict | None = None

    model = request.model or settings.default_model

    for step_idx in range(max_steps):
        if time.monotonic() >= deadline:
            finish_reason = "budget_exhausted"
            break

        step_start = time.monotonic()
        try:
            result = await engine.generate(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=getattr(settings, "agent_runtime_max_tokens", 2048),
                stop=None,
                tools=all_tool_defs if all_tool_defs else None,
                response_format="text",
                json_schema=None,
            )
        except Exception as exc:
            logger.warning("agent engine.generate failed at step {}: {}", step_idx, exc)
            finish_reason = "error"
            final_content = f"Agent error: {exc}"
            break

        step_latency_ms = int((time.monotonic() - step_start) * 1000)
        total_tokens_in += result.tokens_in
        total_tokens_out += result.tokens_out

        # --- STOP: model is done (both conditions must hold) ---
        if result.finish_reason == "stop" and not result.tool_calls:
            final_content = result.content or ""
            finish_reason = "stop"
            # Trace the final step as a "stop" step
            await trace_mod.log_step(
                conn=trace_conn,
                request_id=req_id,
                consumer_key_id=getattr(app_state, "_agent_key_id", "unknown"),
                step=step_idx,
                tool_name="_final",
                tool_args={},
                observation=final_content,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                latency_ms=step_latency_ms,
                no_store=opts.no_store,
            )
            break

        # --- TOOL_CALLS: model wants to use a tool ---
        # Assign stable IDs so tool results can be matched back
        tc_ids = [uuid.uuid4().hex[:8] for _ in result.tool_calls]
        messages.append({
            "role": "assistant",
            "content": result.content or "",
            "tool_calls": [
                {
                    "id": tc_ids[i],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc.get("arguments", {})},
                }
                for i, tc in enumerate(result.tool_calls)
            ],
        })

        for i, tc in enumerate(result.tool_calls):
            tool_name = tc.get("name", "")
            tool_args = tc.get("arguments") or {}

            if tool_name in builtin_names:
                # Execute built-in tool
                tool_start = time.monotonic()
                observation, sources = await _execute_builtin(
                    tool_name, tool_args, settings, redis_client, engine
                )
                tool_latency_ms = int((time.monotonic() - tool_start) * 1000)

                # Collect sources from web_search for the final response
                for src in sources:
                    if src not in all_sources:
                        all_sources.append(src)

                messages = memory_mod.add_message(messages, {
                    "role": "tool",
                    "tool_call_id": tc_ids[i],
                    "name": tool_name,
                    "content": observation,
                })

                step_record = AgentStep(
                    step=step_idx,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    observation=observation,
                    latency_ms=tool_latency_ms,
                )
                steps.append(step_record)

                await trace_mod.log_step(
                    conn=trace_conn,
                    request_id=req_id,
                    consumer_key_id=getattr(app_state, "_agent_key_id", "unknown"),
                    step=step_idx,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    observation=observation,
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    latency_ms=step_latency_ms,
                    no_store=opts.no_store,
                )

            elif tool_name in custom_tool_names:
                # Consumer must supply the result — pause and return
                pending_tool_call = {"name": tool_name, "args": tool_args}
                finish_reason = "needs_tool"

                await trace_mod.log_step(
                    conn=trace_conn,
                    request_id=req_id,
                    consumer_key_id=getattr(app_state, "_agent_key_id", "unknown"),
                    step=step_idx,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    observation="_pending_consumer_tool",
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    latency_ms=step_latency_ms,
                    no_store=opts.no_store,
                )
                break  # exit the tc loop
            else:
                # Unknown tool — append empty result and continue
                messages = memory_mod.add_message(messages, {
                    "role": "tool",
                    "name": tool_name,
                    "content": f"error: unknown tool '{tool_name}'",
                })

        if finish_reason == "needs_tool":
            break

        # Summarise context if approaching token limit
        messages = await memory_mod.maybe_summarize(messages, engine, settings)

        # Re-check budget after tool execution
        if time.monotonic() >= deadline:
            finish_reason = "budget_exhausted"
            break

    budget_used_ms = int((budget_ms / 1000.0 - (deadline - time.monotonic())) * 1000)

    from gateway.models.responses import SearchSource
    sources_out = [SearchSource(url=s["url"], title=s.get("title", "")) for s in all_sources]

    return AgentResponse(
        request_id=req_id,
        content=final_content,
        finish_reason=finish_reason,
        sources=sources_out,
        steps=steps if opts.return_trace else [],
        tokens_in=total_tokens_in,
        tokens_out=total_tokens_out,
        budget_used_ms=max(0, budget_used_ms),
        pending_tool_call=pending_tool_call,
    )
