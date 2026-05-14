from __future__ import annotations

from typing import Any


async def run(args: dict, settings: Any, redis_client: Any = None, engine: Any = None) -> str:
    text: str = args.get("text", "").strip()
    target_language: str = args.get("target_language", "").strip()
    if not text:
        return "error: missing 'text' argument"
    if not target_language:
        return "error: missing 'target_language' argument"
    if engine is None:
        return "error: engine not available"
    try:
        result = await engine.generate(
            model=settings.default_model,
            messages=[
                {
                    "role": "user",
                    "content": f"Translate to {target_language}:\n\n{text}",
                }
            ],
            temperature=0.2,
            max_tokens=getattr(settings, "tool_translate_max_tokens", 1024),
            stop=None,
            tools=None,
            response_format="text",
            json_schema=None,
        )
        return result.content.strip()
    except Exception as exc:
        return f"error: translate failed: {exc}"
