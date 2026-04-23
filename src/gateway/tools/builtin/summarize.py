from __future__ import annotations

from typing import Any


async def run(args: dict, settings: Any, redis_client: Any = None, engine: Any = None) -> str:
    text: str = args.get("text", "").strip()
    max_sentences: int = int(args.get("max_sentences", 5))
    if not text:
        return "error: missing 'text' argument"
    if engine is None:
        return "error: engine not available"
    try:
        result = await engine.generate(
            model=settings.default_model,
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize the following in {max_sentences} sentences:\n\n{text}",
                }
            ],
            temperature=0.2,
            max_tokens=512,
            stop=None,
            tools=None,
            response_format="text",
            json_schema=None,
        )
        return result.content.strip()
    except Exception as exc:
        return f"error: summarize failed: {exc}"
