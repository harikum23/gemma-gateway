from __future__ import annotations

import json
from typing import Any


_DEFAULT_TYPES = ["person", "org", "date", "ticker"]


async def run(args: dict, settings: Any, redis_client: Any = None, engine: Any = None) -> str:
    text: str = args.get("text", "").strip()
    types: list[str] = args.get("types", _DEFAULT_TYPES)
    if not text:
        return "error: missing 'text' argument"
    if engine is None:
        return "error: engine not available"

    type_list = ", ".join(types)
    prompt = (
        f"Extract all named entities of types [{type_list}] from the following text. "
        f"Return a JSON array of objects with keys 'entity' and 'type'. "
        f"Return only the JSON array, no explanation.\n\nText:\n{text}"
    )
    try:
        result = await engine.generate(
            model=settings.default_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=getattr(settings, "tool_extract_entities_max_tokens", 512),
            stop=None,
            tools=None,
            response_format="json",
            json_schema=None,
        )
        raw = result.content.strip()
        # Validate it is parseable JSON
        parsed = json.loads(raw)
        return json.dumps(parsed)
    except json.JSONDecodeError:
        # Return the raw text so the model can still interpret it
        return result.content.strip()
    except Exception as exc:
        return f"error: extract_entities failed: {exc}"
