from __future__ import annotations

from typing import Any


async def run(args: dict, settings: Any, redis_client: Any = None) -> str:
    text: str = args.get("text", "").strip()
    relative_to: str | None = args.get("relative_to")
    if not text:
        return "error: missing 'text' argument"
    try:
        import dateparser  # type: ignore[import]
        parse_kwargs: dict = {"RETURN_AS_TIMEZONE_AWARE": False}
        if relative_to:
            # dateparser supports RELATIVE_BASE as a datetime object
            from datetime import datetime
            try:
                base = datetime.fromisoformat(relative_to)
                parse_kwargs["RELATIVE_BASE"] = base
            except ValueError:
                pass
        parsed = dateparser.parse(text, settings=parse_kwargs)
        if parsed is None:
            return "unparseable"
        return parsed.isoformat()
    except ImportError:
        return "error: dateparser package not installed"
    except Exception as exc:
        return f"error: {exc}"
