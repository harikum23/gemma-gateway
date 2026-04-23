from __future__ import annotations

import re
import unicodedata
from typing import Any


def _clean(text: str) -> str:
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize unicode to NFC
    text = unicodedata.normalize("NFC", text)
    # Collapse whitespace (tabs, multiple spaces, newlines) into single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


async def run(args: dict, settings: Any, redis_client: Any = None) -> str:
    text: str = args.get("text", "")
    if not isinstance(text, str):
        return "error: 'text' must be a string"
    return _clean(text)
