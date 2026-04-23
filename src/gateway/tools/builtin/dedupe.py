from __future__ import annotations

import json
from typing import Any


def _similarity(a: str, b: str) -> float:
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    if not a_words or not b_words:
        return 0.0
    intersection = len(a_words & b_words)
    return intersection / max(len(a_words), len(b_words))


def _dedupe(items: list[str], threshold: float = 0.8) -> list[str]:
    kept: list[str] = []
    for candidate in items:
        is_dup = any(_similarity(candidate, k) > threshold for k in kept)
        if not is_dup:
            kept.append(candidate)
    return kept


async def run(args: dict, settings: Any, redis_client: Any = None) -> str:
    items = args.get("items", [])
    if not isinstance(items, list):
        return "error: 'items' must be a list of strings"
    items = [str(i) for i in items]
    result = _dedupe(items)
    return json.dumps(result)
