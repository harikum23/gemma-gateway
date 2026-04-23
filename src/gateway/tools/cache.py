from __future__ import annotations

import hashlib
import json
from typing import Any


def _cache_key(provider: str, query: str, days: int | None) -> str:
    digest = hashlib.md5(query.encode()).hexdigest()
    return f"search:{provider}:{digest}:{days or 0}"


def _ttl(days: int | None, news_ttl: int, evergreen_ttl: int) -> int:
    """Return TTL in seconds: news TTL for recent queries (days <= 7), evergreen otherwise."""
    if days is not None and days <= 7:
        return news_ttl
    return evergreen_ttl


async def get_cached(
    redis_client: Any,
    provider: str,
    query: str,
    days: int | None,
) -> dict | None:
    """Return cached search result dict or None on miss / Redis unavailable."""
    if redis_client is None:
        return None
    key = _cache_key(provider, query, days)
    try:
        raw = await redis_client.get(key)
        if raw is not None:
            return json.loads(raw)
    except Exception:
        pass
    return None


async def set_cached(
    redis_client: Any,
    provider: str,
    query: str,
    days: int | None,
    result: dict,
    news_ttl: int,
    evergreen_ttl: int,
) -> None:
    """Store search result in Redis. Silently ignores errors."""
    if redis_client is None:
        return
    key = _cache_key(provider, query, days)
    ttl = _ttl(days, news_ttl, evergreen_ttl)
    try:
        await redis_client.set(key, json.dumps(result), ex=ttl)
    except Exception:
        pass
