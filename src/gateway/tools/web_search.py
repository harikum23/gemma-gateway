from __future__ import annotations

from typing import Any

from gateway.settings import Settings
from gateway.tools import cache as cache_mod
from gateway.tools import quota as quota_mod
from gateway.tools.providers import gemini as gemini_provider
from gateway.tools.providers import searxng as searxng_provider


async def web_search(
    query: str,
    days: int | None,
    settings: Settings,
    redis_client: Any,
    api_key_id: str,
) -> dict:
    """Provider-agnostic web search with caching and quota enforcement.

    Returns:
        {
            "text": str,
            "sources": list[{"url": str, "title": str}],
            "cache_hit": bool,
        }
    """
    provider = settings.search_provider.lower()

    # 1. Check cache before quota — cache hits don't consume quota
    cached = await cache_mod.get_cached(redis_client, provider, query, days)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    # 2. Enforce per-key daily quota
    await quota_mod.check_and_increment(redis_client, api_key_id, settings.search_daily_quota_per_key)

    # 3. Call provider
    if provider == "gemini":
        result = await gemini_provider.search(query, settings.gemini_api_key)
    else:
        result = await searxng_provider.search(query)

    # 4. Store in cache (fire-and-forget; errors suppressed inside set_cached)
    await cache_mod.set_cached(
        redis_client,
        provider,
        query,
        days,
        result,
        settings.search_cache_ttl_news_seconds,
        settings.search_cache_ttl_evergreen_seconds,
    )

    return {**result, "cache_hit": False}
