from __future__ import annotations

from typing import Any

from loguru import logger

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
    await quota_mod.check_and_increment(
        redis_client,
        api_key_id,
        settings.search_daily_quota_per_key,
        redis_required=getattr(settings, "redis_required", False),
    )

    # 3. Call provider — fall back to SearXNG if Gemini fails
    if provider == "gemini":
        try:
            result = await gemini_provider.search(query, settings.gemini_api_key)
        except Exception as exc:
            logger.warning("gemini search failed, falling back to searxng: {}", exc)
            try:
                result = await searxng_provider.search(query)
            except Exception as exc2:
                logger.warning("searxng fallback also failed: {}", exc2)
                return {"text": "", "sources": [], "cache_hit": False}
    else:
        try:
            result = await searxng_provider.search(query)
        except Exception as exc:
            logger.warning("searxng search failed: {}", exc)
            return {"text": "", "sources": [], "cache_hit": False}

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
