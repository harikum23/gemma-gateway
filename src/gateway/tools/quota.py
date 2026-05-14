from __future__ import annotations

from datetime import date
from typing import Any

from gateway.errors import QuotaExceededError


def _quota_key(api_key_id: str) -> str:
    return f"search_quota:{api_key_id}:{date.today()}"


async def check_and_increment(
    redis_client: Any,
    api_key_id: str,
    daily_limit: int,
    *,
    redis_required: bool = False,
) -> None:
    """Increment the daily search counter for this API key.

    Raises QuotaExceededError if the limit has been reached or if
    redis_required=True and Redis is unreachable. Otherwise fails open when
    Redis is absent (quota best-effort).
    """
    if redis_client is None:
        if redis_required:
            raise QuotaExceededError(api_key_id, daily_limit)
        return

    key = _quota_key(api_key_id)
    try:
        count = await redis_client.incr(key)
        # Set expiry on first write (25 hours so it covers timezone edge cases)
        if count == 1:
            await redis_client.expire(key, 25 * 3600)
        if count > daily_limit:
            raise QuotaExceededError(api_key_id, daily_limit)
    except QuotaExceededError:
        raise
    except Exception:
        if redis_required:
            raise QuotaExceededError(api_key_id, daily_limit) from None
        # Fail open on Redis errors — don't block search on infra issues
