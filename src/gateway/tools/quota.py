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
) -> None:
    """Increment the daily search counter for this API key.

    Raises QuotaExceededError if the limit has been reached.
    If Redis is unavailable, quota is not enforced (fail open).
    """
    if redis_client is None:
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
        # Fail open on Redis errors — don't block search on infra issues
        pass
