from __future__ import annotations

import time
from dataclasses import dataclass

from gateway.errors import RateLimitError


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class TokenBucketRateLimiter:
    """Single-process, in-memory token bucket per key. Sufficient for v1 LOCKED.

    Horizontal scale later: swap impl to a Redis-backed bucket without touching callers.
    """

    def __init__(self, rps: float, burst: float | None = None) -> None:
        self.rps = float(rps)
        self.burst = float(burst if burst is not None else max(rps * 2, 1.0))
        self._buckets: dict[str, _Bucket] = {}

    def check(self, key: str, cost: float = 1.0) -> None:
        now = time.monotonic()
        b = self._buckets.get(key)
        if b is None:
            b = _Bucket(tokens=self.burst, last_refill=now)
            self._buckets[key] = b
        elapsed = now - b.last_refill
        b.tokens = min(self.burst, b.tokens + elapsed * self.rps)
        b.last_refill = now
        if b.tokens < cost:
            needed = cost - b.tokens
            retry_after = needed / self.rps if self.rps > 0 else 1.0
            raise RateLimitError(retry_after)
        b.tokens -= cost
