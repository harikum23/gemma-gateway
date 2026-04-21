from __future__ import annotations

import pytest

from gateway.errors import RateLimitError
from gateway.rate_limit import TokenBucketRateLimiter


def test_rate_limit_rejects_after_burst_exhausted() -> None:
    rl = TokenBucketRateLimiter(rps=10, burst=3)
    for _ in range(3):
        rl.check("k")
    with pytest.raises(RateLimitError):
        rl.check("k")


def test_rate_limit_isolates_keys() -> None:
    rl = TokenBucketRateLimiter(rps=1, burst=1)
    rl.check("a")
    with pytest.raises(RateLimitError):
        rl.check("a")
    rl.check("b")  # separate bucket
