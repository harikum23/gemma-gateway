from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

State = Literal["closed", "open", "half_open"]


@dataclass
class CircuitBreaker:
    error_threshold: float = 0.2
    window_s: int = 30
    reset_s: float = 30.0
    min_samples: int = 20

    _events: deque[tuple[float, bool]] = field(default_factory=deque)
    _state: State = "closed"
    _opened_at: float = 0.0

    def state(self) -> State:
        now = time.monotonic()
        if self._state == "open" and (now - self._opened_at) >= self.reset_s:
            self._state = "half_open"
        return self._state

    def allow(self, *, is_high_priority: bool) -> bool:
        s = self.state()
        if s == "closed":
            return True
        if s == "half_open":
            return True
        return is_high_priority

    def record_success(self) -> None:
        self._push(True)
        if self._state in ("open", "half_open"):
            self._state = "closed"
            self._events.clear()

    def record_failure(self) -> None:
        self._push(False)
        self._maybe_trip()

    def _push(self, ok: bool) -> None:
        now = time.monotonic()
        self._events.append((now, ok))
        cutoff = now - self.window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _maybe_trip(self) -> None:
        if self._state == "open":
            return
        n = len(self._events)
        if n < self.min_samples:
            return
        fails = sum(1 for _, ok in self._events if not ok)
        rate = fails / n
        if rate >= self.error_threshold:
            self._state = "open"
            self._opened_at = time.monotonic()


class CircuitRegistry:
    """Lazily-created per-(engine, model) circuit breakers.

    A failing backend or model should not trip the breaker for unrelated ones.
    Callers pass the engine name + model id; the registry returns (and caches)
    a breaker for that key. The no-arg methods (allow/record_success/
    record_failure/state) operate on a shared "default" breaker for callers
    that don't carry routing info — preserves the old single-breaker API.
    """

    def __init__(
        self,
        *,
        error_threshold: float = 0.2,
        window_s: int = 30,
        reset_s: float = 30.0,
        min_samples: int = 20,
    ) -> None:
        self._params = dict(
            error_threshold=error_threshold,
            window_s=window_s,
            reset_s=reset_s,
            min_samples=min_samples,
        )
        self._breakers: dict[str, CircuitBreaker] = {}

    def for_key(self, engine: str = "default", model: str = "default") -> CircuitBreaker:
        key = f"{engine}::{model}"
        b = self._breakers.get(key)
        if b is None:
            b = CircuitBreaker(**self._params)
            self._breakers[key] = b
        return b

    # Convenience pass-throughs against the shared default breaker so
    # existing call sites keep working until they migrate to for_key().
    def allow(self, *, is_high_priority: bool) -> bool:
        return self.for_key().allow(is_high_priority=is_high_priority)

    def record_success(self) -> None:
        self.for_key().record_success()

    def record_failure(self) -> None:
        self.for_key().record_failure()

    def state(self) -> State:
        # Worst state across all known breakers — health probes care about
        # whether *anything* is open, not just the default key.
        if not self._breakers:
            return "closed"
        order = {"open": 2, "half_open": 1, "closed": 0}
        return max((b.state() for b in self._breakers.values()), key=lambda s: order[s])

    def snapshot(self) -> dict[str, State]:
        return {k: b.state() for k, b in self._breakers.items()}
