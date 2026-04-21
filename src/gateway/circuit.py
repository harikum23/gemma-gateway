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
    reset_s: int = 30
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
