from __future__ import annotations

import time

from gateway.circuit import CircuitBreaker


def test_circuit_closed_by_default() -> None:
    cb = CircuitBreaker()
    assert cb.state() == "closed"
    assert cb.allow(is_high_priority=False) is True


def test_circuit_trips_on_failure_rate() -> None:
    cb = CircuitBreaker(error_threshold=0.5, min_samples=10, window_s=60, reset_s=1)
    for _ in range(5):
        cb.record_success()
    for _ in range(6):
        cb.record_failure()
    assert cb.state() == "open"
    assert cb.allow(is_high_priority=False) is False
    # high-priority still allowed
    assert cb.allow(is_high_priority=True) is True


def test_circuit_half_open_after_reset() -> None:
    cb = CircuitBreaker(error_threshold=0.1, min_samples=2, window_s=60, reset_s=0.001)
    cb.record_failure()
    cb.record_failure()
    assert cb.state() == "open"
    time.sleep(0.01)
    assert cb.state() == "half_open"
    cb.record_success()
    assert cb.state() == "closed"
