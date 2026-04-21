from __future__ import annotations

import asyncio

import pytest

from gateway.errors import QueueFullError
from gateway.queue import AdmissionQueue


@pytest.mark.asyncio
async def test_queue_executes_and_returns_result() -> None:
    q = AdmissionQueue(concurrency=2, max_depth=10, default_max_wait_ms=5000)
    q.start()
    try:
        async def work():
            await asyncio.sleep(0.01)
            return 42

        result = await q.submit(work)
        assert result == 42
    finally:
        await q.stop()


@pytest.mark.asyncio
async def test_queue_rejects_when_full() -> None:
    q = AdmissionQueue(concurrency=1, max_depth=1, default_max_wait_ms=5000)
    q.start()
    try:
        async def slow():
            await asyncio.sleep(0.5)
            return 1

        first = asyncio.create_task(q.submit(slow))
        await asyncio.sleep(0.05)
        # one in flight, queue capacity 1 — second should fast-fail
        with pytest.raises(QueueFullError):
            await q.submit(slow)
        await first
    finally:
        await q.stop()
