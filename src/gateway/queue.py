from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from gateway.errors import QueueFullError, QueueTimeoutError

Priority = Literal["high", "normal"]


@dataclass(order=True)
class _QueueItem:
    priority_rank: int
    enqueued_at: float
    seq: int
    fut: asyncio.Future[Any] = field(compare=False)
    coro_factory: Callable[[], Awaitable[Any]] = field(compare=False)
    max_wait_ms: int = field(compare=False, default=60_000)


class AdmissionQueue:
    """Priority admission queue with age-based TTL drops.

    `high` priority items run before `normal`. Items older than `max_wait_ms` are
    dropped with a typed 504 when dequeued. Rejects with 429 when depth is at capacity.
    """

    _PRIORITY_RANK: dict[Priority, int] = {"high": 0, "normal": 1}

    def __init__(self, *, concurrency: int, max_depth: int, default_max_wait_ms: int) -> None:
        self.max_depth = max_depth
        self.default_max_wait_ms = default_max_wait_ms
        self._concurrency = concurrency
        self._sem = asyncio.Semaphore(concurrency)
        self._items: asyncio.PriorityQueue[_QueueItem] = asyncio.PriorityQueue()
        self._seq = 0
        self._workers_started = False
        self._workers: list[asyncio.Task[None]] = []

    def depth(self) -> int:
        return self._items.qsize()

    def start(self) -> None:
        if self._workers_started:
            return
        self._workers_started = True
        loop = asyncio.get_event_loop()
        for _ in range(self._concurrency):
            self._workers.append(loop.create_task(self._worker()))

    async def stop(self) -> None:
        for w in self._workers:
            w.cancel()
        for w in self._workers:
            try:
                await w
            except (asyncio.CancelledError, Exception):
                pass
        self._workers.clear()
        self._workers_started = False

    async def submit(
        self,
        coro_factory: Callable[[], Awaitable[Any]],
        *,
        priority: Priority = "normal",
        max_wait_ms: int | None = None,
    ) -> Any:
        if self.depth() >= self.max_depth:
            raise QueueFullError(self.depth(), self.max_depth)
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        self._seq += 1
        item = _QueueItem(
            priority_rank=self._PRIORITY_RANK[priority],
            enqueued_at=time.monotonic(),
            seq=self._seq,
            fut=fut,
            coro_factory=coro_factory,
            max_wait_ms=max_wait_ms or self.default_max_wait_ms,
        )
        await self._items.put(item)
        return await fut

    async def _worker(self) -> None:
        while True:
            item = await self._items.get()
            async with self._sem:
                age_ms = (time.monotonic() - item.enqueued_at) * 1000
                if age_ms > item.max_wait_ms:
                    if not item.fut.done():
                        item.fut.set_exception(QueueTimeoutError())
                    continue
                try:
                    result = await item.coro_factory()
                    if not item.fut.done():
                        item.fut.set_result(result)
                except Exception as exc:
                    if not item.fut.done():
                        item.fut.set_exception(exc)
