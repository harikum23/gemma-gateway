"""Shared httpx.AsyncClient pool for outbound HTTP calls.

Creating a fresh AsyncClient per request leaks connections and forces a
TCP/TLS handshake every time. The pool lazily creates one client per
(host, timeout) bucket and reuses it for the lifetime of the process.

Clients are not explicitly closed; httpx's connection pool is GC-clean
on process exit. For tests, call ``reset_pool()`` to drop cached clients.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx


_pool: dict[tuple[str, float], httpx.AsyncClient] = {}
_lock = asyncio.Lock()


def _bucket_key(base_url: str, timeout: float) -> tuple[str, float]:
    return (base_url, round(timeout, 2))


async def get_client(base_url: str = "", *, timeout: float = 10.0, **kwargs: Any) -> httpx.AsyncClient:
    """Return a shared AsyncClient for the given host/timeout bucket."""
    key = _bucket_key(base_url, timeout)
    client = _pool.get(key)
    if client is not None and not client.is_closed:
        return client
    async with _lock:
        client = _pool.get(key)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(base_url=base_url, timeout=timeout, **kwargs)
            _pool[key] = client
        return client


async def reset_pool() -> None:
    """Close and forget all pooled clients. Call from test teardown."""
    async with _lock:
        for c in _pool.values():
            try:
                await c.aclose()
            except Exception:
                pass
        _pool.clear()
