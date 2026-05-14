from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from gateway.tools.http_client import get_client

_MAX_CHARS = 3000
_TIMEOUT_S = 8.0


async def run(args: dict, settings: Any, redis_client: Any = None) -> str:
    url: str = args.get("url", "").strip()
    if not url:
        return "error: missing 'url' argument"
    try:
        client = await get_client(timeout=_TIMEOUT_S, follow_redirects=True)
        resp = await client.get(url, headers={"User-Agent": "gemma-gateway/2.0"})
        resp.raise_for_status()
        raw_html = resp.text
    except httpx.HTTPStatusError as exc:
        return f"error: HTTP {exc.response.status_code} fetching {url}"
    except Exception as exc:
        logger.warning("fetch_url failed for {}: {}", url, exc)
        return f"error: {exc}"

    try:
        import trafilatura  # type: ignore[import]
        text = trafilatura.extract(raw_html) or ""
    except Exception:
        # Fallback: strip tags manually
        import re
        text = re.sub(r"<[^>]+>", " ", raw_html)

    text = text.strip()
    if not text:
        return "error: no text content extracted"
    return text[:_MAX_CHARS]
