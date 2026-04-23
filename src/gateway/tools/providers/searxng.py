from __future__ import annotations

from typing import TypedDict

import httpx
from loguru import logger


class SearchSource(TypedDict):
    url: str
    title: str


class SearchResult(TypedDict):
    text: str
    sources: list[SearchSource]


_SEARXNG_BASE = "http://searxng:8888/search"


async def search(query: str, base_url: str = _SEARXNG_BASE) -> SearchResult:
    """Query self-hosted SearXNG and return joined snippets + sources."""
    params = {
        "q": query,
        "format": "json",
        "engines": "google,bing",
    }
    async with httpx.AsyncClient(timeout=4.0) as client:
        try:
            resp = await client.get(base_url, params=params)
        except httpx.TimeoutException as exc:
            raise RuntimeError(f"SearXNG search timed out after 4s: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"SearXNG search returned HTTP {resp.status_code}: {resp.text[:300]}"
        )

    data = resp.json()
    results = data.get("results") or []

    sources: list[SearchSource] = []
    snippets: list[str] = []
    for r in results:
        url = r.get("url", "")
        title = r.get("title", "")
        content = r.get("content", "")
        if url:
            sources.append({"url": url, "title": title})
        if content:
            snippets.append(content)

    text = "\n\n".join(snippets)
    logger.debug("searxng search: query={!r} results={}", query, len(results))
    return {"text": text, "sources": sources}
