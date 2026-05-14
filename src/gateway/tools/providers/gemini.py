from __future__ import annotations

from typing import Any, TypedDict

import httpx
from loguru import logger

from gateway.tools.http_client import get_client


class SearchSource(TypedDict):
    url: str
    title: str


class SearchResult(TypedDict):
    text: str
    sources: list[SearchSource]


_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/gemini-2.5-flash:generateContent"
)


async def search(query: str, api_key: str) -> SearchResult:
    """Call Gemini 2.5 Flash with google_search grounding and return synthesized answer + sources."""
    payload: dict[str, Any] = {
        "contents": [{"parts": [{"text": query}]}],
        "tools": [{"google_search": {}}],
    }
    client = await get_client(timeout=12.0)
    try:
        resp = await client.post(
            _ENDPOINT,
            params={"key": api_key},
            json=payload,
        )
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Gemini search timed out after 12s: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"Gemini search returned HTTP {resp.status_code}: {resp.text[:300]}"
        )

    data = resp.json()

    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini search returned no candidates")

    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = parts[0].get("text", "") if parts else ""

    grounding = candidate.get("groundingMetadata") or {}
    chunks = grounding.get("groundingChunks") or []
    sources: list[SearchSource] = []
    for chunk in chunks:
        web = chunk.get("web") or {}
        uri = web.get("uri", "")
        title = web.get("title", "")
        if uri:
            sources.append({"url": uri, "title": title})

    logger.debug("gemini search: query={!r} sources={}", query, len(sources))
    return {"text": text, "sources": sources}
