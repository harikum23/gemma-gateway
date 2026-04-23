from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.settings import get_settings


# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_calculator_valid_expression():
    from gateway.tools.builtin.calculator import run
    settings = get_settings()
    result = await run({"expression": "2 + 3 * 4"}, settings)
    assert result == "14"


@pytest.mark.asyncio
async def test_calculator_float():
    from gateway.tools.builtin.calculator import run
    settings = get_settings()
    result = await run({"expression": "7 / 2"}, settings)
    assert result == "3.5"


@pytest.mark.asyncio
async def test_calculator_unsafe_expression_rejected():
    from gateway.tools.builtin.calculator import run
    settings = get_settings()
    result = await run({"expression": "__import__('os').system('ls')"}, settings)
    assert result.startswith("error:")


@pytest.mark.asyncio
async def test_calculator_division_by_zero():
    from gateway.tools.builtin.calculator import run
    settings = get_settings()
    result = await run({"expression": "1 / 0"}, settings)
    assert "division by zero" in result


@pytest.mark.asyncio
async def test_calculator_power():
    from gateway.tools.builtin.calculator import run
    settings = get_settings()
    result = await run({"expression": "2 ** 10"}, settings)
    assert result == "1024"


# ---------------------------------------------------------------------------
# dedupe
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dedupe_removes_duplicates():
    from gateway.tools.builtin.dedupe import run
    settings = get_settings()
    # Second item shares >80% words with the first (9/10 words identical)
    items = [
        "apple banana cherry date elderberry fig grape",
        "apple banana cherry date elderberry fig grape mango",  # 7/8 overlap > 0.8
        "a completely different sentence here now",
    ]
    raw = await run({"items": items}, settings)
    result = json.loads(raw)
    # The highly overlapping second item should be removed
    assert items[0] in result
    assert items[2] in result
    assert len(result) < len(items)


@pytest.mark.asyncio
async def test_dedupe_keeps_originals():
    from gateway.tools.builtin.dedupe import run
    settings = get_settings()
    items = ["apple banana", "cherry date elderberry", "fig grape"]
    raw = await run({"items": items}, settings)
    result = json.loads(raw)
    assert result == items  # no duplicates, all kept


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_clean_text_strips_html():
    from gateway.tools.builtin.clean_text import run
    settings = get_settings()
    result = await run({"text": "<h1>Hello</h1> <p>World</p>"}, settings)
    assert "<" not in result
    assert "Hello" in result
    assert "World" in result


@pytest.mark.asyncio
async def test_clean_text_collapses_whitespace():
    from gateway.tools.builtin.clean_text import run
    settings = get_settings()
    result = await run({"text": "foo   \t  bar\n\nbaz"}, settings)
    assert result == "foo bar baz"


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_schema_valid():
    from gateway.tools.builtin.validate_schema import run
    settings = get_settings()
    data = {"name": "Alice", "age": 30}
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }
    result = await run({"data": data, "schema": schema}, settings)
    assert result == "valid"


@pytest.mark.asyncio
async def test_validate_schema_invalid():
    from gateway.tools.builtin.validate_schema import run
    settings = get_settings()
    data = {"name": 123}  # name should be string
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = await run({"data": data, "schema": schema}, settings)
    assert "validation error" in result or "error" in result


# ---------------------------------------------------------------------------
# date_parse
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_date_parse_iso():
    from gateway.tools.builtin.date_parse import run
    settings = get_settings()
    result = await run({"text": "2024-06-15"}, settings)
    assert "2024-06-15" in result


@pytest.mark.asyncio
async def test_date_parse_natural_language():
    from gateway.tools.builtin.date_parse import run
    settings = get_settings()
    result = await run({"text": "January 1st 2025"}, settings)
    assert "2025-01-01" in result


@pytest.mark.asyncio
async def test_date_parse_unparseable():
    from gateway.tools.builtin.date_parse import run
    settings = get_settings()
    result = await run({"text": "xyznotadate"}, settings)
    assert result == "unparseable"


# ---------------------------------------------------------------------------
# fetch_url (mock httpx)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_url_extracts_text():
    from gateway.tools.builtin import fetch_url

    html = "<html><body><article><p>Hello world from the article.</p></article></body></html>"

    mock_response = MagicMock()
    mock_response.text = html
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_response)

    settings = get_settings()
    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await fetch_url.run({"url": "http://example.com"}, settings)

    assert isinstance(result, str)
    assert len(result) > 0
    assert "error" not in result.lower() or "hello" in result.lower()


@pytest.mark.asyncio
async def test_fetch_url_missing_url():
    from gateway.tools.builtin.fetch_url import run
    settings = get_settings()
    result = await run({}, settings)
    assert result.startswith("error")


# ---------------------------------------------------------------------------
# summarize (mock engine)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summarize_calls_engine():
    from gateway.tools.builtin.summarize import run

    mock_result = MagicMock()
    mock_result.content = "This is the summary."
    mock_engine = AsyncMock()
    mock_engine.generate = AsyncMock(return_value=mock_result)

    settings = get_settings()
    result = await run({"text": "Long text here.", "max_sentences": 2}, settings, engine=mock_engine)
    assert result == "This is the summary."
    mock_engine.generate.assert_called_once()


@pytest.mark.asyncio
async def test_summarize_no_engine():
    from gateway.tools.builtin.summarize import run
    settings = get_settings()
    result = await run({"text": "text"}, settings, engine=None)
    assert "error" in result


# ---------------------------------------------------------------------------
# extract_entities (mock engine)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_entities_returns_json():
    from gateway.tools.builtin.extract_entities import run

    entities = [{"entity": "Apple Inc", "type": "org"}, {"entity": "Tim Cook", "type": "person"}]
    mock_result = MagicMock()
    mock_result.content = json.dumps(entities)
    mock_engine = AsyncMock()
    mock_engine.generate = AsyncMock(return_value=mock_result)

    settings = get_settings()
    result = await run({"text": "Tim Cook leads Apple Inc."}, settings, engine=mock_engine)
    parsed = json.loads(result)
    assert len(parsed) == 2
    assert parsed[0]["type"] == "org"


# ---------------------------------------------------------------------------
# translate (mock engine)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_translate_calls_engine():
    from gateway.tools.builtin.translate import run

    mock_result = MagicMock()
    mock_result.content = "Hola mundo"
    mock_engine = AsyncMock()
    mock_engine.generate = AsyncMock(return_value=mock_result)

    settings = get_settings()
    result = await run({"text": "Hello world", "target_language": "Spanish"}, settings, engine=mock_engine)
    assert result == "Hola mundo"


@pytest.mark.asyncio
async def test_translate_missing_args():
    from gateway.tools.builtin.translate import run
    settings = get_settings()
    result = await run({"text": "hello"}, settings, engine=None)
    assert "error" in result
