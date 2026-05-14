from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS agent_traces (
  request_id TEXT,
  consumer_key_id TEXT,
  step INTEGER,
  tool_name TEXT,
  tool_args TEXT,
  observation TEXT,
  tokens_in INTEGER,
  tokens_out INTEGER,
  latency_ms INTEGER,
  ts TIMESTAMP DEFAULT current_timestamp
)
"""

_INSERT_SQL = """
INSERT INTO agent_traces
  (request_id, consumer_key_id, step, tool_name, tool_args, observation, tokens_in, tokens_out, latency_ms)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_SQL = """
SELECT request_id, consumer_key_id, step, tool_name, tool_args, observation,
       tokens_in, tokens_out, latency_ms, ts
FROM agent_traces
WHERE request_id = ?
ORDER BY step ASC
"""


def ensure_table(conn: Any) -> None:
    """Create the traces table if it does not exist. Call once on startup."""
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()


def _do_insert(conn: Any, params: tuple) -> None:
    conn.execute(_INSERT_SQL, params)
    conn.commit()


async def log_step(
    *,
    conn: Any,
    request_id: str,
    consumer_key_id: str,
    step: int,
    tool_name: str,
    tool_args: dict,
    observation: str,
    tokens_in: int,
    tokens_out: int,
    latency_ms: int,
    no_store: bool = False,
) -> None:
    """Persist one agent step to DuckDB. Silently skips if no_store=True.

    DuckDB is synchronous, so the actual write runs in a worker thread to
    keep the asyncio event loop unblocked even under heavy trace volume.
    """
    if no_store or conn is None:
        return
    params = (
        request_id,
        consumer_key_id,
        step,
        tool_name,
        json.dumps(tool_args),
        observation,
        tokens_in,
        tokens_out,
        latency_ms,
    )
    try:
        await asyncio.to_thread(_do_insert, conn, params)
    except Exception as exc:
        logger.warning("agent trace log_step failed: {}", exc)


def _do_select(conn: Any, request_id: str) -> list[tuple]:
    return conn.execute(_SELECT_SQL, [request_id]).fetchall()


async def get_trace(conn: Any, request_id: str) -> list[dict]:
    """Return all steps for a given request_id, ordered by step."""
    if conn is None:
        return []
    try:
        rows = await asyncio.to_thread(_do_select, conn, request_id)
        cols = ["request_id", "consumer_key_id", "step", "tool_name", "tool_args",
                "observation", "tokens_in", "tokens_out", "latency_ms", "ts"]
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            try:
                d["tool_args"] = json.loads(d["tool_args"] or "{}")
            except Exception:
                pass
            result.append(d)
        return result
    except Exception as exc:
        logger.warning("agent trace get_trace failed: {}", exc)
        return []
