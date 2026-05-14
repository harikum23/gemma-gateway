from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from threading import Lock

SCHEMA = """
CREATE TABLE IF NOT EXISTS request_metrics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL NOT NULL,
    endpoint      TEXT NOT NULL,
    status        INTEGER NOT NULL,
    latency_ms    REAL NOT NULL,
    tokens_in     INTEGER NOT NULL DEFAULT 0,
    tokens_out    INTEGER NOT NULL DEFAULT 0,
    model         TEXT,
    workflow      TEXT,
    api_key_id    TEXT,
    error_code    TEXT
);
CREATE INDEX IF NOT EXISTS idx_request_metrics_ts ON request_metrics(ts);
CREATE INDEX IF NOT EXISTS idx_request_metrics_endpoint ON request_metrics(endpoint);
"""


class MetricsStore:
    """Lightweight SQLite metrics sink for v1 LOCKED (no Prometheus yet)."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = Lock()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _record_sync(
        self,
        *,
        endpoint: str,
        status: int,
        latency_ms: float,
        tokens_in: int,
        tokens_out: int,
        model: str | None,
        workflow: str | None,
        api_key_id: str | None,
        error_code: str | None,
    ) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO request_metrics
                   (ts, endpoint, status, latency_ms, tokens_in, tokens_out,
                    model, workflow, api_key_id, error_code)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(), endpoint, status, latency_ms, tokens_in, tokens_out,
                    model, workflow, api_key_id, error_code,
                ),
            )

    def record(
        self,
        *,
        endpoint: str,
        status: int,
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        model: str | None = None,
        workflow: str | None = None,
        api_key_id: str | None = None,
        error_code: str | None = None,
    ) -> None:
        """Fire-and-forget async write so the event loop is never blocked."""
        asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._record_sync(
                endpoint=endpoint,
                status=status,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model=model,
                workflow=workflow,
                api_key_id=api_key_id,
                error_code=error_code,
            ),
        )

    def prune(self, retention_days: int = 30) -> int:
        """Delete rows older than *retention_days*. Returns row count deleted."""
        cutoff = time.time() - retention_days * 86400
        with self._lock, self._conn() as conn:
            cur = conn.execute("DELETE FROM request_metrics WHERE ts < ?", (cutoff,))
            return cur.rowcount

    def summary(self, window_s: int = 300) -> dict[str, float | int]:
        since = time.time() - window_s
        with self._conn() as conn:
            row = conn.execute(
                """SELECT
                       COUNT(*) AS n,
                       COALESCE(AVG(latency_ms), 0) AS avg_latency_ms,
                       COALESCE(SUM(tokens_in), 0) AS tokens_in,
                       COALESCE(SUM(tokens_out), 0) AS tokens_out,
                       COALESCE(SUM(CASE WHEN status >= 500 THEN 1 ELSE 0 END), 0) AS errors
                   FROM request_metrics WHERE ts >= ?""",
                (since,),
            ).fetchone()
        return {
            "window_s": window_s,
            "requests": int(row["n"]),
            "avg_latency_ms": float(row["avg_latency_ms"]),
            "tokens_in": int(row["tokens_in"]),
            "tokens_out": int(row["tokens_out"]),
            "errors": int(row["errors"]),
        }
