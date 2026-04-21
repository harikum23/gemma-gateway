from __future__ import annotations

import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Request
from loguru import logger

from gateway.errors import UnauthorizedError


_HASHER = PasswordHasher()
SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    key_id        TEXT NOT NULL UNIQUE,
    key_hash      TEXT NOT NULL,
    label         TEXT NOT NULL DEFAULT '',
    is_admin      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    last_used_at  TEXT,
    revoked       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_id ON api_keys(key_id);
"""


@dataclass(frozen=True)
class ApiKeyRecord:
    id: int
    key_id: str
    label: str
    is_admin: bool


class ApiKeyStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _split(raw: str) -> tuple[str, str]:
        if "." not in raw:
            raise UnauthorizedError()
        key_id, secret = raw.split(".", 1)
        if not key_id or not secret:
            raise UnauthorizedError()
        return key_id, secret

    def generate(self, label: str = "", *, is_admin: bool = False) -> str:
        key_id = "gk_" + secrets.token_urlsafe(6)
        secret = secrets.token_urlsafe(32)
        raw = f"{key_id}.{secret}"
        hashed = _HASHER.hash(secret)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO api_keys (key_id, key_hash, label, is_admin) VALUES (?, ?, ?, ?)",
                (key_id, hashed, label, 1 if is_admin else 0),
            )
        return raw

    def bootstrap(self, raw_key: str, label: str = "bootstrap", *, is_admin: bool = True) -> None:
        key_id, secret = self._split(raw_key)
        hashed = _HASHER.hash(secret)
        with self._conn() as conn:
            cur = conn.execute("SELECT 1 FROM api_keys WHERE key_id = ?", (key_id,))
            if cur.fetchone() is not None:
                return
            conn.execute(
                "INSERT INTO api_keys (key_id, key_hash, label, is_admin) VALUES (?, ?, ?, ?)",
                (key_id, hashed, label, 1 if is_admin else 0),
            )

    def count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM api_keys WHERE revoked = 0").fetchone()
            return int(row["n"])

    def list_keys(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT key_id, label, is_admin, created_at, last_used_at, revoked FROM api_keys ORDER BY id DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def revoke(self, key_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("UPDATE api_keys SET revoked = 1 WHERE key_id = ?", (key_id,))
            return cur.rowcount > 0

    def verify(self, raw: str) -> ApiKeyRecord:
        key_id, secret = self._split(raw)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, key_id, key_hash, label, is_admin, revoked FROM api_keys WHERE key_id = ?",
                (key_id,),
            ).fetchone()
        if row is None or int(row["revoked"]) == 1:
            raise UnauthorizedError()
        try:
            _HASHER.verify(row["key_hash"], secret)
        except VerifyMismatchError as e:
            raise UnauthorizedError() from e
        with self._conn() as conn:
            conn.execute(
                "UPDATE api_keys SET last_used_at = datetime('now') WHERE id = ?",
                (row["id"],),
            )
        return ApiKeyRecord(
            id=int(row["id"]),
            key_id=row["key_id"],
            label=row["label"],
            is_admin=int(row["is_admin"]) == 1,
        )


def ensure_bootstrap_key(store: ApiKeyStore, bootstrap_key: str | None) -> str | None:
    if bootstrap_key:
        store.bootstrap(bootstrap_key)
        logger.info("api-key bootstrap: using provided GATEWAY_BOOTSTRAP_API_KEY")
        return None
    if store.count() > 0:
        return None
    generated = store.generate(label="auto-bootstrap", is_admin=True)
    logger.warning(
        "api-key bootstrap: generated initial admin key (copy now; not shown again)\n"
        f"    GATEWAY_BOOTSTRAP_API_KEY={generated}"
    )
    return generated


def extract_bearer(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.lower().startswith("bearer "):
        raise UnauthorizedError()
    return auth.split(" ", 1)[1].strip()


def require_api_key(request: Request) -> ApiKeyRecord:
    store: ApiKeyStore = request.app.state.api_keys
    return store.verify(extract_bearer(request))


def require_admin_key(request: Request) -> ApiKeyRecord:
    rec = require_api_key(request)
    if not rec.is_admin:
        raise UnauthorizedError("admin privileges required")
    return rec
