from __future__ import annotations

from pathlib import Path

import pytest

from gateway.auth import ApiKeyStore, ensure_bootstrap_key
from gateway.errors import UnauthorizedError


def test_generate_and_verify(tmp_path: Path) -> None:
    store = ApiKeyStore(tmp_path / "k.db")
    raw = store.generate(label="test", is_admin=True)
    record = store.verify(raw)
    assert record.is_admin is True
    assert record.label == "test"


def test_reject_unknown_key(tmp_path: Path) -> None:
    store = ApiKeyStore(tmp_path / "k.db")
    with pytest.raises(UnauthorizedError):
        store.verify("gk_missing.whatever")


def test_reject_bad_secret(tmp_path: Path) -> None:
    store = ApiKeyStore(tmp_path / "k.db")
    raw = store.generate()
    key_id = raw.split(".", 1)[0]
    with pytest.raises(UnauthorizedError):
        store.verify(f"{key_id}.wrongsecret")


def test_revoke(tmp_path: Path) -> None:
    store = ApiKeyStore(tmp_path / "k.db")
    raw = store.generate()
    key_id = raw.split(".", 1)[0]
    assert store.revoke(key_id) is True
    with pytest.raises(UnauthorizedError):
        store.verify(raw)


def test_bootstrap_is_idempotent(tmp_path: Path) -> None:
    store = ApiKeyStore(tmp_path / "k.db")
    raw = "gk_boot.secretsecretsecret"
    ensure_bootstrap_key(store, raw)
    ensure_bootstrap_key(store, raw)
    assert store.count() == 1
    rec = store.verify(raw)
    assert rec.is_admin is True


def test_auto_bootstrap_on_empty(tmp_path: Path) -> None:
    store = ApiKeyStore(tmp_path / "k.db")
    generated = ensure_bootstrap_key(store, None)
    assert generated is not None
    assert store.count() == 1
    assert store.verify(generated).is_admin is True
