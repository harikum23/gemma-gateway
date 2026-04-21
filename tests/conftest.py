from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("GATEWAY_DATA_DIR", str(data))
    monkeypatch.setenv("GATEWAY_ENGINE_URL", "http://ollama.invalid")
    monkeypatch.setenv("GATEWAY_BOOTSTRAP_API_KEY", "gk_test1.secret_for_tests_0123456789")
    monkeypatch.setenv("GATEWAY_LOG_LEVEL", "WARNING")
    # Force fresh settings singleton each test session.
    from gateway import settings as settings_mod

    settings_mod._settings = None
    yield data
    settings_mod._settings = None
