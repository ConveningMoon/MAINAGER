from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mainager.config import Settings, load_settings


def test_auth_header_uses_bearer_scheme(settings: Settings) -> None:
    assert settings.auth_header == {"Authorization": "Bearer oc_test"}


def test_token_is_not_exposed_by_repr(settings: Settings) -> None:
    assert "oc_test" not in repr(settings)


def test_missing_token_fails_loudly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VIBE_API_TOKEN", raising=False)
    monkeypatch.chdir(tmp_path)  # step away from any local .env
    with pytest.raises(ValidationError):
        load_settings()
