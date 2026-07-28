from __future__ import annotations

from pathlib import Path

import pytest

from mainager.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        VIBE_API_TOKEN="oc_test",  # type: ignore[call-arg]
        VIBE_API_BASE_URL="https://api.test/agent",
        MAINAGER_DATA_DIR=tmp_path,
    )
