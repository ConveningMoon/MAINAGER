"""Fetch and persist the ``GET /capabilities`` catalog.

The catalog is the source of truth for which models exist, what parameters they
take and what limits they enforce. Everything downstream is built from it, and a
snapshot on disk is what the tests run against, so the fetch is deliberately dumb:
no reshaping, no filtering, no schema opinion. Interpretation happens in preflight.

``/capabilities`` is a free read endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from mainager.config import Settings
from mainager.providers.vibemarketolog.errors import VibeApiError

CAPABILITIES_PATH = "/capabilities"
SNAPSHOT_FILENAME = "capabilities.json"


async def fetch_capabilities(settings: Settings) -> dict[str, Any]:
    """Read the live catalog. Raises ``VibeApiError`` on any non-2xx response."""
    async with httpx.AsyncClient(
        base_url=settings.api_base_url,
        headers=settings.auth_header,
        timeout=settings.http_timeout_s,
    ) as client:
        response = await client.get(CAPABILITIES_PATH)

    if response.is_error:
        raise VibeApiError.from_response(response)

    payload = response.json()
    if not isinstance(payload, dict):
        raise VibeApiError(
            response.status_code,
            "unexpected_payload",
            f"expected a JSON object, got {type(payload).__name__}",
        )
    return payload


def write_snapshot(payload: dict[str, Any], data_dir: Path) -> Path:
    """Write the catalog to ``<data_dir>/capabilities.json``, sorted and indented.

    Stable formatting keeps the diff readable when the catalog drifts, which is
    the point of tracking the snapshot at all.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / SNAPSHOT_FILENAME
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def read_snapshot(data_dir: Path) -> dict[str, Any]:
    """Load the tracked catalog snapshot."""
    payload: Any = json.loads((data_dir / SNAPSHOT_FILENAME).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{SNAPSHOT_FILENAME} does not contain a JSON object")
    return payload
