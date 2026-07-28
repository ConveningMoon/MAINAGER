from __future__ import annotations

import httpx
import pytest
import respx

from mainager.config import Settings
from mainager.providers.vibemarketolog.capabilities import (
    fetch_capabilities,
    read_snapshot,
    write_snapshot,
)
from mainager.providers.vibemarketolog.errors import VibeApiError

CATALOG = {"models": [{"id": "grok-itv-10", "type": "video"}], "version": "2026-07-01"}


@respx.mock
async def test_fetch_sends_bearer_token_and_returns_catalog(settings: Settings) -> None:
    route = respx.get("https://api.test/agent/capabilities").mock(
        return_value=httpx.Response(200, json=CATALOG)
    )

    assert await fetch_capabilities(settings) == CATALOG
    assert route.calls.last.request.headers["Authorization"] == "Bearer oc_test"


@respx.mock
async def test_insufficient_scope_surfaces_the_scope_gap(settings: Settings) -> None:
    """Shape taken from a real 403 on GET /yandex with a read+generate key."""
    respx.get("https://api.test/agent/capabilities").mock(
        return_value=httpx.Response(
            403,
            json={
                "status": "error",
                "error": "insufficient_scope",
                "message": "no yandex right for this method",
                "required": "yandex",
                "granted": ["read", "generate"],
                "request_id": "79dcf2f7-1fdf-4a8f-bcb4-130d69a3bb66",
            },
        )
    )

    with pytest.raises(VibeApiError) as excinfo:
        await fetch_capabilities(settings)

    assert excinfo.value.code == "insufficient_scope"
    assert excinfo.value.required_scope == "yandex"
    assert excinfo.value.granted_scopes == ["read", "generate"]
    assert excinfo.value.request_id == "79dcf2f7-1fdf-4a8f-bcb4-130d69a3bb66"


@respx.mock
async def test_non_json_error_body_still_produces_a_typed_error(settings: Settings) -> None:
    respx.get("https://api.test/agent/capabilities").mock(
        return_value=httpx.Response(502, text="<html>bad gateway</html>")
    )

    with pytest.raises(VibeApiError) as excinfo:
        await fetch_capabilities(settings)

    assert excinfo.value.status_code == 502
    assert excinfo.value.code == "http_502"
    assert excinfo.value.required_scope is None
    assert excinfo.value.granted_scopes == []
    assert excinfo.value.request_id is None


def test_snapshot_roundtrips(settings: Settings) -> None:
    destination = write_snapshot(CATALOG, settings.data_dir)

    assert destination.name == "capabilities.json"
    assert read_snapshot(settings.data_dir) == CATALOG
