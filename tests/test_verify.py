"""Each `mainager verify` check, against mocked HTTP.

`GET /capabilities` is mocked with the real committed snapshot rather than a
hand-built fixture, so a check that quietly relies on a field the live catalog
no longer has fails here the same way it would live. `seedream-5-lite-edit` is
a deliberate example of that: it is priced but absent from the catalog (it is
one of finding 4's own "priced_without_model" identifiers), which is exactly
why check 9 does not use it as a pair.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from mainager.config import Settings
from mainager.preflight.registry import ModelRegistry
from mainager.verify import (
    DOCS_URL,
    _catalog_parameter_names,
    _mcp_declared_param_names,
    check_slug,
    resolve_selection,
    run_all,
)

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data"
CAPABILITIES = json.loads((SNAPSHOT_DIR / "capabilities.json").read_text(encoding="utf-8"))

BASE = "https://api.test/agent"

# A minimal but internally consistent /prices payload. Deliberately omits
# seedream-5-lite-edit's "priced_without_model" partner-in-crime so check 4 has
# something to find, and omits grok-itv-30 so check 3's estimate is the only
# source for that figure (matching how the real API behaves).
PRICES = {
    "status": "ok",
    "prices": {
        "video": {"grok-ttv": 36, "grok-itv": 36},
        "other": {"seedream-5-lite-edit": 18, "resize-2k": 30},
        "image": {"z-image": 1.2},
    },
}

DOCS_HTML = """
<html><body>
<p>Авто-ответчик: если агент не забрал сообщение за ~4 секунд, платформа
отвечает сама. Списание с баланса владельца по прайсу текстового чата:
Claude Opus 4.8 — 10₽/ответ, страховочный Sonnet 4.6 — 8₽/ответ.</p>
</body></html>
"""


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        VIBE_API_TOKEN="oc_test",  # type: ignore[call-arg]
        VIBE_API_BASE_URL=BASE,
        MAINAGER_DATA_DIR=tmp_path,
    )


def _mock_common(respx_mock: respx.MockRouter, *, estimate_side_effect) -> None:  # type: ignore[no-untyped-def]
    respx_mock.get(f"{BASE}/capabilities").mock(return_value=httpx.Response(200, json=CAPABILITIES))
    respx_mock.get(f"{BASE}/prices").mock(return_value=httpx.Response(200, json=PRICES))
    respx_mock.get(f"{BASE}/balance").mock(
        return_value=httpx.Response(200, json={"status": "ok", "balance": 600})
    )
    for path in ("/me", "/health", "/generations", "/inbox"):
        respx_mock.get(f"{BASE}{path}").mock(return_value=httpx.Response(200, json={"ok": True}))
    respx_mock.post(f"{BASE}/generate/estimate").mock(side_effect=estimate_side_effect)
    respx_mock.get(DOCS_URL).mock(return_value=httpx.Response(200, text=DOCS_HTML))


def _estimate_response(
    cost: float, *, valid: bool = True, rejected: list[str] | None = None
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "valid": valid,
            "estimated_cost_rub": cost,
            "rejected": rejected or [],
            "balance": {"current": 600, "after": 600 - cost},
        },
    )


def _default_estimate(request: httpx.Request) -> httpx.Response:
    """Replays the measured 28.07 figures for any recognised model+body."""
    body = json.loads(request.content)
    model = body.get("model")

    if model == "grok-ttv" and "image_input" in body:
        return _estimate_response(196, valid=True, rejected=["image_input"])
    if model == "grok-ttv-30":
        return _estimate_response(436)
    if model == "grok-itv":
        return _estimate_response(36)
    if model == "grok-itv-20":
        return _estimate_response(316)
    if model == "nano-banana-2-lite":
        return _estimate_response(9)
    return _estimate_response(1.2)


async def test_all_checks_pass_against_the_measured_figures(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    """The happy path: replaying 28.07's numbers should match every claim."""
    _mock_common(respx_mock, estimate_side_effect=_default_estimate)

    results = await run_all(_settings(tmp_path))

    assert len(results) == 10
    outcomes = {r.n: r.outcome for r in results}
    # Checks 1 and 10 (MCP) have no network mock here and are expected to
    # error cleanly rather than crash the run — check 1 covered on its own
    # below; check 10 needs the same tools/list call.
    assert outcomes[1] == "error"
    assert outcomes[10] == "error"
    for n in (2, 3, 5, 6, 7, 8, 9):
        assert outcomes[n] == "match", f"check {n}: {[r for r in results if r.n == n]}"
    # check 4 is the one finding whose "match" means the two sources disagree;
    # our PRICES fixture deliberately disagrees with the catalog.
    assert outcomes[4] == "match"


async def test_check_1_reports_a_clean_error_without_mcp_configured(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    """No MCP server is mocked, so this proves the run does not crash without one."""
    _mock_common(respx_mock, estimate_side_effect=_default_estimate)

    results = await run_all(_settings(tmp_path))

    check_1 = next(r for r in results if r.n == 1)
    assert check_1.outcome == "error"
    assert "strict" in check_1.claim


async def test_check_2_flags_it_when_valid_is_finally_false(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    """If the provider fixes the bug, valid:true+rejected should stop happening."""

    def fixed_estimate(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("model") == "grok-ttv" and "image_input" in body:
            return _estimate_response(196, valid=False, rejected=["image_input"])
        return _default_estimate(request)

    _mock_common(respx_mock, estimate_side_effect=fixed_estimate)

    results = await run_all(_settings(tmp_path))

    check_2 = next(r for r in results if r.n == 2)
    assert check_2.outcome == "differs"
    assert "valid=False" in check_2.measured


async def test_check_3_flags_it_when_the_gap_closes(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    def small_gap(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("model") == "grok-ttv-30":
            return _estimate_response(72)  # 2x, not >= 10x
        return _default_estimate(request)

    _mock_common(respx_mock, estimate_side_effect=small_gap)

    results = await run_all(_settings(tmp_path))

    check_3 = next(r for r in results if r.n == 3)
    assert check_3.outcome == "differs"
    assert "ratio=2.0x" in check_3.measured


async def test_check_4_reports_match_when_sources_disagree(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    _mock_common(respx_mock, estimate_side_effect=_default_estimate)

    results = await run_all(_settings(tmp_path))

    check_4 = next(r for r in results if r.n == 4)
    assert check_4.outcome == "match"
    assert "model(s) with no price" in check_4.measured


async def test_check_4_reports_differs_when_sources_agree(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    """A clean reconciliation is good news, but it is not what finding 4 claims."""
    respx_mock.get(f"{BASE}/capabilities").mock(return_value=httpx.Response(200, json=CAPABILITIES))
    # Every routable id priced, nothing extra: reconcile() sees no drift. Must
    # cover tier-expanded ids too (grok-itv-10 etc.), not just the top-level
    # catalog keys — the registry routes on the expanded set.
    registry = ModelRegistry.from_capabilities(CAPABILITIES)
    all_priced = {"prices": {"image": {}, "video": {}, "voice": {}, "music": {}, "other": {}}}
    for model_id in registry.ids():
        media_type = registry.get(model_id).media_type
        all_priced["prices"].setdefault(media_type, {})[model_id] = 1
    respx_mock.get(f"{BASE}/prices").mock(return_value=httpx.Response(200, json=all_priced))
    respx_mock.get(f"{BASE}/balance").mock(
        return_value=httpx.Response(200, json={"status": "ok", "balance": 600})
    )
    for path in ("/me", "/health", "/generations", "/inbox"):
        respx_mock.get(f"{BASE}{path}").mock(return_value=httpx.Response(200, json={"ok": True}))
    respx_mock.post(f"{BASE}/generate/estimate").mock(side_effect=_default_estimate)
    respx_mock.get(DOCS_URL).mock(return_value=httpx.Response(200, text=DOCS_HTML))

    results = await run_all(_settings(tmp_path))

    check_4 = next(r for r in results if r.n == 4)
    assert check_4.outcome == "differs"
    assert check_4.note != ""


async def test_check_5_uses_the_real_snapshots_tier_contradiction(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    """Against the committed snapshot, getting_started really does name a -30
    tier that models.video['grok-itv'].tiers omits."""
    _mock_common(respx_mock, estimate_side_effect=_default_estimate)

    results = await run_all(_settings(tmp_path))

    check_5 = next(r for r in results if r.n == 5)
    assert check_5.outcome == "match"
    assert "30" in check_5.measured


async def test_check_6_flags_it_when_the_multiplier_shrinks(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    def flat_pricing(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("model") == "grok-itv-20":
            return _estimate_response(40)  # ~1.1x, not ~8.8x
        return _default_estimate(request)

    _mock_common(respx_mock, estimate_side_effect=flat_pricing)

    results = await run_all(_settings(tmp_path))

    check_6 = next(r for r in results if r.n == 6)
    assert check_6.outcome == "differs"


async def test_check_7_flags_it_if_the_provider_documents_the_endpoints(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    """If they add /me etc. to capabilities.endpoints, the finding is resolved
    and the check should say so rather than silently keep reporting match."""
    documented = dict(CAPABILITIES)
    documented["endpoints"] = {
        **CAPABILITIES["endpoints"],
        "GET /api/agent/me": "now documented",
    }
    respx_mock.get(f"{BASE}/capabilities").mock(return_value=httpx.Response(200, json=documented))
    respx_mock.get(f"{BASE}/prices").mock(return_value=httpx.Response(200, json=PRICES))
    respx_mock.get(f"{BASE}/balance").mock(
        return_value=httpx.Response(200, json={"status": "ok", "balance": 600})
    )
    for path in ("/me", "/health", "/generations", "/inbox"):
        respx_mock.get(f"{BASE}{path}").mock(return_value=httpx.Response(200, json={"ok": True}))
    respx_mock.post(f"{BASE}/generate/estimate").mock(side_effect=_default_estimate)
    respx_mock.get(DOCS_URL).mock(return_value=httpx.Response(200, text=DOCS_HTML))

    results = await run_all(_settings(tmp_path))

    check_7 = next(r for r in results if r.n == 7)
    assert check_7.outcome == "differs"
    assert "documented=True" in check_7.measured


async def test_check_8_reads_the_public_docs_page(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    _mock_common(respx_mock, estimate_side_effect=_default_estimate)

    results = await run_all(_settings(tmp_path))

    check_8 = next(r for r in results if r.n == 8)
    assert check_8.outcome == "match"


async def test_check_8_flags_it_when_the_docs_change(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    _mock_common(respx_mock, estimate_side_effect=_default_estimate)
    respx_mock.get(DOCS_URL).mock(return_value=httpx.Response(200, text="<html>rewritten</html>"))

    results = await run_all(_settings(tmp_path))

    check_8 = next(r for r in results if r.n == 8)
    assert check_8.outcome == "differs"
    assert check_8.note != ""


async def test_check_9_avoids_the_delisted_edit_pair(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    """seedream-5-lite-edit is priced-without-model; check 9 must not choke on it."""
    _mock_common(respx_mock, estimate_side_effect=_default_estimate)

    results = await run_all(_settings(tmp_path))

    check_9 = next(r for r in results if r.n == 9)
    assert check_9.outcome == "match"
    assert "seedream-5-lite-edit" not in check_9.measured
    assert "seedream-5-pro" in check_9.measured


async def test_a_broken_check_does_not_hide_the_others(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    """/generate/estimate erroring must not take down checks that don't need it."""

    def half_broken(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal"})

    _mock_common(respx_mock, estimate_side_effect=half_broken)

    results = await run_all(_settings(tmp_path))

    assert len(results) == 10
    # checks 4, 5, 7 never call /generate/estimate and should still resolve.
    for n in (4, 5, 7):
        assert next(r for r in results if r.n == n).outcome != "error"
    # checks that do call it should report the failure as an error, not crash.
    for n in (2, 3, 6, 9):
        assert next(r for r in results if r.n == n).outcome == "error"
    # check 10 needs MCP tools/list, which is not mocked in this scenario either.
    assert next(r for r in results if r.n == 10).outcome == "error"


async def test_nothing_ever_posts_to_generate_or_calls_generate_content(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    """The one property that must never break: verify() only reads."""
    charge_route = respx_mock.post(f"{BASE}/generate").mock(
        return_value=httpx.Response(200, json={"status": "processing"})
    )
    _mock_common(respx_mock, estimate_side_effect=_default_estimate)

    await run_all(_settings(tmp_path))

    assert charge_route.call_count == 0


def test_catalog_parameter_names_unions_models_and_text_models() -> None:
    capabilities = {
        "models": {
            "image": {
                "z-image": {"required": ["prompt"], "optional": ["aspect_ratio"]},
            },
            "video": {
                "grok-itv": {
                    "required": ["prompt", "image_urls"],
                    "optional": ["duration"],
                    "enums": {"resolution": ["480p", "720p"]},
                },
            },
        },
        "text_models": {"params": ["prompt (required)", "effort (low|medium|high)", "thinking"]},
    }

    names = _catalog_parameter_names(capabilities)

    assert names == {"aspect_ratio", "image_urls", "duration", "resolution", "effort", "thinking"}
    assert "prompt" not in names  # universal field, already a top-level tool argument


def test_mcp_declared_param_names_reads_the_nested_schema() -> None:
    empty = {"type": "object", "properties": {"params": {"type": "object", "description": "..."}}}
    assert _mcp_declared_param_names(empty) == frozenset()

    typed = {
        "type": "object",
        "properties": {
            "params": {
                "type": "object",
                "properties": {"aspect_ratio": {"type": "string"}, "duration": {"type": "integer"}},
            }
        },
    }
    assert _mcp_declared_param_names(typed) == {"aspect_ratio", "duration"}
    assert _mcp_declared_param_names(None) == frozenset()


def test_resolve_selection_accepts_numbers_and_slugs() -> None:
    by_number = resolve_selection(("1",))
    by_slug = resolve_selection(("mcp_strict",))
    assert [check_slug(c) for c in by_number] == ["mcp_strict"]
    assert by_number == by_slug


def test_resolve_selection_rejects_unknown_selectors() -> None:
    with pytest.raises(ValueError, match="unknown check"):
        resolve_selection(("nonexistent_slug",))


async def test_run_all_with_only_runs_a_single_check(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    """The isolated-demo path: one finding, one command, one row back."""
    _mock_common(respx_mock, estimate_side_effect=_default_estimate)

    results = await run_all(_settings(tmp_path), only=("edit_pricing",))

    assert len(results) == 1
    assert results[0].n == 9
