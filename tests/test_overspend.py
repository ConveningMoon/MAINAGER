"""Retrospective overspend analysis over an account's generation history."""

from __future__ import annotations

from pathlib import Path

import pytest

from mainager.postflight.overspend import (
    overspend_report,
    parse_generations,
    reconstruct_intent,
)
from mainager.preflight.registry import ModelRegistry
from mainager.providers.vibemarketolog.capabilities import read_snapshot
from tests.test_routing import StubPricer

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    return ModelRegistry.from_capabilities(read_snapshot(SNAPSHOT_DIR))


def _entry(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "gen_1",
        "type": "video",
        "model": "grok-itv-20",
        "cost": 316,
        "status": "complete",
        "params": {
            "prompt": "slow pan over the product",
            "duration": 5,
            "image_urls": ["https://cdn.example/a.png"],
        },
    }
    base.update(overrides)
    return base


def test_parse_tolerates_a_sparse_record() -> None:
    parsed = parse_generations({"generations": [{"type": "image", "model": "z-image"}]})

    assert len(parsed) == 1
    assert parsed[0].cost_rub == 0.0
    assert parsed[0].params == {}
    assert parsed[0].generation_id == "0"  # falls back to the index


def test_parse_accepts_an_empty_history() -> None:
    assert parse_generations({"generations": [], "total": 0}) == ()
    assert parse_generations({"status": "ok"}) == ()


def test_missing_prompt_makes_a_record_unanalysable(registry: ModelRegistry) -> None:
    generation = parse_generations({"generations": [_entry(params={"duration": 5})]})[0]

    intent, reason = reconstruct_intent(generation, registry)

    assert intent is None
    assert reason is not None
    assert "prompt" in reason


def test_missing_duration_makes_a_duration_priced_record_unanalysable(
    registry: ModelRegistry,
) -> None:
    generation = parse_generations({"generations": [_entry(params={"prompt": "x"})]})[0]

    intent, reason = reconstruct_intent(generation, registry)

    assert intent is None
    assert reason is not None
    assert "duration" in reason


def test_retired_model_is_reported_not_guessed(registry: ModelRegistry) -> None:
    generation = parse_generations({"generations": [_entry(model="veo2-legacy")]})[0]

    intent, reason = reconstruct_intent(generation, registry)

    assert intent is None
    assert reason is not None
    assert "no longer in the catalog" in reason


async def test_pinned_tier_history_shows_the_avoidable_spend(
    registry: ModelRegistry,
) -> None:
    """The measured case: 316 RUB paid for a 5s clip that routes at 36 today."""
    history = parse_generations({"generations": [_entry()]})

    report = await overspend_report(history, registry, StubPricer())

    assert len(report.analysed) == 1
    item = report.analysed[0]
    assert item.model_id == "grok-itv-20"
    assert item.paid_rub == 316.0
    assert item.cheapest_model_id == "grok-itv"
    assert item.cheapest_rub == 36.0
    assert item.saving_rub == 280.0
    assert report.overspend_rub == 280.0


async def test_a_well_routed_generation_shows_no_saving(registry: ModelRegistry) -> None:
    history = parse_generations({"generations": [_entry(model="grok-itv", cost=36)]})

    report = await overspend_report(history, registry, StubPricer())

    assert report.overspend_rub == 0.0
    assert report.analysed[0].saving_rub == 0.0


async def test_refunded_generations_are_excluded_from_the_total(
    registry: ModelRegistry,
) -> None:
    history = parse_generations({"generations": [_entry(refunded=True)]})

    report = await overspend_report(history, registry, StubPricer())

    assert report.analysed == ()
    assert len(report.skipped) == 1
    assert "refunded" in (report.skipped[0].unanalysable_reason or "")
    assert report.overspend_rub == 0.0


async def test_unanalysable_entries_do_not_pollute_the_total(
    registry: ModelRegistry,
) -> None:
    history = parse_generations(
        {
            "generations": [
                _entry(),
                _entry(id="gen_2", model="veo2-legacy", cost=999),
            ]
        }
    )

    report = await overspend_report(history, registry, StubPricer())

    assert report.paid_rub == 316.0  # the 999 is excluded, not counted
    assert report.overspend_rub == 280.0
    assert len(report.skipped) == 1
    assert "not analysable" in report.summary()


async def test_summary_reports_the_share_as_well_as_the_amount(
    registry: ModelRegistry,
) -> None:
    report = await overspend_report(
        parse_generations({"generations": [_entry()]}), registry, StubPricer()
    )

    summary = report.summary()
    assert "paid 316 RUB" in summary
    assert "avoidable 280 RUB" in summary
    assert "89%" in summary


async def test_empty_history_says_so_rather_than_reporting_zero_savings(
    registry: ModelRegistry,
) -> None:
    report = await overspend_report((), registry, StubPricer())
    assert "nothing analysable" in report.summary()
