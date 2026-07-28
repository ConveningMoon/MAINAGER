"""Router, planner and catalog reconciliation.

Prices come from a stub that replays figures measured against the live dry-run
endpoint on 2026-07-28, so these run offline while still asserting the real
economics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mainager.preflight.compiler import Intent
from mainager.preflight.planner import (
    AccountState,
    PlanCycleError,
    PlanStep,
    estimate_plan,
    execution_order,
)
from mainager.preflight.pricing import Estimate
from mainager.preflight.reconcile import (
    flatten_prices,
    reconcile,
    unlisted_tiers,
)
from mainager.preflight.registry import ModelRegistry
from mainager.preflight.router import NoViableModelError, candidate_specs, route
from mainager.providers.vibemarketolog.capabilities import read_snapshot
from mainager.providers.vibemarketolog.pricing import parse_estimate

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data"

# Measured 2026-07-28 via POST /generate/estimate.
MEASURED_VIDEO_PRICES: dict[tuple[str, int], float] = {
    ("grok-itv", 5): 36.0,
    ("grok-itv", 10): 196.0,
    ("grok-itv", 15): 316.0,
    ("grok-itv-10", 5): 196.0,
    ("grok-itv-20", 5): 316.0,
    ("veo3_fast", 5): 120.0,
    ("veo3_fast", 10): 120.0,
    ("veo3_fast", 15): 120.0,
    ("veo3.1", 5): 120.0,
    ("veo3.1", 10): 120.0,
    ("veo3.1", 15): 120.0,
    ("veo3", 5): 570.0,
    ("veo3", 10): 570.0,
    ("veo3", 15): 570.0,
    ("kling-3.0-std", 5): 250.0,
    ("kling-3.0-std", 10): 250.0,
    ("kling-3.0-std", 15): 250.0,
    ("kling-3.0-pro", 5): 549.0,
    ("kling-3.0-pro", 10): 549.0,
    ("kling-3.0-pro", 15): 549.0,
    ("seedance-2-mini", 5): 70.0,
    ("seedance-2-mini", 10): 140.0,
    ("seedance-2-mini", 15): 210.0,
    ("seedance-2-fast", 10): 708.0,
    ("seedance-2", 10): 1188.0,
    ("gemini-omni-video", 5): 149.0,
    ("gemini-omni-video", 10): 149.0,
    ("gemini-omni-video", 15): 149.0,
}

IMAGE_PRICES = {"z-image": 1.2, "nano-banana-2-lite": 9.0, "seedream-5-pro": 40.0}


class StubPricer:
    """Replays measured prices; records what it was asked."""

    def __init__(self, *, balance: float = 600.0, daily_remaining: float = 500.0) -> None:
        self.balance = balance
        self.daily_remaining = daily_remaining
        self.calls: list[dict[str, Any]] = []

    async def estimate(self, body: dict[str, Any]) -> Estimate:
        self.calls.append(body)
        model_id = str(body["model"])
        duration = body.get("duration")
        if model_id in IMAGE_PRICES:
            cost = IMAGE_PRICES[model_id]
        else:
            key = (model_id, int(duration) if isinstance(duration, int) else 10)
            cost = MEASURED_VIDEO_PRICES.get(key, 999.0)

        warnings: list[str] = []
        if cost > self.balance:
            warnings.append(f"insufficient_balance: need {cost:g}RUB, have {self.balance:g}RUB.")
        if cost > self.daily_remaining:
            warnings.append("daily_spend_limit would be exceeded.")

        return parse_estimate(
            {
                "model": model_id,
                "estimated_cost_rub": cost,
                "valid": not warnings,
                "rejected": [],
                "warnings": warnings,
                "balance": {"current": self.balance, "after": self.balance - cost},
                "daily_spend": {"within_limit": cost <= self.daily_remaining},
            },
            model_id,
        )


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    return ModelRegistry.from_capabilities(read_snapshot(SNAPSHOT_DIR))


# --- routing ---------------------------------------------------------------


def test_pinned_tiers_are_excluded_from_candidates_by_default(
    registry: ModelRegistry,
) -> None:
    intent = Intent(media_type="video", prompt="a cat", source_image_url="https://x/a.png")
    ids = {spec.model_id for spec in candidate_specs(intent, registry)}

    assert "grok-itv" in ids
    assert "grok-itv-10" not in ids
    assert {
        spec.model_id for spec in candidate_specs(intent, registry, allow_pinned_tiers=True)
    } > ids


async def test_short_clip_routes_to_the_tier_selecting_base_id(registry: ModelRegistry) -> None:
    """5s: grok-itv at 36 beats veo3_fast at 120."""
    intent = Intent(
        media_type="video", prompt="a cat", source_image_url="https://x/a.png", duration_s=5
    )

    decision = await route(intent, registry, StubPricer())

    assert decision.spec.model_id == "grok-itv"
    assert decision.cost_rub == 36.0
    assert decision.body["strict"] is True


async def test_the_cheapest_model_changes_with_duration(registry: ModelRegistry) -> None:
    """15s: grok-itv climbs to 316, so a flat-priced veo model wins instead.

    veo3.1 and veo3_fast both cost 120; the tie is broken by model id so the
    choice is reproducible.
    """
    intent = Intent(
        media_type="video", prompt="a cat", source_image_url="https://x/a.png", duration_s=15
    )

    decision = await route(intent, registry, StubPricer())

    assert decision.cost_rub == 120.0
    assert decision.spec.model_id in {"veo3.1", "veo3_fast"}
    assert decision.spec.model_id != "grok-itv"


async def test_catalog_price_ordering_would_have_picked_the_dearer_model(
    registry: ModelRegistry,
) -> None:
    """The regression this router exists to prevent.

    The catalog lists grok-itv at 36 and veo3_fast at 120, so ranking by the
    catalog picks grok-itv for a 15s clip. It actually costs 316.
    """
    assert registry.get("grok-itv").price == 36
    assert registry.get("veo3_fast").price == 120

    intent = Intent(
        media_type="video", prompt="a cat", source_image_url="https://x/a.png", duration_s=15
    )
    decision = await route(intent, registry, StubPricer())

    assert decision.cost_rub == 120.0
    assert MEASURED_VIDEO_PRICES[("grok-itv", 15)] / decision.cost_rub > 2.5


async def test_budget_ceiling_excludes_models_and_is_recorded(registry: ModelRegistry) -> None:
    intent = Intent(
        media_type="video", prompt="a cat", source_image_url="https://x/a.png", duration_s=5
    )

    decision = await route(intent, registry, StubPricer(), budget_ceiling_rub=100.0)

    assert decision.spec.model_id == "grok-itv"
    assert decision.cost_rub == 36.0
    over = {c.model_id for c in decision.considered if c.status == "over_ceiling"}
    assert "veo3_fast" in over
    assert "kling-3.0-std" in over


async def test_every_candidate_is_accounted_for_in_the_decision(
    registry: ModelRegistry,
) -> None:
    intent = Intent(
        media_type="video", prompt="a cat", source_image_url="https://x/a.png", duration_s=5
    )
    decision = await route(intent, registry, StubPricer())

    statuses = {c.model_id: c.status for c in decision.considered}
    assert statuses["grok-itv"] == "chosen"
    assert statuses["grok-ttv"] == "incompatible"  # text-to-video takes no image
    assert all(c.reason for c in decision.considered if c.status == "incompatible")


async def test_no_viable_model_carries_the_audit_trail(registry: ModelRegistry) -> None:
    intent = Intent(
        media_type="video", prompt="a cat", source_image_url="https://x/a.png", duration_s=5
    )

    with pytest.raises(NoViableModelError) as excinfo:
        await route(intent, registry, StubPricer(), budget_ceiling_rub=1.0)

    assert excinfo.value.considered
    assert any(c.status == "over_ceiling" for c in excinfo.value.considered)


async def test_unaffordable_models_are_skipped_not_chosen(registry: ModelRegistry) -> None:
    intent = Intent(media_type="video", prompt="a cat", duration_s=10)
    pricer = StubPricer(balance=130.0, daily_remaining=130.0)

    decision = await route(intent, registry, pricer)

    assert decision.cost_rub <= 130.0
    assert any(c.status == "unaffordable" for c in decision.considered)


# --- planning --------------------------------------------------------------


def test_execution_order_is_topological() -> None:
    steps = (
        PlanStep(
            name="render", intent=Intent(media_type="image", prompt="x"), depends_on=("script",)
        ),
        PlanStep(name="script", intent=Intent(media_type="image", prompt="x")),
    )
    assert execution_order(steps) == ("script", "render")


def test_dependency_cycles_are_rejected() -> None:
    steps = (
        PlanStep(name="a", intent=Intent(media_type="image", prompt="x"), depends_on=("b",)),
        PlanStep(name="b", intent=Intent(media_type="image", prompt="x"), depends_on=("a",)),
    )
    with pytest.raises(PlanCycleError):
        execution_order(steps)


def test_dangling_dependency_is_rejected() -> None:
    steps = (
        PlanStep(name="a", intent=Intent(media_type="image", prompt="x"), depends_on=("ghost",)),
    )
    with pytest.raises(PlanCycleError, match="ghost"):
        execution_order(steps)


async def test_plan_totals_every_step_and_says_go(registry: ModelRegistry) -> None:
    steps = (
        PlanStep(name="poster", intent=Intent(media_type="image", prompt="a poster")),
        PlanStep(
            name="clip",
            intent=Intent(
                media_type="video", prompt="a cat", source_image_url="https://x/a.png", duration_s=5
            ),
            depends_on=("poster",),
        ),
    )

    plan = await estimate_plan(
        steps, registry, StubPricer(), AccountState(balance_rub=600, daily_limit_rub=500)
    )

    assert plan.verdict == "go"
    assert plan.total_rub == pytest.approx(1.2 + 36.0)
    assert [s.name for s in plan.steps] == ["poster", "clip"]
    assert plan.projected_balance_rub == pytest.approx(600 - 37.2)


async def test_plan_refuses_when_it_would_exceed_the_daily_limit(
    registry: ModelRegistry,
) -> None:
    steps = tuple(
        PlanStep(
            name=f"clip{i}",
            intent=Intent(
                media_type="video",
                prompt="a cat",
                source_image_url="https://x/a.png",
                duration_s=15,
            ),
        )
        for i in range(5)
    )

    plan = await estimate_plan(
        steps,
        registry,
        StubPricer(daily_remaining=500.0),
        AccountState(balance_rub=600, daily_limit_rub=500, spent_today_rub=0),
    )

    assert plan.verdict == "no_go"
    assert plan.total_rub == pytest.approx(5 * 120.0)
    assert any("today's limit" in b for b in plan.blockers)


async def test_plan_costs_every_step_even_when_one_is_unroutable(
    registry: ModelRegistry,
) -> None:
    steps = (
        PlanStep(name="ok", intent=Intent(media_type="image", prompt="a poster")),
        PlanStep(
            name="impossible",
            intent=Intent(media_type="video", prompt="a cat", duration_s=99),
        ),
    )

    plan = await estimate_plan(steps, registry, StubPricer(), AccountState(balance_rub=600))

    assert plan.verdict == "no_go"
    blocked = {s.name: s for s in plan.steps}
    assert blocked["ok"].cost_rub == pytest.approx(1.2)
    assert blocked["impossible"].is_blocked


# --- reconciliation --------------------------------------------------------

LIVE_PRICES = {
    "prices": {
        "image": {"z-image": 1.2, "nano-banana-2-lite": 9, "seedream-5-lite-edit": 18},
        "video": {"grok-ttv": 36, "grok-ttv-10": 196, "grok-ttv-30": 436, "veo3_fast": 120},
        "other": {"grok-itv": 36, "grok-itv-30": 436, "resize-2k": 30},
        "text": {"gemini-omni-audio": 39, "gemini-omni-character": 49},
    }
}


def test_flatten_prices_keeps_the_category(registry: ModelRegistry) -> None:
    flat = flatten_prices(LIVE_PRICES)
    assert flat["grok-ttv-30"] == ("video", 436.0)
    assert flat["grok-itv"] == ("other", 36.0)


def test_drift_reports_both_directions(registry: ModelRegistry) -> None:
    drift = reconcile(registry, flatten_prices(LIVE_PRICES))

    assert "grok-ttv-30" in drift.priced_without_model
    assert "seedream-5-lite-edit" in drift.priced_without_model
    assert "seedream-5-pro" in drift.generatable_without_price
    assert not drift.is_clean
    assert "no price" in drift.summary()


def test_price_list_categories_that_contradict_the_catalog_are_flagged(
    registry: ModelRegistry,
) -> None:
    drift = reconcile(registry, flatten_prices(LIVE_PRICES))
    flagged = {model_id: (ours, theirs) for model_id, ours, theirs in drift.category_mismatch}

    assert flagged["gemini-omni-audio"] == ("voice", "text")
    assert flagged["gemini-omni-character"] == ("image", "text")


def test_catalog_alone_understates_the_tier_ceiling(registry: ModelRegistry) -> None:
    """The measured consequence: a ceiling built from the catalog is too low.

    grok-itv declares tiers up to 316; the price list sells grok-itv-30 at 436.
    """
    unlisted, (parent, gap) = unlisted_tiers(registry, flatten_prices(LIVE_PRICES))

    assert ("grok-itv", "grok-itv-30", 436.0) in unlisted
    assert ("grok-ttv", "grok-ttv-30", 436.0) in unlisted
    assert parent in {"grok-itv", "grok-ttv"}
    assert gap == pytest.approx(436.0 - 316.0)
    assert gap / 316.0 > 0.37  # the 38% understatement
