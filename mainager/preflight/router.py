"""Choose the model that serves an intent for the least money.

The catalog's `price` field is not usable for this. Measured against the live
pricer, a 15-second clip costs 316 RUB on `grok-itv` but 120 RUB on `veo3_fast`,
while the catalog lists those models at 36 and 120 — so ranking by the catalog
picks the model that turns out to be 2.6x more expensive. Every candidate is
therefore priced by dry-run, which is free.

One rule is baked in rather than discovered per request: prefer a base model id
over one of its pinned tiers. On the base id `duration` selects the cheapest
sufficient tier automatically; on a pinned tier `duration` is accepted and
ignored. Measured spread for a 5-second clip: 36 RUB via `grok-itv` against
316 RUB via `grok-itv-20`.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import BaseModel, ConfigDict

from mainager.preflight.compiler import (
    IncompatibleIntentError,
    Intent,
    compile_request,
)
from mainager.preflight.pricing import Estimate, Pricer
from mainager.preflight.registry import ModelRegistry, ModelSpec

Policy = Literal["cheapest"]

CandidateStatus = Literal[
    "chosen",
    "priced",
    "incompatible",
    "dirty_body",
    "over_ceiling",
    "unaffordable",
]

_MAX_CONCURRENT_ESTIMATES = 4


class Candidate(BaseModel):
    """One model that was considered, and what happened to it."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    status: CandidateStatus
    cost_rub: float | None = None
    reason: str | None = None


class NoViableModelError(RuntimeError):
    """Nothing in the catalog can serve this intent within the constraints."""

    def __init__(self, considered: tuple[Candidate, ...]) -> None:
        super().__init__(f"no viable model among {len(considered)} candidates")
        self.considered = considered


class RoutingDecision(BaseModel):
    """The chosen model plus every candidate that was rejected, and why.

    The full list is kept deliberately: an autonomous system that spends money
    should be able to answer "why this model" after the fact.
    """

    model_config = ConfigDict(frozen=True)

    spec: ModelSpec
    body: dict[str, object]
    estimate: Estimate
    considered: tuple[Candidate, ...]

    @property
    def cost_rub(self) -> float:
        return self.estimate.cost_rub


def candidate_specs(
    intent: Intent, registry: ModelRegistry, *, allow_pinned_tiers: bool = False
) -> tuple[ModelSpec, ...]:
    """Models of the right type whose contract this intent can satisfy."""
    specs = registry.by_type(intent.media_type)
    if not allow_pinned_tiers:
        specs = tuple(s for s in specs if s.is_tier_of is None)
    viable = []
    for spec in specs:
        try:
            compile_request(intent, spec)
        except IncompatibleIntentError:
            continue
        viable.append(spec)
    return tuple(viable)


async def route(
    intent: Intent,
    registry: ModelRegistry,
    pricer: Pricer,
    *,
    budget_ceiling_rub: float | None = None,
    policy: Policy = "cheapest",
    allow_pinned_tiers: bool = False,
) -> RoutingDecision:
    """Price every compatible model and return the cheapest usable one.

    Raises ``NoViableModelError`` with the full audit trail when nothing
    qualifies. Costs nothing: estimates do not debit.
    """
    considered: list[Candidate] = []
    bodies: dict[str, dict[str, object]] = {}
    specs_by_id: dict[str, ModelSpec] = {}

    for spec in registry.by_type(intent.media_type):
        if spec.is_tier_of is not None and not allow_pinned_tiers:
            continue
        try:
            bodies[spec.model_id] = compile_request(intent, spec)
            specs_by_id[spec.model_id] = spec
        except IncompatibleIntentError as exc:
            considered.append(
                Candidate(model_id=spec.model_id, status="incompatible", reason=exc.reason)
            )

    if not bodies:
        raise NoViableModelError(tuple(considered))

    estimates = await _price_all(bodies, pricer)

    priced: list[tuple[float, str, Estimate]] = []
    for model_id, estimate in estimates.items():
        if isinstance(estimate, BaseException):
            considered.append(
                Candidate(model_id=model_id, status="incompatible", reason=str(estimate))
            )
            continue
        if not estimate.body_is_clean:
            considered.append(
                Candidate(
                    model_id=model_id,
                    status="dirty_body",
                    cost_rub=estimate.cost_rub,
                    reason=f"provider would drop {', '.join(estimate.rejected)} and still charge",
                )
            )
            continue
        if budget_ceiling_rub is not None and estimate.cost_rub > budget_ceiling_rub:
            considered.append(
                Candidate(
                    model_id=model_id,
                    status="over_ceiling",
                    cost_rub=estimate.cost_rub,
                    reason=f"above ceiling of {budget_ceiling_rub} RUB",
                )
            )
            continue
        if not estimate.affordable:
            considered.append(
                Candidate(
                    model_id=model_id,
                    status="unaffordable",
                    cost_rub=estimate.cost_rub,
                    reason="; ".join(estimate.warnings) or "balance or daily limit",
                )
            )
            continue
        priced.append((estimate.cost_rub, model_id, estimate))

    if not priced:
        raise NoViableModelError(tuple(considered))

    priced.sort(key=lambda row: (row[0], row[1]))
    best_cost, best_id, best_estimate = priced[0]
    for cost, model_id, _ in priced[1:]:
        considered.append(Candidate(model_id=model_id, status="priced", cost_rub=cost))
    considered.append(Candidate(model_id=best_id, status="chosen", cost_rub=best_cost))

    return RoutingDecision(
        spec=specs_by_id[best_id],
        body=bodies[best_id],
        estimate=best_estimate,
        considered=tuple(sorted(considered, key=lambda c: c.model_id)),
    )


async def _price_all(
    bodies: dict[str, dict[str, object]], pricer: Pricer
) -> dict[str, Estimate | BaseException]:
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_ESTIMATES)

    async def one(body: dict[str, object]) -> Estimate:
        async with semaphore:
            return await pricer.estimate(dict(body))

    model_ids = list(bodies)
    results = await asyncio.gather(
        *(one(bodies[model_id]) for model_id in model_ids), return_exceptions=True
    )
    return dict(zip(model_ids, results, strict=True))
