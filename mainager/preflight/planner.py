"""Cost a whole scenario before any of it runs.

Billing is per generation, so the number an operator actually needs — what this
scenario will cost end to end, and whether the account can absorb it — does not
exist anywhere in the provider's API. It is assembled here from free dry-runs.

A plan is a DAG of intents. The dependency edges decide execution order; the
cost is the sum over every step, because every step runs. The verdict compares
that sum against both ceilings that can stop a run: the account balance and the
daily spend limit.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mainager.preflight.compiler import Intent
from mainager.preflight.pricing import Pricer
from mainager.preflight.registry import ModelRegistry
from mainager.preflight.router import NoViableModelError, route

Verdict = Literal["go", "no_go"]


class PlanStep(BaseModel):
    """One intent in a scenario, optionally waiting on earlier ones."""

    model_config = ConfigDict(frozen=True)

    name: str
    intent: Intent
    depends_on: tuple[str, ...] = ()


class StepCost(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    model_id: str | None
    cost_rub: float | None
    blocked_reason: str | None = None

    @property
    def is_blocked(self) -> bool:
        return self.blocked_reason is not None


class AccountState(BaseModel):
    """Balance and limits, as reported by the provider."""

    model_config = ConfigDict(frozen=True)

    balance_rub: float
    daily_limit_rub: float | None = None
    spent_today_rub: float = 0.0

    @property
    def daily_remaining_rub(self) -> float | None:
        if self.daily_limit_rub is None:
            return None
        return max(0.0, self.daily_limit_rub - self.spent_today_rub)


class PlanEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    steps: tuple[StepCost, ...]
    total_rub: float
    account: AccountState
    verdict: Verdict
    blockers: tuple[str, ...] = Field(default_factory=tuple)

    @property
    def projected_balance_rub(self) -> float:
        return self.account.balance_rub - self.total_rub


class PlanCycleError(ValueError):
    """The plan's dependencies are not a DAG."""


def execution_order(steps: tuple[PlanStep, ...]) -> tuple[str, ...]:
    """Topologically sort the steps, rejecting cycles and dangling references."""
    by_name = {step.name: step for step in steps}
    if len(by_name) != len(steps):
        raise PlanCycleError("step names must be unique")

    for step in steps:
        for dependency in step.depends_on:
            if dependency not in by_name:
                raise PlanCycleError(f"{step.name!r} depends on unknown step {dependency!r}")

    ordered: list[str] = []
    permanent: set[str] = set()
    temporary: set[str] = set()

    def visit(name: str) -> None:
        if name in permanent:
            return
        if name in temporary:
            raise PlanCycleError(f"dependency cycle involving {name!r}")
        temporary.add(name)
        for dependency in by_name[name].depends_on:
            visit(dependency)
        temporary.discard(name)
        permanent.add(name)
        ordered.append(name)

    for step in steps:
        visit(step.name)
    return tuple(ordered)


async def estimate_plan(
    steps: tuple[PlanStep, ...],
    registry: ModelRegistry,
    pricer: Pricer,
    account: AccountState,
    *,
    step_ceiling_rub: float | None = None,
    plan_ceiling_rub: float | None = None,
) -> PlanEstimate:
    """Price every step and decide whether the plan may run.

    Nothing is spent. A step that cannot be routed does not abort the walk: the
    whole plan is costed so the operator sees the entire picture at once, rather
    than discovering the second problem after fixing the first.
    """
    order = execution_order(steps)
    by_name = {step.name: step for step in steps}

    costs: list[StepCost] = []
    for name in order:
        step = by_name[name]
        try:
            decision = await route(
                step.intent, registry, pricer, budget_ceiling_rub=step_ceiling_rub
            )
        except NoViableModelError as exc:
            reasons = "; ".join(f"{c.model_id}: {c.reason}" for c in exc.considered if c.reason)
            costs.append(
                StepCost(
                    name=name,
                    model_id=None,
                    cost_rub=None,
                    blocked_reason=reasons or "no viable model",
                )
            )
            continue
        costs.append(
            StepCost(name=name, model_id=decision.spec.model_id, cost_rub=decision.cost_rub)
        )

    total = sum(c.cost_rub or 0.0 for c in costs)
    blockers: list[str] = []

    for cost in costs:
        if cost.is_blocked:
            blockers.append(f"step {cost.name!r} cannot be routed: {cost.blocked_reason}")

    if total > account.balance_rub:
        blockers.append(f"plan costs {total:g} RUB, balance is {account.balance_rub:g} RUB")

    remaining = account.daily_remaining_rub
    if remaining is not None and total > remaining:
        blockers.append(
            f"plan costs {total:g} RUB, only {remaining:g} RUB left under today's limit"
        )

    if plan_ceiling_rub is not None and total > plan_ceiling_rub:
        blockers.append(f"plan costs {total:g} RUB, ceiling is {plan_ceiling_rub:g} RUB")

    return PlanEstimate(
        steps=tuple(costs),
        total_rub=total,
        account=account,
        verdict="no_go" if blockers else "go",
        blockers=tuple(blockers),
    )
