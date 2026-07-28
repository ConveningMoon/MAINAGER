"""The guard plane: everything that can stop an action before it costs money.

Checks run cheapest first, and the ordering is the design. A kill switch lookup
is one round trip to a shared store; a semantic deduplication query costs an
embedding and a vector search. There is no reason to pay for the second when the
first already said no, and the inbound channel gives roughly four seconds before
the platform answers on the agent's behalf.

No model is consulted anywhere in this plane. Every decision here is
deterministic, which is what makes it auditable and what keeps it fast.
"""

from __future__ import annotations

import hashlib
import json
from enum import IntEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mainager.guard.audit import AuditLog
from mainager.guard.ports import Embedder, KeyValueStore, VectorIndex

KILL_SWITCH_KEY = "mainager:kill_switch"

Outcome = Literal["allow", "deny", "shadow", "reuse"]


class AutonomyLevel(IntEnum):
    """Mapped one-to-one onto the platform's own commercial ladder."""

    ADVISOR = 0
    ASSISTANT = 1
    OPERATOR = 2
    AUTOPILOT = 3


#: Minimum level each action class requires.
ACTION_LEVELS: dict[str, AutonomyLevel] = {
    "read": AutonomyLevel.ASSISTANT,
    "estimate": AutonomyLevel.ASSISTANT,
    "generate": AutonomyLevel.OPERATOR,
    "upload": AutonomyLevel.OPERATOR,
    "campaign_write": AutonomyLevel.AUTOPILOT,
    "autopilot_run": AutonomyLevel.AUTOPILOT,
}


class GuardRequest(BaseModel):
    """One action asking for permission."""

    model_config = ConfigDict(frozen=True)

    action: str
    tenant: str = "default"
    prompt: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    estimated_cost_rub: float | None = None


class GuardDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    outcome: Outcome
    check: str
    reason: str
    detail: dict[str, Any] = Field(default_factory=dict)

    @property
    def may_proceed(self) -> bool:
        return self.outcome == "allow"


class GuardConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    autonomy: AutonomyLevel = AutonomyLevel.OPERATOR
    call_ceiling_rub: float | None = None
    daily_ceiling_rub: float | None = None
    loop_window_s: int = 300
    loop_threshold: int = 3
    dedup_threshold: float = 0.97
    shadow_actions: frozenset[str] = frozenset({"campaign_write", "autopilot_run"})


def intent_fingerprint(request: GuardRequest) -> str:
    """Stable hash of what is being asked for, ignoring key order."""
    payload = {
        "action": request.action,
        "tenant": request.tenant,
        "prompt": " ".join((request.prompt or "").lower().split()),
        "params": request.params,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


class GuardPlane:
    """Runs the checks in cost order and records every decision."""

    def __init__(
        self,
        store: KeyValueStore,
        audit: AuditLog,
        config: GuardConfig | None = None,
        *,
        embedder: Embedder | None = None,
        index: VectorIndex | None = None,
    ) -> None:
        self._store = store
        self._audit = audit
        self._config = config or GuardConfig()
        self._embedder = embedder
        self._index = index

    async def trip_kill_switch(self, reason: str) -> None:
        await self._store.set(KILL_SWITCH_KEY, reason)
        await self._audit.record("kill_switch", "tripped", {"reason": reason})

    async def reset_kill_switch(self) -> None:
        await self._store.delete(KILL_SWITCH_KEY)
        await self._audit.record("kill_switch", "reset", {})

    async def evaluate(self, request: GuardRequest) -> GuardDecision:
        decision = await self._evaluate(request)
        await self._audit.record(
            request.action,
            decision.outcome,
            {
                "check": decision.check,
                "reason": decision.reason,
                "tenant": request.tenant,
                "fingerprint": intent_fingerprint(request),
                **decision.detail,
            },
        )
        return decision

    async def _evaluate(self, request: GuardRequest) -> GuardDecision:
        # 1. Kill switch. One lookup, and it overrides everything.
        tripped = await self._store.get(KILL_SWITCH_KEY)
        if tripped is not None:
            return GuardDecision(
                outcome="deny",
                check="kill_switch",
                reason=f"kill switch is engaged: {tripped}",
            )

        # 2. Autonomy level. Pure comparison, no I/O.
        required = ACTION_LEVELS.get(request.action, AutonomyLevel.AUTOPILOT)
        if self._config.autonomy < required:
            return GuardDecision(
                outcome="deny",
                check="autonomy",
                reason=(
                    f"{request.action} needs level {required.name}, "
                    f"running at {self._config.autonomy.name}"
                ),
                detail={"required": int(required), "current": int(self._config.autonomy)},
            )

        # 3. Shadow mode for actions that change someone else's account.
        if request.action in self._config.shadow_actions:
            return GuardDecision(
                outcome="shadow",
                check="shadow_mode",
                reason=f"{request.action} produces a diff for approval instead of executing",
            )

        # 4. Ceilings. Before the network, when a cost is known.
        ceiling_decision = self._check_ceilings(request)
        if ceiling_decision is not None:
            return ceiling_decision

        # 5. Loop detector. One increment against a shared window.
        loop_decision = await self._check_loop(request)
        if loop_decision is not None:
            return loop_decision

        # 6. Semantic deduplication. The most expensive check, so it runs last.
        dedup_decision = await self._check_dedup(request)
        if dedup_decision is not None:
            return dedup_decision

        return GuardDecision(outcome="allow", check="all", reason="all checks passed")

    def _check_ceilings(self, request: GuardRequest) -> GuardDecision | None:
        cost = request.estimated_cost_rub
        if cost is None:
            return None
        ceiling = self._config.call_ceiling_rub
        if ceiling is not None and cost > ceiling:
            return GuardDecision(
                outcome="deny",
                check="call_ceiling",
                reason=f"{cost:g} RUB is above the per-call ceiling of {ceiling:g} RUB",
                detail={"cost_rub": cost, "ceiling_rub": ceiling},
            )
        return None

    async def _check_loop(self, request: GuardRequest) -> GuardDecision | None:
        if self._config.loop_threshold <= 0:
            return None
        fingerprint = intent_fingerprint(request)
        key = f"mainager:loop:{request.tenant}:{fingerprint}"
        count = await self._store.incr_in_window(key, window_s=self._config.loop_window_s)
        if count > self._config.loop_threshold:
            return GuardDecision(
                outcome="deny",
                check="loop_detector",
                reason=(
                    f"same request {count} times in {self._config.loop_window_s}s; "
                    f"threshold is {self._config.loop_threshold}"
                ),
                detail={"count": count, "fingerprint": fingerprint},
            )
        return None

    async def _check_dedup(self, request: GuardRequest) -> GuardDecision | None:
        if self._embedder is None or self._index is None or not request.prompt:
            return None
        if request.action != "generate":
            return None

        vector = await self._embedder.embed(request.prompt)
        matches = await self._index.nearest(vector, limit=1)
        if not matches:
            return None

        similarity, key, payload = matches[0]
        if similarity < self._config.dedup_threshold:
            return None
        return GuardDecision(
            outcome="reuse",
            check="semantic_dedup",
            reason=f"a generation with similarity {similarity:.3f} already exists",
            detail={"similarity": similarity, "existing_key": key, "existing": payload},
        )

    async def remember_generation(self, key: str, prompt: str, payload: dict[str, Any]) -> None:
        """Index a completed generation so later duplicates can reuse it."""
        if self._embedder is None or self._index is None:
            return
        vector = await self._embedder.embed(prompt)
        await self._index.add(key, vector, payload)
