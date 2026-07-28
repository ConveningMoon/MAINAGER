"""Guard logic for a proxied MCP tool surface.

The provider's connector exposes nine tools. Two of them reach the billing path,
and neither declares the parameter that stops a malformed request from being
charged: `strict` is absent from both `inputSchema` blocks, absent from the tool
descriptions, and absent from the server's own `instructions`. It works when
passed — it just cannot be discovered by an agent reading the schema.

Everything here is a pure transformation over tool definitions and call
arguments, so it is testable without a network or an MCP runtime. The server
module is only wiring.

Three transformations, in order of how much they matter:

1. `strict` is added to the schema of the two generation tools and defaulted to
   true, so the protection is both visible and on by default.
2. The verdict is re-derived from `rejected` rather than taken from `valid`,
   which is unreliable in both directions: true for a body carrying a parameter
   the provider will drop, false for a well-formed body that merely exceeds the
   spend limit.
3. A pinned tier is rewritten to its base identifier when that is cheaper — but
   only when the call states a duration, because without one the base identifier
   would silently produce a shorter clip. Never silently: every rewrite is
   reported back in the tool result.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from mainager.preflight.registry import ModelRegistry

#: Tools that can reach the billing path.
GENERATION_TOOLS: frozenset[str] = frozenset({"estimate_generation", "generate_content"})

#: The tool that actually debits.
CHARGING_TOOL = "generate_content"

_STRICT_SCHEMA: dict[str, Any] = {
    "type": "boolean",
    "default": True,
    "description": (
        "Reject unknown or incompatible parameters before any debit. "
        "Without it the provider drops them, reports them in ignored_params, "
        "and still charges. Leave this on unless you have a specific reason."
    ),
}


class GuardedCall(BaseModel):
    """Arguments after guarding, plus what was changed and why."""

    model_config = ConfigDict(frozen=True)

    arguments: dict[str, Any]
    notes: tuple[str, ...] = ()

    @property
    def was_modified(self) -> bool:
        return bool(self.notes)


class CallVerdict(BaseModel):
    """What a dry-run response actually means, independent of its `valid` flag."""

    model_config = ConfigDict(frozen=True)

    cost_rub: float
    rejected: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    provider_valid: bool | None = None

    @property
    def body_is_clean(self) -> bool:
        return not self.rejected

    @property
    def affordable(self) -> bool:
        return not any(
            "insufficient_balance" in w or "daily_spend_limit" in w for w in self.warnings
        )

    @property
    def safe_to_charge(self) -> bool:
        return self.body_is_clean and self.affordable

    def refusal_reason(self) -> str | None:
        if not self.body_is_clean:
            dropped = ", ".join(self.rejected)
            return (
                f"the provider would drop {dropped} and still charge "
                f"{self.cost_rub:g} RUB. Fix the parameters or pass strict=false "
                f"deliberately."
            )
        if not self.affordable:
            return "; ".join(self.warnings) or "balance or daily limit would be exceeded"
        return None


def augment_tool_schema(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Expose `strict` on the tools that can be charged for.

    Returns the schema unchanged for every other tool.
    """
    if name not in GENERATION_TOOLS:
        return schema

    properties = dict(schema.get("properties") or {})
    if "strict" in properties:
        return schema
    properties["strict"] = dict(_STRICT_SCHEMA)

    augmented = dict(schema)
    augmented["properties"] = properties
    return augmented


def guard_call(
    name: str,
    arguments: dict[str, Any],
    registry: ModelRegistry | None = None,
    *,
    rewrite_pinned_tiers: bool = True,
) -> GuardedCall:
    """Apply the guard transformations to one tool call."""
    if name not in GENERATION_TOOLS:
        return GuardedCall(arguments=dict(arguments))

    guarded = dict(arguments)
    notes: list[str] = []

    if "strict" not in guarded:
        guarded["strict"] = True
        notes.append("strict=true was added; the upstream schema does not expose it")
    elif guarded["strict"] is False:
        notes.append(
            "strict=false was passed through as requested; incompatible parameters "
            "will be dropped and still billed"
        )

    if rewrite_pinned_tiers and registry is not None:
        guarded, rewrite_note = _rewrite_pinned_tier(guarded, registry)
        if rewrite_note:
            notes.append(rewrite_note)

    return GuardedCall(arguments=guarded, notes=tuple(notes))


def _rewrite_pinned_tier(
    arguments: dict[str, Any], registry: ModelRegistry
) -> tuple[dict[str, Any], str | None]:
    """Swap a pinned tier for its base id when the call states a duration.

    Without a duration the base id would pick its own default tier, which can be
    shorter than the pinned one. That would change the output behind the
    caller's back, so it is left alone and nothing is claimed about it.
    """
    model_id = arguments.get("model")
    if not isinstance(model_id, str):
        return arguments, None

    spec = registry.find(model_id)
    if spec is None or spec.is_tier_of is None:
        return arguments, None

    parent = registry.find(spec.is_tier_of)
    if parent is None:
        return arguments, None

    params = arguments.get("params")
    duration = params.get("duration") if isinstance(params, dict) else None
    if duration is None:
        return arguments, (
            f"{model_id} pins a price tier ({spec.price:g} RUB) and ignores duration. "
            f"Pass params.duration and use {parent.model_id} to pay for the length "
            f"you actually need."
        )

    rewritten = dict(arguments)
    rewritten["model"] = parent.model_id
    return rewritten, (
        f"model rewritten {model_id} -> {parent.model_id}: on the base id the "
        f"duration selects the cheapest sufficient tier, on a pinned tier it is "
        f"accepted and ignored"
    )


def read_verdict(payload: dict[str, Any]) -> CallVerdict:
    """Interpret a dry-run response without trusting its `valid` field."""

    def _strs(key: str) -> tuple[str, ...]:
        value = payload.get(key)
        return tuple(str(v) for v in value) if isinstance(value, list) else ()

    cost = payload.get("estimated_cost_rub")
    valid = payload.get("valid")
    return CallVerdict(
        cost_rub=float(cost) if isinstance(cost, int | float) else 0.0,
        rejected=_strs("rejected"),
        warnings=_strs("warnings"),
        provider_valid=bool(valid) if isinstance(valid, bool) else None,
    )


def ceiling_refusal(verdict: CallVerdict, ceiling_rub: float | None) -> str | None:
    """Refusal text when a charge would exceed the configured ceiling."""
    if ceiling_rub is None or verdict.cost_rub <= ceiling_rub:
        return None
    return (
        f"this call would cost {verdict.cost_rub:g} RUB, above the configured "
        f"ceiling of {ceiling_rub:g} RUB. Raise MAINAGER_PLAN_CEILING_RUB or pick "
        f"a cheaper model."
    )
