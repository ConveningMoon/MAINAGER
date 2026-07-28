"""Guard transformations over the proxied MCP tool surface.

The upstream schemas and the dry-run payloads replayed here were captured from
the live server on 2026-07-28.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mainager.guard.mcp_guard import (
    augment_tool_schema,
    ceiling_refusal,
    guard_call,
    read_verdict,
)
from mainager.preflight.registry import ModelRegistry
from mainager.providers.vibemarketolog.capabilities import read_snapshot

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data"

# Verbatim from tools/list on https://lk.vibemarketolog.ru/api/mcp
UPSTREAM_GENERATE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["video", "image", "voice", "music", "text"]},
        "model": {"type": "string", "description": "Model id from list_capabilities"},
        "prompt": {"type": "string"},
        "params": {"type": "object", "description": "..."},
        "webhook_url": {"type": "string"},
        "idempotency_key": {"type": "string"},
    },
    "required": ["type", "model"],
}

# Verbatim from POST /generate/estimate with image_input on a video model.
TRAP_ESTIMATE = {
    "valid": True,
    "dry_run": True,
    "model": "grok-ttv-10",
    "type": "video",
    "estimated_cost_rub": 196,
    "rejected": ["image_input"],
    "warnings": ["image_input is for type=image only. For image-to-video use image_urls[]..."],
}

# Verbatim shape from a well-formed body the account cannot afford.
UNAFFORDABLE_ESTIMATE = {
    "valid": False,
    "model": "seedance-2",
    "estimated_cost_rub": 1188,
    "rejected": [],
    "warnings": [
        "insufficient_balance: need 1188₽, have 600₽.",
        "daily_spend_limit would be exceeded.",
    ],
}


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    return ModelRegistry.from_capabilities(read_snapshot(SNAPSHOT_DIR))


# --- schema ----------------------------------------------------------------


def test_strict_is_added_to_the_charging_tools() -> None:
    augmented = augment_tool_schema("generate_content", UPSTREAM_GENERATE_SCHEMA)

    strict = augmented["properties"]["strict"]
    assert strict["type"] == "boolean"
    assert strict["default"] is True
    assert "still charges" in strict["description"]


def test_the_upstream_schema_is_not_mutated() -> None:
    augment_tool_schema("generate_content", UPSTREAM_GENERATE_SCHEMA)
    assert "strict" not in UPSTREAM_GENERATE_SCHEMA["properties"]


def test_non_charging_tools_are_left_alone() -> None:
    schema = {"type": "object", "properties": {"query": {"type": "string"}}}
    assert augment_tool_schema("search", schema) == schema
    assert augment_tool_schema("get_balance", schema) == schema


# --- arguments -------------------------------------------------------------


def test_strict_is_injected_when_absent(registry: ModelRegistry) -> None:
    guarded = guard_call(
        "generate_content", {"type": "video", "model": "grok-ttv", "prompt": "a cat"}, registry
    )

    assert guarded.arguments["strict"] is True
    assert any("does not expose it" in note for note in guarded.notes)


def test_an_explicit_strict_false_is_honoured_but_flagged(registry: ModelRegistry) -> None:
    guarded = guard_call(
        "generate_content",
        {"type": "video", "model": "grok-ttv", "prompt": "a cat", "strict": False},
        registry,
    )

    assert guarded.arguments["strict"] is False
    assert any("still billed" in note for note in guarded.notes)


def test_tools_that_cannot_charge_pass_through_untouched(registry: ModelRegistry) -> None:
    guarded = guard_call("list_capabilities", {}, registry)
    assert guarded.arguments == {}
    assert not guarded.was_modified


def test_pinned_tier_with_a_duration_is_rewritten_to_the_base_id(
    registry: ModelRegistry,
) -> None:
    guarded = guard_call(
        "generate_content",
        {
            "type": "video",
            "model": "grok-itv-20",
            "prompt": "a cat",
            "params": {"duration": 5, "image_urls": ["https://x/a.png"]},
        },
        registry,
    )

    assert guarded.arguments["model"] == "grok-itv"
    assert any("rewritten grok-itv-20 -> grok-itv" in note for note in guarded.notes)


def test_pinned_tier_without_a_duration_is_left_alone_and_explained(
    registry: ModelRegistry,
) -> None:
    """Rewriting here would shorten the clip behind the caller's back."""
    arguments = {"type": "video", "model": "grok-itv-20", "prompt": "a cat"}

    guarded = guard_call("generate_content", arguments, registry)

    assert guarded.arguments["model"] == "grok-itv-20"
    note = " ".join(guarded.notes)
    assert "ignores duration" in note
    assert "grok-itv" in note


def test_a_base_id_is_never_rewritten(registry: ModelRegistry) -> None:
    guarded = guard_call(
        "generate_content",
        {"type": "video", "model": "grok-itv", "prompt": "a cat", "params": {"duration": 5}},
        registry,
    )
    assert guarded.arguments["model"] == "grok-itv"
    assert not any("rewritten" in note for note in guarded.notes)


def test_rewriting_can_be_switched_off(registry: ModelRegistry) -> None:
    guarded = guard_call(
        "generate_content",
        {"type": "video", "model": "grok-itv-20", "prompt": "a cat", "params": {"duration": 5}},
        registry,
        rewrite_pinned_tiers=False,
    )
    assert guarded.arguments["model"] == "grok-itv-20"


# --- verdicts --------------------------------------------------------------


def test_the_trap_is_refused_even_though_the_provider_said_valid() -> None:
    verdict = read_verdict(TRAP_ESTIMATE)

    assert verdict.provider_valid is True
    assert verdict.body_is_clean is False
    assert verdict.safe_to_charge is False
    reason = verdict.refusal_reason()
    assert reason is not None
    assert "image_input" in reason
    assert "196 RUB" in reason


def test_an_unaffordable_but_well_formed_body_is_reported_as_such() -> None:
    verdict = read_verdict(UNAFFORDABLE_ESTIMATE)

    assert verdict.provider_valid is False
    assert verdict.body_is_clean is True  # the body was fine
    assert verdict.affordable is False
    assert "insufficient_balance" in (verdict.refusal_reason() or "")


def test_a_clean_affordable_estimate_passes() -> None:
    verdict = read_verdict(
        {"valid": True, "estimated_cost_rub": 36, "rejected": [], "warnings": []}
    )

    assert verdict.safe_to_charge
    assert verdict.refusal_reason() is None


def test_ceiling_blocks_an_expensive_call() -> None:
    verdict = read_verdict({"estimated_cost_rub": 316, "rejected": [], "warnings": []})

    assert ceiling_refusal(verdict, 100.0) is not None
    assert "316 RUB" in (ceiling_refusal(verdict, 100.0) or "")
    assert ceiling_refusal(verdict, 400.0) is None
    assert ceiling_refusal(verdict, None) is None
