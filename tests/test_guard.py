"""Guard plane logic, exercised against the in-process ports."""

from __future__ import annotations

import pytest

from mainager.guard.audit import GENESIS_HASH, AuditLog, compute_hash
from mainager.guard.memory import (
    HashingEmbedder,
    MemoryAuditSink,
    MemoryKeyValueStore,
    MemoryVectorIndex,
    cosine_similarity,
)
from mainager.guard.plane import (
    AutonomyLevel,
    GuardConfig,
    GuardPlane,
    GuardRequest,
    intent_fingerprint,
)


@pytest.fixture
def sink() -> MemoryAuditSink:
    return MemoryAuditSink()


@pytest.fixture
def audit(sink: MemoryAuditSink) -> AuditLog:
    return AuditLog(sink)


def _plane(audit: AuditLog, config: GuardConfig | None = None, **kwargs: object) -> GuardPlane:
    return GuardPlane(MemoryKeyValueStore(), audit, config, **kwargs)  # type: ignore[arg-type]


# --- audit chain -----------------------------------------------------------


async def test_first_record_chains_to_the_genesis_hash(audit: AuditLog) -> None:
    record = await audit.record("generate", "allow")

    assert record.sequence == 0
    assert record.previous_hash == GENESIS_HASH
    assert record.record_hash == compute_hash(record.payload_for_hashing())


async def test_each_record_chains_to_its_predecessor(audit: AuditLog) -> None:
    first = await audit.record("generate", "allow")
    second = await audit.record("generate", "deny")

    assert second.sequence == 1
    assert second.previous_hash == first.record_hash

    intact, problem = await audit.verify()
    assert intact
    assert problem is None


async def test_editing_a_record_is_detected(audit: AuditLog, sink: MemoryAuditSink) -> None:
    await audit.record("generate", "allow", {"cost_rub": 36})
    await audit.record("generate", "allow", {"cost_rub": 316})
    await audit.record("generate", "allow", {"cost_rub": 40})

    sink.tamper(1, detail={"cost_rub": 1})

    intact, problem = await audit.verify()
    assert intact is False
    assert problem is not None
    assert "modified" in problem


async def test_removing_a_record_is_detected(audit: AuditLog, sink: MemoryAuditSink) -> None:
    await audit.record("generate", "allow")
    await audit.record("campaign_write", "shadow")
    await audit.record("generate", "allow")

    del sink._records[1]  # simulate an entry being removed from storage

    intact, _ = await audit.verify()
    assert intact is False


async def test_an_empty_log_verifies(audit: AuditLog) -> None:
    intact, problem = await audit.verify()
    assert intact
    assert problem is None


# --- kill switch -----------------------------------------------------------


async def test_kill_switch_denies_everything(audit: AuditLog) -> None:
    plane = _plane(audit)
    await plane.trip_kill_switch("manual stop during incident")

    decision = await plane.evaluate(GuardRequest(action="read"))

    assert decision.outcome == "deny"
    assert decision.check == "kill_switch"
    assert "incident" in decision.reason


async def test_kill_switch_can_be_reset(audit: AuditLog) -> None:
    plane = _plane(audit)
    await plane.trip_kill_switch("stop")
    await plane.reset_kill_switch()

    assert (await plane.evaluate(GuardRequest(action="read"))).may_proceed


# --- autonomy --------------------------------------------------------------


@pytest.mark.parametrize(
    ("level", "action", "allowed"),
    [
        (AutonomyLevel.ADVISOR, "read", False),
        (AutonomyLevel.ASSISTANT, "read", True),
        (AutonomyLevel.ASSISTANT, "generate", False),
        (AutonomyLevel.OPERATOR, "generate", True),
        (AutonomyLevel.OPERATOR, "campaign_write", False),
        (AutonomyLevel.AUTOPILOT, "campaign_write", True),
    ],
)
async def test_autonomy_ladder(
    audit: AuditLog, level: AutonomyLevel, action: str, allowed: bool
) -> None:
    plane = _plane(audit, GuardConfig(autonomy=level))

    decision = await plane.evaluate(GuardRequest(action=action))

    assert (decision.outcome != "deny") is allowed


async def test_an_unknown_action_needs_the_highest_level(audit: AuditLog) -> None:
    plane = _plane(audit, GuardConfig(autonomy=AutonomyLevel.OPERATOR))

    decision = await plane.evaluate(GuardRequest(action="delete_everything"))

    assert decision.outcome == "deny"
    assert decision.check == "autonomy"


async def test_write_actions_produce_a_diff_instead_of_executing(audit: AuditLog) -> None:
    plane = _plane(audit, GuardConfig(autonomy=AutonomyLevel.AUTOPILOT))

    decision = await plane.evaluate(GuardRequest(action="campaign_write"))

    assert decision.outcome == "shadow"
    assert decision.may_proceed is False


# --- ceilings --------------------------------------------------------------


async def test_a_call_above_the_ceiling_is_denied(audit: AuditLog) -> None:
    plane = _plane(audit, GuardConfig(call_ceiling_rub=100))

    decision = await plane.evaluate(GuardRequest(action="generate", estimated_cost_rub=316))

    assert decision.outcome == "deny"
    assert decision.check == "call_ceiling"
    assert decision.detail["cost_rub"] == 316


async def test_a_call_under_the_ceiling_passes(audit: AuditLog) -> None:
    plane = _plane(audit, GuardConfig(call_ceiling_rub=100))

    decision = await plane.evaluate(GuardRequest(action="generate", estimated_cost_rub=36))

    assert decision.may_proceed


# --- loop detection --------------------------------------------------------


async def test_repeated_identical_requests_trip_the_loop_detector(audit: AuditLog) -> None:
    plane = _plane(audit, GuardConfig(loop_threshold=3))
    request = GuardRequest(action="generate", prompt="a cat", params={"duration": 5})

    outcomes = [(await plane.evaluate(request)).outcome for _ in range(5)]

    assert outcomes[:3] == ["allow", "allow", "allow"]
    assert outcomes[3:] == ["deny", "deny"]


async def test_different_requests_do_not_share_a_window(audit: AuditLog) -> None:
    plane = _plane(audit, GuardConfig(loop_threshold=1))

    first = await plane.evaluate(GuardRequest(action="generate", prompt="a cat"))
    second = await plane.evaluate(GuardRequest(action="generate", prompt="a dog"))

    assert first.may_proceed
    assert second.may_proceed


async def test_tenants_do_not_share_a_loop_window(audit: AuditLog) -> None:
    plane = _plane(audit, GuardConfig(loop_threshold=1))
    prompt = "a cat"

    await plane.evaluate(GuardRequest(action="generate", prompt=prompt, tenant="acme"))
    other = await plane.evaluate(GuardRequest(action="generate", prompt=prompt, tenant="globex"))

    assert other.may_proceed


def test_fingerprint_ignores_parameter_order() -> None:
    left = GuardRequest(action="generate", prompt="A  cat", params={"a": 1, "b": 2})
    right = GuardRequest(action="generate", prompt="a cat", params={"b": 2, "a": 1})

    assert intent_fingerprint(left) == intent_fingerprint(right)


# --- semantic dedup --------------------------------------------------------


async def test_a_near_identical_prompt_is_served_from_the_index(audit: AuditLog) -> None:
    plane = _plane(
        audit,
        GuardConfig(dedup_threshold=0.95, loop_threshold=0),
        embedder=HashingEmbedder(),
        index=MemoryVectorIndex(),
    )
    await plane.remember_generation(
        "gen_1", "a cat walking on a beach", {"display_url": "https://cdn/x.mp4"}
    )

    decision = await plane.evaluate(
        GuardRequest(action="generate", prompt="A cat walking on a beach")
    )

    assert decision.outcome == "reuse"
    assert decision.detail["existing"]["display_url"] == "https://cdn/x.mp4"
    assert decision.may_proceed is False


async def test_an_unrelated_prompt_is_not_deduplicated(audit: AuditLog) -> None:
    plane = _plane(
        audit,
        GuardConfig(dedup_threshold=0.95, loop_threshold=0),
        embedder=HashingEmbedder(),
        index=MemoryVectorIndex(),
    )
    await plane.remember_generation("gen_1", "a cat walking on a beach", {})

    decision = await plane.evaluate(
        GuardRequest(action="generate", prompt="quarterly revenue chart for the board")
    )

    assert decision.may_proceed


async def test_dedup_is_skipped_when_no_index_is_configured(audit: AuditLog) -> None:
    plane = _plane(audit, GuardConfig(loop_threshold=0))

    assert (await plane.evaluate(GuardRequest(action="generate", prompt="a cat"))).may_proceed


def test_cosine_similarity_bounds() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity([1, 0], [1, 0, 0])


async def test_embedder_is_deterministic() -> None:
    embedder = HashingEmbedder()
    assert await embedder.embed("a cat") == await embedder.embed("a  CAT ")


# --- every decision is audited ---------------------------------------------


async def test_every_decision_lands_in_the_chain(audit: AuditLog, sink: MemoryAuditSink) -> None:
    plane = _plane(audit, GuardConfig(autonomy=AutonomyLevel.ASSISTANT))

    await plane.evaluate(GuardRequest(action="read"))
    await plane.evaluate(GuardRequest(action="generate"))

    records = await sink.all()
    assert [r["outcome"] for r in records] == ["allow", "deny"]
    assert all(r["detail"]["fingerprint"] for r in records)

    intact, _ = await audit.verify()
    assert intact
