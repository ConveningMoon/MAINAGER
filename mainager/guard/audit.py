"""Hash-chained audit log.

Each record carries the hash of the record before it, so removing or editing an
entry after the fact breaks every hash downstream. That does not make the log
unforgeable — anyone who can rewrite the whole chain can still rewrite it — but
it makes selective tampering detectable, which is the realistic threat when an
autonomous system is spending someone else's money and a single awkward entry is
the one worth deleting.

The chain is computed over a canonical JSON serialisation so that two processes
hashing the same record agree.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mainager.guard.ports import AuditSink

GENESIS_HASH = "0" * 64


class AuditRecord(BaseModel):
    """One entry in the chain."""

    model_config = ConfigDict(frozen=True)

    sequence: int
    recorded_at: str
    action: str
    outcome: str
    previous_hash: str
    record_hash: str
    detail: dict[str, Any] = Field(default_factory=dict)

    def payload_for_hashing(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "recorded_at": self.recorded_at,
            "action": self.action,
            "outcome": self.outcome,
            "previous_hash": self.previous_hash,
            "detail": self.detail,
        }


def compute_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditLog:
    """Appends records to a sink, chaining each to the one before it."""

    def __init__(self, sink: AuditSink) -> None:
        self._sink = sink

    async def record(
        self,
        action: str,
        outcome: str,
        detail: dict[str, Any] | None = None,
        *,
        now: datetime | None = None,
    ) -> AuditRecord:
        previous = await self._sink.last()
        sequence = int(previous["sequence"]) + 1 if previous else 0
        previous_hash = str(previous["record_hash"]) if previous else GENESIS_HASH

        stamp = (now or datetime.now(UTC)).isoformat()
        payload: dict[str, Any] = {
            "sequence": sequence,
            "recorded_at": stamp,
            "action": action,
            "outcome": outcome,
            "previous_hash": previous_hash,
            "detail": detail or {},
        }
        record = AuditRecord(
            sequence=sequence,
            recorded_at=stamp,
            action=action,
            outcome=outcome,
            previous_hash=previous_hash,
            detail=detail or {},
            record_hash=compute_hash(payload),
        )
        await self._sink.append(record.model_dump())
        return record

    async def verify(self) -> tuple[bool, str | None]:
        """Walk the chain. Returns `(intact, first_problem)`."""
        records = await self._sink.all()
        expected_previous = GENESIS_HASH

        for index, raw in enumerate(records):
            record = AuditRecord(**raw)
            if record.sequence != index:
                return False, f"record {index} claims sequence {record.sequence}"
            if record.previous_hash != expected_previous:
                return False, f"record {index} does not chain to its predecessor"
            if compute_hash(record.payload_for_hashing()) != record.record_hash:
                return False, f"record {index} has been modified after it was written"
            expected_previous = record.record_hash

        return True, None
