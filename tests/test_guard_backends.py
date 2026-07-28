"""Integration tests for the real Redis and pgvector adapters.

These run only when the services are reachable, which is the case in CI and for
anyone who has started `docker compose up`. They are skipped otherwise rather
than mocked: an adapter tested against a mock of itself proves nothing, and the
whole reason these adapters exist is behaviour the in-process ones cannot have.

    docker compose up -d
    MAINAGER_TEST_REDIS_URL=redis://localhost:6379/0 \
    MAINAGER_TEST_POSTGRES_DSN=postgresql://mainager:mainager@localhost:5432/mainager \
    pytest tests/test_guard_backends.py
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest

from mainager.guard.audit import AuditLog
from mainager.guard.memory import HashingEmbedder
from mainager.guard.plane import GuardConfig, GuardPlane, GuardRequest

REDIS_URL = os.environ.get("MAINAGER_TEST_REDIS_URL")
POSTGRES_DSN = os.environ.get("MAINAGER_TEST_POSTGRES_DSN")

needs_redis = pytest.mark.skipif(not REDIS_URL, reason="MAINAGER_TEST_REDIS_URL is not set")
needs_postgres = pytest.mark.skipif(
    not POSTGRES_DSN, reason="MAINAGER_TEST_POSTGRES_DSN is not set"
)

DIMENSIONS = 64


@pytest.fixture
async def redis_store() -> AsyncIterator[Any]:
    from mainager.guard.redis_store import RedisKeyValueStore

    assert REDIS_URL is not None
    store = RedisKeyValueStore.from_url(REDIS_URL)
    yield store
    await store.aclose()


@pytest.fixture
async def pg_connection() -> AsyncIterator[Any]:
    import psycopg

    assert POSTGRES_DSN is not None
    connection = await psycopg.AsyncConnection.connect(POSTGRES_DSN)
    yield connection
    await connection.close()


def _key(prefix: str) -> str:
    return f"test:{prefix}:{uuid.uuid4().hex}"


# --- redis -----------------------------------------------------------------


@needs_redis
async def test_values_round_trip(redis_store: Any) -> None:
    key = _key("kv")
    assert await redis_store.get(key) is None

    await redis_store.set(key, "engaged")
    assert await redis_store.get(key) == "engaged"

    await redis_store.delete(key)
    assert await redis_store.get(key) is None


@needs_redis
async def test_the_window_counter_expires_from_its_first_increment(
    redis_store: Any,
) -> None:
    """The TTL must be set once, not refreshed on every increment."""
    key = _key("loop")

    assert await redis_store.incr_in_window(key, window_s=60) == 1
    assert await redis_store.incr_in_window(key, window_s=60) == 2
    assert await redis_store.incr_in_window(key, window_s=60) == 3

    ttl = await redis_store._client.ttl(key)
    assert 0 < ttl <= 60


@needs_redis
async def test_a_kill_switch_set_by_one_store_is_seen_by_another(
    redis_store: Any,
) -> None:
    """The property the in-process store cannot have."""
    from mainager.guard.memory import MemoryAuditSink
    from mainager.guard.redis_store import RedisKeyValueStore

    assert REDIS_URL is not None
    other = RedisKeyValueStore.from_url(REDIS_URL)
    try:
        worker_a = GuardPlane(redis_store, AuditLog(MemoryAuditSink()))
        worker_b = GuardPlane(other, AuditLog(MemoryAuditSink()))

        await worker_a.trip_kill_switch("incident 4711")
        decision = await worker_b.evaluate(GuardRequest(action="read"))

        assert decision.outcome == "deny"
        assert "4711" in decision.reason
    finally:
        await redis_store.delete("mainager:kill_switch")
        await other.aclose()


# --- postgres --------------------------------------------------------------


@needs_postgres
async def test_the_chain_survives_a_round_trip_through_postgres(
    pg_connection: Any,
) -> None:
    from mainager.guard.postgres import PostgresAuditSink

    sink = PostgresAuditSink(pg_connection)
    await sink.create_schema()
    async with pg_connection.cursor() as cursor:
        await cursor.execute("TRUNCATE mainager_audit")
    await pg_connection.commit()

    log = AuditLog(sink)
    await log.record("generate", "allow", {"cost_rub": 36})
    await log.record("generate", "deny", {"cost_rub": 316})
    await log.record("campaign_write", "shadow", {})

    intact, problem = await log.verify()
    assert intact, problem


@needs_postgres
async def test_tampering_in_the_database_is_detected(pg_connection: Any) -> None:
    from mainager.guard.postgres import PostgresAuditSink

    sink = PostgresAuditSink(pg_connection)
    await sink.create_schema()
    async with pg_connection.cursor() as cursor:
        await cursor.execute("TRUNCATE mainager_audit")
    await pg_connection.commit()

    log = AuditLog(sink)
    await log.record("generate", "allow", {"cost_rub": 36})
    await log.record("generate", "allow", {"cost_rub": 316})

    async with pg_connection.cursor() as cursor:
        await cursor.execute(
            "UPDATE mainager_audit SET detail = %s WHERE sequence = 1",
            ('{"cost_rub": 1}',),
        )
    await pg_connection.commit()

    intact, problem = await log.verify()
    assert intact is False
    assert problem is not None and "modified" in problem


@needs_postgres
async def test_a_duplicate_sequence_is_rejected_rather_than_forking_the_chain(
    pg_connection: Any,
) -> None:
    from mainager.guard.postgres import PostgresAuditSink

    sink = PostgresAuditSink(pg_connection)
    await sink.create_schema()
    async with pg_connection.cursor() as cursor:
        await cursor.execute("TRUNCATE mainager_audit")
    await pg_connection.commit()

    log = AuditLog(sink)
    record = await log.record("generate", "allow")

    clashing = dict(record.model_dump())
    clashing["record_hash"] = "f" * 64
    with pytest.raises(Exception, match=r"(?i)duplicate|unique"):
        await sink.append(clashing)
    await pg_connection.rollback()


@needs_postgres
async def test_pgvector_finds_the_near_duplicate(pg_connection: Any) -> None:
    from mainager.guard.postgres import PgVectorIndex

    index = PgVectorIndex(pg_connection, DIMENSIONS)
    await index.create_schema()
    async with pg_connection.cursor() as cursor:
        await cursor.execute("TRUNCATE mainager_generations")
    await pg_connection.commit()

    embedder = HashingEmbedder(DIMENSIONS)
    await index.add(
        "gen_1",
        await embedder.embed("a cat walking on a beach"),
        {"display_url": "https://cdn/x.mp4"},
    )
    await index.add(
        "gen_2",
        await embedder.embed("quarterly revenue chart for the board"),
        {},
    )

    matches = await index.nearest(await embedder.embed("A cat walking on a beach"), limit=1)

    assert matches
    similarity, key, payload = matches[0]
    assert key == "gen_1"
    assert similarity > 0.95
    assert payload["display_url"] == "https://cdn/x.mp4"


@needs_postgres
async def test_the_guard_deduplicates_through_pgvector(pg_connection: Any) -> None:
    from mainager.guard.memory import MemoryAuditSink, MemoryKeyValueStore
    from mainager.guard.postgres import PgVectorIndex

    index = PgVectorIndex(pg_connection, DIMENSIONS)
    await index.create_schema()
    async with pg_connection.cursor() as cursor:
        await cursor.execute("TRUNCATE mainager_generations")
    await pg_connection.commit()

    plane = GuardPlane(
        MemoryKeyValueStore(),
        AuditLog(MemoryAuditSink()),
        GuardConfig(dedup_threshold=0.95, loop_threshold=0),
        embedder=HashingEmbedder(DIMENSIONS),
        index=index,
    )
    await plane.remember_generation(
        "gen_1", "a cat walking on a beach", {"display_url": "https://cdn/x.mp4"}
    )

    decision = await plane.evaluate(
        GuardRequest(action="generate", prompt="A cat walking on a beach")
    )

    assert decision.outcome == "reuse"
    assert decision.detail["existing"]["display_url"] == "https://cdn/x.mp4"
