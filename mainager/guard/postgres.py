"""PostgreSQL adapters: the durable audit log and the pgvector dedup index.

The audit log is append-only by construction — there is no update or delete path
in this module, and `sequence` carries a unique constraint so two workers cannot
both claim the same position in the chain. A concurrent append loses the race
with a uniqueness violation rather than silently forking the chain, which is the
behaviour worth having when the log is the record of who spent what.

Similarity uses pgvector's cosine distance operator, so the ordering happens in
the index rather than in Python.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from psycopg import AsyncConnection

# `recorded_at` is TEXT on purpose. The hash is computed over the timestamp
# exactly as it was written, so it has to come back byte-identical. A TIMESTAMPTZ
# round-trips through the session time zone: write "…+00:00" from a worker,
# read it back from a session running in Europe/Moscow, and you get "…+03:00" —
# the same instant, a different string, and a chain that fails verification for
# no reason. `queried_at` keeps the value queryable without being part of the hash.
AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS mainager_audit (
    sequence      BIGINT      PRIMARY KEY,
    recorded_at   TEXT        NOT NULL,
    queried_at    TIMESTAMPTZ NOT NULL,
    action        TEXT        NOT NULL,
    outcome       TEXT        NOT NULL,
    previous_hash CHAR(64)    NOT NULL,
    record_hash   CHAR(64)    NOT NULL UNIQUE,
    detail        JSONB       NOT NULL DEFAULT '{}'::jsonb
);
"""

VECTOR_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS mainager_generations (
    key       TEXT PRIMARY KEY,
    embedding vector(%(dimensions)s) NOT NULL,
    payload   JSONB NOT NULL DEFAULT '{}'::jsonb
);
"""

_VECTOR_INDEX = """
CREATE INDEX IF NOT EXISTS mainager_generations_embedding_idx
ON mainager_generations USING hnsw (embedding vector_cosine_ops);
"""


def _as_vector(vector: list[float]) -> str:
    """pgvector's text input format."""
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


class PostgresAuditSink:
    """Durable, append-only sink for the hash chain."""

    def __init__(self, connection: AsyncConnection[Any]) -> None:
        self._connection = connection

    async def create_schema(self) -> None:
        async with self._connection.cursor() as cursor:
            await cursor.execute(AUDIT_SCHEMA)
        await self._connection.commit()

    async def append(self, record: dict[str, Any]) -> None:
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO mainager_audit
                    (sequence, recorded_at, queried_at, action, outcome,
                     previous_hash, record_hash, detail)
                VALUES (%s, %s, %s::timestamptz, %s, %s, %s, %s, %s)
                """,
                (
                    record["sequence"],
                    record["recorded_at"],
                    record["recorded_at"],
                    record["action"],
                    record["outcome"],
                    record["previous_hash"],
                    record["record_hash"],
                    json.dumps(record.get("detail") or {}, ensure_ascii=False),
                ),
            )
        await self._connection.commit()

    async def last(self) -> dict[str, Any] | None:
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT sequence, recorded_at, action, outcome,
                       previous_hash, record_hash, detail
                FROM mainager_audit ORDER BY sequence DESC LIMIT 1
                """
            )
            row = await cursor.fetchone()
        return None if row is None else _row_to_record(row)

    async def all(self) -> list[dict[str, Any]]:
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT sequence, recorded_at, action, outcome,
                       previous_hash, record_hash, detail
                FROM mainager_audit ORDER BY sequence
                """
            )
            rows = await cursor.fetchall()
        return [_row_to_record(row) for row in rows]


def _row_to_record(row: tuple[Any, ...]) -> dict[str, Any]:
    sequence, recorded_at, action, outcome, previous_hash, record_hash, detail = row
    return {
        "sequence": int(sequence),
        "recorded_at": str(recorded_at),
        "action": action,
        "outcome": outcome,
        "previous_hash": previous_hash,
        "record_hash": record_hash,
        "detail": detail if isinstance(detail, dict) else json.loads(detail or "{}"),
    }


class PgVectorIndex:
    """Nearest-neighbour lookup over generated prompts."""

    def __init__(self, connection: AsyncConnection[Any], dimensions: int) -> None:
        self._connection = connection
        self._dimensions = dimensions

    async def create_schema(self) -> None:
        async with self._connection.cursor() as cursor:
            await cursor.execute(VECTOR_SCHEMA, {"dimensions": self._dimensions})
            await cursor.execute(_VECTOR_INDEX)
        await self._connection.commit()

    async def add(self, key: str, vector: list[float], payload: dict[str, Any]) -> None:
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO mainager_generations (key, embedding, payload)
                VALUES (%s, %s::vector, %s)
                ON CONFLICT (key) DO UPDATE
                    SET embedding = EXCLUDED.embedding, payload = EXCLUDED.payload
                """,
                (key, _as_vector(vector), json.dumps(payload, ensure_ascii=False)),
            )
        await self._connection.commit()

    async def nearest(
        self, vector: list[float], *, limit: int = 1
    ) -> list[tuple[float, str, dict[str, Any]]]:
        async with self._connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT 1 - (embedding <=> %s::vector) AS similarity, key, payload
                FROM mainager_generations
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (_as_vector(vector), _as_vector(vector), limit),
            )
            rows = await cursor.fetchall()
        return [
            (
                float(similarity),
                str(key),
                payload if isinstance(payload, dict) else json.loads(payload or "{}"),
            )
            for similarity, key, payload in rows
        ]
