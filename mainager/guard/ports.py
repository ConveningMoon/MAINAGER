"""Ports the guard plane depends on.

The guard needs three kinds of state: something fast and shared for flags and
sliding windows, something durable for the audit trail, and something that can
answer nearest-neighbour queries for semantic deduplication. Redis, PostgreSQL
and pgvector are the intended implementations, but the plane is written against
these protocols so its logic can be tested without any of them running.

Every method is async. The guard sits in front of a channel with a four-second
budget before the platform's own auto-responder takes over, so nothing here may
block an event loop.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class KeyValueStore(Protocol):
    """Shared, expiring key-value state: kill switch, loop windows, rate limits."""

    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, *, ttl_s: int | None = None) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def incr_in_window(self, key: str, *, window_s: int) -> int:
        """Increment a counter that expires `window_s` after its first increment.

        Returns the count after incrementing. The expiry must be set only when
        the counter is created, otherwise a steady stream of events keeps
        pushing the window forward and it never closes.
        """
        ...


@runtime_checkable
class AuditSink(Protocol):
    """Append-only destination for the hash-chained audit log."""

    async def append(self, record: dict[str, Any]) -> None: ...

    async def last(self) -> dict[str, Any] | None: ...

    async def all(self) -> list[dict[str, Any]]: ...


@runtime_checkable
class Embedder(Protocol):
    """Turns a prompt into a vector for similarity search."""

    @property
    def dimensions(self) -> int: ...

    async def embed(self, text: str) -> list[float]: ...


@runtime_checkable
class VectorIndex(Protocol):
    """Nearest-neighbour lookup over previously generated prompts."""

    async def add(self, key: str, vector: list[float], payload: dict[str, Any]) -> None: ...

    async def nearest(
        self, vector: list[float], *, limit: int = 1
    ) -> list[tuple[float, str, dict[str, Any]]]:
        """Return `(cosine_similarity, key, payload)`, most similar first."""
        ...
