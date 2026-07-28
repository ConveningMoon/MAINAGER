"""In-process implementations of the guard ports.

These exist so the guard's logic can be tested and run without infrastructure.
They are not a substitute for the real thing: nothing here survives a restart or
is shared between workers, and a kill switch that only one process can see is not
a kill switch. Use them for tests and single-process development; use the Redis
and PostgreSQL adapters in anything that matters.
"""

from __future__ import annotations

import math
import time
from typing import Any


class MemoryKeyValueStore:
    """Dict-backed store with real expiry semantics."""

    def __init__(self) -> None:
        self._values: dict[str, tuple[str, float | None]] = {}
        self._counters: dict[str, tuple[int, float]] = {}

    def _live(self, key: str) -> str | None:
        entry = self._values.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.monotonic() >= expires_at:
            del self._values[key]
            return None
        return value

    async def get(self, key: str) -> str | None:
        return self._live(key)

    async def set(self, key: str, value: str, *, ttl_s: int | None = None) -> None:
        expires_at = time.monotonic() + ttl_s if ttl_s is not None else None
        self._values[key] = (value, expires_at)

    async def delete(self, key: str) -> None:
        self._values.pop(key, None)
        self._counters.pop(key, None)

    async def incr_in_window(self, key: str, *, window_s: int) -> int:
        now = time.monotonic()
        entry = self._counters.get(key)
        if entry is None or now >= entry[1]:
            self._counters[key] = (1, now + window_s)
            return 1
        count, expires_at = entry
        self._counters[key] = (count + 1, expires_at)
        return count + 1


class MemoryAuditSink:
    """Append-only list."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    async def append(self, record: dict[str, Any]) -> None:
        self._records.append(dict(record))

    async def last(self) -> dict[str, Any] | None:
        return dict(self._records[-1]) if self._records else None

    async def all(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._records]

    def tamper(self, index: int, **changes: Any) -> None:
        """Edit a stored record in place. Exists only so tests can prove the
        chain notices."""
        self._records[index].update(changes)


class MemoryVectorIndex:
    """Brute-force cosine similarity. Fine for tests, linear in corpus size."""

    def __init__(self) -> None:
        self._items: list[tuple[str, list[float], dict[str, Any]]] = []

    async def add(self, key: str, vector: list[float], payload: dict[str, Any]) -> None:
        self._items.append((key, list(vector), dict(payload)))

    async def nearest(
        self, vector: list[float], *, limit: int = 1
    ) -> list[tuple[float, str, dict[str, Any]]]:
        scored = [
            (cosine_similarity(vector, stored), key, payload)
            for key, stored, payload in self._items
        ]
        scored.sort(key=lambda row: (-row[0], row[1]))
        return scored[:limit]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"dimension mismatch: {len(left)} vs {len(right)}")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


class HashingEmbedder:
    """Deterministic bag-of-character-trigrams embedding.

    Chosen deliberately over pulling in a sentence-transformer: it needs no model
    download, no GPU and no network, and it is enough to demonstrate the
    deduplication mechanism and to test it. It captures surface similarity, not
    meaning — two paraphrases with no shared trigrams will not match. A
    production deployment should swap in a real embedding model behind the same
    `Embedder` protocol, and the threshold will need re-tuning when it does.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        normalised = " ".join(text.lower().split())
        vector = [0.0] * self._dimensions
        if not normalised:
            return vector
        padded = f"  {normalised}  "
        for index in range(len(padded) - 2):
            trigram = padded[index : index + 3]
            bucket = hash_trigram(trigram) % self._dimensions
            vector[bucket] += 1.0
        return vector


def hash_trigram(trigram: str) -> int:
    """Stable across processes, unlike the built-in `hash` for str."""
    digest = 0
    for char in trigram:
        digest = (digest * 131 + ord(char)) & 0xFFFFFFFF
    return digest
