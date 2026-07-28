"""Redis-backed shared state for the guard plane.

The kill switch and the loop-detector windows only mean anything if every worker
sees the same value, which is the whole reason this adapter exists and the
in-process one is not enough.

`incr_in_window` runs as a Lua script so the increment and the expiry are one
atomic operation. Doing it as INCR followed by a conditional EXPIRE leaves a
window where a process can die between the two and leave a counter that never
expires — a loop detector that has silently latched on is worse than none,
because it denies real traffic forever.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from redis.asyncio import Redis

# KEYS[1] = counter key, ARGV[1] = window in seconds.
# Sets the TTL only when the counter is created, so a steady stream of requests
# cannot keep pushing the window forward and prevent it from ever closing.
_INCR_IN_WINDOW = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return count
"""


class RedisKeyValueStore:
    """`KeyValueStore` over redis.asyncio."""

    def __init__(self, client: Redis) -> None:
        self._client = client
        self._incr_script = client.register_script(_INCR_IN_WINDOW)

    @classmethod
    def from_url(cls, url: str) -> RedisKeyValueStore:
        from redis.asyncio import Redis as AsyncRedis

        return cls(AsyncRedis.from_url(url, decode_responses=True))

    async def get(self, key: str) -> str | None:
        value: Any = await self._client.get(key)
        return None if value is None else str(value)

    async def set(self, key: str, value: str, *, ttl_s: int | None = None) -> None:
        await self._client.set(key, value, ex=ttl_s)

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    async def incr_in_window(self, key: str, *, window_s: int) -> int:
        count: Any = await self._incr_script(keys=[key], args=[window_s])
        return int(count)

    async def aclose(self) -> None:
        await self._client.aclose()
