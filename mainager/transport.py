"""Retry and rate-limit policy for calls that can cost money.

Two rules drive everything here, and both come from the fact that a retry is not
free when the endpoint bills.

**Never retry blindly after a charge may have landed.** A 429 or a 5xx on a paid
call is ambiguous: the request may have been rejected, or it may have been
accepted and the response lost. Retrying the first case is correct; retrying the
second bills twice. The rule is therefore: a paid call is retried only when it
carries an idempotency key, because that is the only thing that makes the second
attempt safe. Without one, the failure is surfaced to the operator instead.

**Stay under the limit rather than discovering it.** Reacting to 429s means the
limit is found by exceeding it, and on a channel with a four-second budget a
rejected request has already cost the window. A token bucket per scope keeps
calls under the documented rate without ever provoking the error.

`402 insufficient_balance` is never retried under any circumstances. More money
will not appear because the client asked again.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

import httpx

Scope = Literal["read", "generate", "write", "autopilot", "yandex"]

#: Documented per-scope ceilings, in requests per minute.
SCOPE_LIMITS: dict[Scope, int] = {
    "read": 120,
    "generate": 30,
    "write": 10,
    "autopilot": 5,
    "yandex": 120,
}

#: Status codes worth another attempt, provided the call is safe to repeat.
RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})

#: Error codes that must never be retried, whatever the status code says.
TERMINAL_ERRORS: frozenset[str] = frozenset(
    {
        "insufficient_balance",
        "daily_spend_limit_exceeded",
        "idempotency_key_conflict",
        "duplicate_request",
        "insufficient_scope",
        "invalid_token",
        "missing_token",
        "validation_failed",
    }
)


class UnsafeRetryError(RuntimeError):
    """A billable call failed ambiguously and cannot be repeated safely.

    Raised instead of retrying when no idempotency key was supplied. The caller
    must decide, because only the caller knows whether a duplicate charge is
    acceptable.
    """

    def __init__(self, status_code: int, attempts: int) -> None:
        super().__init__(
            f"HTTP {status_code} on a billable call with no idempotency key; "
            f"stopped after {attempts} attempt(s) rather than risk a double charge"
        )
        self.status_code = status_code
        self.attempts = attempts


@dataclass
class RetryPolicy:
    """Exponential backoff with full jitter."""

    max_attempts: int = 4
    base_delay_s: float = 0.5
    max_delay_s: float = 20.0
    #: Injected so tests do not sleep and so the jitter is reproducible.
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    #: (low, high) -> delay. Defaults to a uniform draw across the window.
    jitter: Callable[[float, float], float] = random.uniform

    def delay_for(self, attempt: int, retry_after_s: float | None = None) -> float:
        """Seconds to wait before `attempt` (1-based). Server hint wins."""
        if retry_after_s is not None:
            return min(retry_after_s, self.max_delay_s)
        ceiling = min(self.base_delay_s * (2 ** (attempt - 1)), self.max_delay_s)
        # Full jitter: uniform across the whole window, not ceiling/2 + noise.
        # Retrying in lockstep is how a fleet turns one blip into an outage.
        return self.jitter(0.0, ceiling)


@dataclass
class TokenBucket:
    """Per-scope limiter that paces calls instead of provoking 429s."""

    capacity: int
    per_seconds: float = 60.0
    tokens: float = field(init=False)
    #: None until the first call. A float sentinel cannot work here: 0.0 is a
    #: perfectly ordinary reading from a monotonic clock, and treating it as
    #: "never seen" stops the bucket from ever refilling.
    _updated: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)

    def _refill(self, now: float) -> None:
        if self._updated is None:
            self._updated = now
            return
        elapsed = now - self._updated
        self._updated = now
        self.tokens = min(
            float(self.capacity), self.tokens + elapsed * (self.capacity / self.per_seconds)
        )

    def take(self, now: float) -> float:
        """Consume a token, returning how long the caller should wait first."""
        self._refill(now)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return 0.0
        deficit = 1.0 - self.tokens
        self.tokens = 0.0
        return deficit * (self.per_seconds / self.capacity)


def retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def error_code(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    return str(payload["error"]) if isinstance(payload, dict) and "error" in payload else None


def should_retry(response: httpx.Response) -> bool:
    """Whether the response is worth another attempt at all."""
    if response.status_code not in RETRYABLE_STATUS:
        return False
    code = error_code(response)
    return not (code is not None and code in TERMINAL_ERRORS)


class ResilientCaller:
    """Wraps a request function with pacing and safe retries."""

    def __init__(
        self,
        policy: RetryPolicy | None = None,
        limits: dict[Scope, int] | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._policy = policy or RetryPolicy()
        self._clock = clock or asyncio.get_event_loop().time
        self._buckets = {
            scope: TokenBucket(capacity) for scope, capacity in (limits or SCOPE_LIMITS).items()
        }
        self.attempts_made = 0
        self.waits_s: list[float] = []

    async def call(
        self,
        send: Callable[[], Awaitable[httpx.Response]],
        *,
        scope: Scope = "read",
        billable: bool = False,
        idempotency_key: str | None = None,
    ) -> httpx.Response:
        """Send, pacing beforehand and retrying only when it is safe to.

        A billable call without an idempotency key is never repeated: it raises
        ``UnsafeRetryError`` so the operator decides.
        """
        bucket = self._buckets.get(scope)
        response: httpx.Response | None = None

        for attempt in range(1, self._policy.max_attempts + 1):
            if bucket is not None:
                wait = bucket.take(self._clock())
                if wait > 0:
                    self.waits_s.append(wait)
                    await self._policy.sleep(wait)

            self.attempts_made = attempt
            response = await send()

            if not should_retry(response):
                return response

            repeat_is_safe = not billable or idempotency_key is not None
            if not repeat_is_safe:
                raise UnsafeRetryError(response.status_code, attempt)

            if attempt == self._policy.max_attempts:
                return response

            delay = self._policy.delay_for(attempt, retry_after_seconds(response))
            self.waits_s.append(delay)
            await self._policy.sleep(delay)

        assert response is not None  # loop always assigns before exiting
        return response
