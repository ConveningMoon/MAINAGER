"""Retry and rate-limit policy.

The property that matters is negative, as it was for the MCP proxy: a billable
call that fails ambiguously must not be repeated unless repeating it is provably
safe. Everything else here is pacing.
"""

from __future__ import annotations

import httpx
import pytest

from mainager.transport import (
    SCOPE_LIMITS,
    ResilientCaller,
    RetryPolicy,
    TokenBucket,
    UnsafeRetryError,
    error_code,
    retry_after_seconds,
    should_retry,
)


def _response(
    status: int, payload: dict | None = None, headers: dict | None = None
) -> httpx.Response:
    return httpx.Response(
        status, json=payload if payload is not None else {"ok": True}, headers=headers or {}
    )


class Sender:
    """Replays a scripted sequence of responses and counts the calls."""

    def __init__(self, *responses: httpx.Response) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def __call__(self) -> httpx.Response:
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]


def _caller(**kw: object) -> tuple[ResilientCaller, list[float]]:
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    clock = iter(range(0, 100000))
    policy = RetryPolicy(
        sleep=sleep,
        jitter=lambda _lo, hi: hi,  # deterministic: always the top of the window
        **kw,  # type: ignore[arg-type]
    )
    return ResilientCaller(policy, clock=lambda: float(next(clock))), slept


# --- classification --------------------------------------------------------


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_statuses_are_retryable(status: int) -> None:
    assert should_retry(_response(status)) is True


@pytest.mark.parametrize("status", [200, 201, 400, 401, 403, 404, 409, 415, 422])
def test_other_statuses_are_not(status: int) -> None:
    assert should_retry(_response(status)) is False


def test_running_out_of_money_is_never_retryable() -> None:
    """A 402 means the account is empty; asking again will not refill it."""
    assert should_retry(_response(429, {"error": "insufficient_balance"})) is False
    assert should_retry(_response(429, {"error": "daily_spend_limit_exceeded"})) is False


def test_duplicate_request_is_not_retryable() -> None:
    assert should_retry(_response(429, {"error": "duplicate_request"})) is False
    assert should_retry(_response(409, {"error": "idempotency_key_conflict"})) is False


def test_error_code_survives_a_non_json_body() -> None:
    assert error_code(httpx.Response(500, text="<html>oops</html>")) is None


def test_server_retry_after_is_read() -> None:
    assert retry_after_seconds(_response(429, headers={"Retry-After": "7"})) == 7.0
    assert retry_after_seconds(_response(429, headers={"Retry-After": "soon"})) is None
    assert retry_after_seconds(_response(429)) is None


# --- backoff ---------------------------------------------------------------


def test_backoff_grows_and_is_capped() -> None:
    policy = RetryPolicy(base_delay_s=0.5, max_delay_s=4.0, jitter=lambda _lo, hi: hi)
    assert [policy.delay_for(n) for n in (1, 2, 3, 4, 5)] == [0.5, 1.0, 2.0, 4.0, 4.0]


def test_a_server_hint_overrides_the_computed_delay() -> None:
    policy = RetryPolicy(max_delay_s=30.0, jitter=lambda _lo, hi: hi)
    assert policy.delay_for(1, retry_after_s=9.0) == 9.0
    assert policy.delay_for(1, retry_after_s=999.0) == 30.0  # still capped


def test_jitter_spans_the_whole_window() -> None:
    """Full jitter, not ceiling/2 — retrying in lockstep amplifies an outage."""
    lo = RetryPolicy(base_delay_s=1.0, jitter=lambda low, _hi: low)
    hi = RetryPolicy(base_delay_s=1.0, jitter=lambda _lo, high: high)
    assert lo.delay_for(3) == 0.0
    assert hi.delay_for(3) == 4.0


# --- the safety property ---------------------------------------------------


async def test_a_billable_call_without_a_key_is_never_repeated() -> None:
    """The one that costs money if it is wrong."""
    caller, _ = _caller()
    send = Sender(_response(429), _response(200))

    with pytest.raises(UnsafeRetryError) as excinfo:
        await caller.call(send, scope="generate", billable=True)

    assert send.calls == 1
    assert "double charge" in str(excinfo.value)


async def test_a_billable_call_with_a_key_is_repeated(caplog: object) -> None:
    caller, slept = _caller()
    send = Sender(_response(429), _response(200))

    response = await caller.call(send, scope="generate", billable=True, idempotency_key="k-1")

    assert response.status_code == 200
    assert send.calls == 2
    assert slept


async def test_a_free_call_is_repeated_without_a_key() -> None:
    caller, _ = _caller()
    send = Sender(_response(503), _response(503), _response(200))

    response = await caller.call(send, scope="read")

    assert response.status_code == 200
    assert send.calls == 3


async def test_a_terminal_error_stops_immediately() -> None:
    caller, slept = _caller()
    send = Sender(_response(429, {"error": "insufficient_balance"}))

    response = await caller.call(send, scope="generate", billable=True, idempotency_key="k")

    assert response.status_code == 429
    assert send.calls == 1
    assert slept == []


async def test_attempts_are_bounded() -> None:
    caller, _ = _caller(max_attempts=3)
    send = Sender(_response(503))

    response = await caller.call(send, scope="read")

    assert response.status_code == 503
    assert send.calls == 3


async def test_a_success_makes_no_extra_calls() -> None:
    caller, slept = _caller()
    send = Sender(_response(200))

    await caller.call(send, scope="read")

    assert send.calls == 1
    assert slept == []


# --- pacing ----------------------------------------------------------------


def test_the_bucket_starts_full_and_drains() -> None:
    bucket = TokenBucket(capacity=3, per_seconds=60.0)

    assert [bucket.take(0.0) for _ in range(3)] == [0.0, 0.0, 0.0]
    assert bucket.take(0.0) > 0  # fourth call in the same instant has to wait


def test_the_bucket_refills_over_time() -> None:
    bucket = TokenBucket(capacity=60, per_seconds=60.0)  # one per second
    for _ in range(60):
        bucket.take(0.0)

    assert bucket.take(0.0) > 0
    assert bucket.take(10.0) == 0.0  # ten seconds later there is room again


def test_the_wait_matches_the_configured_rate() -> None:
    bucket = TokenBucket(capacity=30, per_seconds=60.0)  # generate scope
    for _ in range(30):
        bucket.take(0.0)

    assert bucket.take(0.0) == pytest.approx(2.0)  # 60s / 30 = one every 2s


def test_documented_scope_ceilings_are_encoded() -> None:
    assert SCOPE_LIMITS["read"] == 120
    assert SCOPE_LIMITS["generate"] == 30
    assert SCOPE_LIMITS["write"] == 10
    assert SCOPE_LIMITS["autopilot"] == 5


async def test_scopes_do_not_share_a_budget() -> None:
    caller, _ = _caller()
    send = Sender(_response(200))

    for _ in range(30):
        await caller.call(send, scope="generate")
    # write has its own bucket and is untouched by the generate traffic
    await caller.call(send, scope="write")

    assert send.calls == 31
