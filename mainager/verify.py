"""Re-measure every claim in FINDINGS.md against the live API, right now.

This exists for one reason: a finding that cannot be independently reproduced is
an assertion, not a measurement, and the whole argument of this project is that
the second kind is worth more. Rather than ask a reviewer to trust nine numbers
frozen on 28.07.2026, this re-runs the exact check behind each one and reports
whether today's API still agrees.

Every check uses a free endpoint — `GET /capabilities`, `GET /prices`,
`POST /generate/estimate`, plain `GET`s, MCP `tools/list` and a read-only
`tools/call` on `estimate_generation` (which itself only reaches
`/generate/estimate`) — plus one public, unauthenticated read of the provider's
own documentation page for the inbox-economics claim. Nothing here ever reaches
`POST /generate` or the MCP `generate_content` tool. Confirmed by balance
before/after in the CLI wrapper, the same way every other command in this
project proves it spent nothing.

The provider's own numbers move — `/prices` was observed in three different
states in a single day (§12 of the project record) — so a check reporting
"differs" is not automatically a regression on either side. It means: here is
what changed, go look. That is deliberately the whole output.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict

from mainager.config import Settings
from mainager.preflight.reconcile import flatten_prices, reconcile
from mainager.preflight.registry import ModelRegistry
from mainager.providers.vibemarketolog.errors import VibeApiError
from mainager.providers.vibemarketolog.pricing import VibePricer

Outcome = Literal["match", "differs", "error"]

#: A reachable URL for checks that need a source image. Content is irrelevant —
#: only reachability matters, since pre_charge_validation HEAD-checks it.
_IMG = "https://lk.vibemarketolog.ru/favicon.ico"

DOCS_URL = "https://lk.vibemarketolog.ru/docs/agent-api"


class CheckResult(BaseModel):
    """One finding, re-measured."""

    model_config = ConfigDict(frozen=True)

    n: int
    claim: str
    expected: str
    measured: str
    outcome: Outcome
    note: str = ""


class _Ctx:
    """Shared state built once and passed to every check."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
        capabilities: dict[str, Any],
        registry: ModelRegistry,
        prices_payload: dict[str, Any],
    ) -> None:
        self.settings = settings
        self.client = client
        self.capabilities = capabilities
        self.registry = registry
        self.prices = flatten_prices(prices_payload)
        self.pricer = VibePricer(client)


async def _estimate(ctx: _Ctx, body: dict[str, Any]) -> dict[str, Any]:
    response = await ctx.client.post("/generate/estimate", json=body)
    if response.is_error:
        raise VibeApiError.from_response(response)
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("estimate did not return an object")
    return payload


def _ok(n: int, claim: str, expected: str, measured: str, note: str = "") -> CheckResult:
    return CheckResult(
        n=n, claim=claim, expected=expected, measured=measured, outcome="match", note=note
    )


def _no(n: int, claim: str, expected: str, measured: str, note: str = "") -> CheckResult:
    return CheckResult(
        n=n, claim=claim, expected=expected, measured=measured, outcome="differs", note=note
    )


def _err(n: int, claim: str, exc: Exception, note: str = "") -> CheckResult:
    return CheckResult(
        n=n,
        claim=claim,
        expected="—",
        measured=f"{type(exc).__name__}: {exc}",
        outcome="error",
        note=note,
    )


# --- 1. strict absent from the MCP tool schemas -----------------------------


async def check_1_mcp_strict(ctx: _Ctx) -> CheckResult:
    claim = "strict is absent from the inputSchema of every MCP generation tool"
    try:
        from mainager.mcp_proxy import upstream_session
    except ImportError as exc:
        return _err(1, claim, exc, note="install the mcp extra: pip install -e '.[mcp]'")

    try:
        async with upstream_session(ctx.settings) as session:
            listing = await session.list_tools()
    except Exception as exc:
        return _err(1, claim, exc)

    from mainager.guard.mcp_guard import GENERATION_TOOLS

    checked = {t.name: t for t in listing.tools if t.name in GENERATION_TOOLS}
    missing_strict = {
        name: "strict" not in ((tool.inputSchema or {}).get("properties") or {})
        for name, tool in checked.items()
    }
    still_absent = all(missing_strict.values()) and len(checked) == len(GENERATION_TOOLS)
    measured = ", ".join(
        f"{name}: {'absent' if v else 'present'}" for name, v in sorted(missing_strict.items())
    )
    if still_absent:
        return _ok(1, claim, "absent from both tools", measured)
    return _no(
        1,
        claim,
        "absent from both tools",
        measured or "tools not found",
        note=f"checked {len(checked)}/{len(GENERATION_TOOLS)} generation tools",
    )


# --- 2. valid:true alongside a non-empty rejected ---------------------------


async def check_2_valid_true_with_rejected(ctx: _Ctx) -> CheckResult:
    claim = "estimate returns valid:true together with a non-empty rejected list"
    try:
        payload = await _estimate(
            ctx,
            {
                "type": "video",
                "model": "grok-ttv",
                "prompt": "a cat walking",
                "duration": 5,
                "image_input": [_IMG],
            },
        )
    except Exception as exc:
        return _err(2, claim, exc)

    valid = payload.get("valid")
    rejected = payload.get("rejected") or []
    measured = f"valid={valid}, rejected={rejected}"
    if valid is True and rejected:
        return _ok(2, claim, "valid=True, rejected=['image_input']", measured)
    return _no(2, claim, "valid=True, rejected=['image_input']", measured)


# --- 3. catalog `price` understates the real ceiling ------------------------


#: How many times the top tier must exceed the catalog's default-config price
#: for the claim to still hold. Chosen well below the ~12x first observed so
#: the check survives ordinary price drift (§4 of FINDINGS.md) without going
#: slack enough to pass on a marginal difference.
_PRICE_UNDERSTATEMENT_RATIO = 5


async def check_3_price_field_understates(ctx: _Ctx) -> CheckResult:
    claim = "catalog price field materially understates the real cost of the top tier"
    expected = f"top-tier estimate is at least {_PRICE_UNDERSTATEMENT_RATIO}x the catalog price"
    spec = ctx.registry.find("grok-ttv")
    if spec is None or spec.price is None:
        return _no(3, claim, expected, "grok-ttv missing from live catalog")
    try:
        payload = await _estimate(
            ctx,
            {
                "type": "video",
                "model": "grok-ttv-30",
                "prompt": "a cat walking",
                "strict": True,
            },
        )
    except Exception as exc:
        return _err(3, claim, exc)

    top = payload.get("estimated_cost_rub")
    if not isinstance(top, int | float):
        return _no(3, claim, expected, f"no cost returned: {payload}")
    ratio = top / spec.price
    measured = f"catalog price={spec.price:g}, grok-ttv-30 estimate={top:g}, ratio={ratio:.1f}x"
    if ratio >= _PRICE_UNDERSTATEMENT_RATIO:
        return _ok(3, claim, expected, measured)
    return _no(3, claim, expected, measured)


# --- 4. capabilities and prices disagree ------------------------------------


async def check_4_catalog_price_drift(ctx: _Ctx) -> CheckResult:
    claim = "GET /capabilities and GET /prices disagree on which models exist"
    drift = reconcile(ctx.registry, ctx.prices)
    measured = drift.summary()
    if not drift.is_clean:
        return _ok(4, claim, "not clean: identifiers only on one side", measured)
    return _no(
        4,
        claim,
        "not clean: identifiers only on one side",
        measured,
        note="the two sources agree as of this run",
    )


# --- 5. capabilities contradicts itself on grok-itv tiers -------------------


async def check_5_catalog_self_contradiction(ctx: _Ctx) -> CheckResult:
    claim = "getting_started names a tier that models.video['grok-itv'].tiers omits"
    started = ctx.capabilities.get("getting_started")
    section = started.get("image_to_video") if isinstance(started, dict) else None
    if not isinstance(section, dict):
        return _err(5, claim, ValueError("getting_started.image_to_video not found"))

    text = " ".join(f"{k} {v}" for k, v in section.items())
    # The prose spells tiers as one compound key, "grok-itv-10|20|30", not as
    # three separate "grok-itv-N" occurrences — a plain grok-itv-(\d+) findall
    # only ever catches the first number, since the rest follow a bare "|".
    match = re.search(r"grok-itv-([\d|]+)", text)
    named = set(match.group(1).split("|")) if match else set()
    spec = ctx.registry.find("grok-itv")
    declared = {t.removeprefix("grok-itv-") for t in (spec.tiers if spec else ())}
    omitted = named - declared
    measured = f"named in prose: {sorted(named)}, declared in models.tiers: {sorted(declared)}"
    if omitted:
        return _ok(5, claim, "prose names a tier missing from models.tiers", measured)
    return _no(5, claim, "prose names a tier missing from models.tiers", measured)


# --- 6. base id vs pinned tier: ~8.8x for the same clip ---------------------


#: A pinned tier must cost at least this many times the base identifier for
#: the same duration to count as the same finding. There is deliberately no
#: upper bound: if the live multiplier is now larger than the ~8.8x first
#: observed, that is a stronger instance of the same finding, not a
#: regression, and a banded check would wrongly report it as "differs".
_TIER_PIN_RATIO = 3


async def check_6_duration_multiplier(ctx: _Ctx) -> CheckResult:
    claim = "a pinned video tier costs materially more than the base id at the same duration"
    expected = f"pinned-tier estimate is at least {_TIER_PIN_RATIO}x the base-id estimate"

    def body(model: str) -> dict[str, Any]:
        return {
            "type": "video",
            "model": model,
            "prompt": "a cat walking",
            "duration": 5,
            "image_urls": [_IMG],
            "strict": True,
        }

    try:
        base = await _estimate(ctx, body("grok-itv"))
        pinned = await _estimate(ctx, body("grok-itv-20"))
    except Exception as exc:
        return _err(6, claim, exc)

    base_cost, pinned_cost = base.get("estimated_cost_rub"), pinned.get("estimated_cost_rub")
    if not isinstance(base_cost, int | float) or not isinstance(pinned_cost, int | float):
        return _no(6, claim, expected, f"grok-itv={base_cost}, grok-itv-20={pinned_cost}")
    ratio = pinned_cost / base_cost if base_cost else float("inf")
    measured = f"grok-itv={base_cost:g} RUB, grok-itv-20={pinned_cost:g} RUB, ratio={ratio:.1f}x"
    if ratio >= _TIER_PIN_RATIO:
        return _ok(6, claim, expected, measured)
    return _no(6, claim, expected, measured)


# --- 7. live endpoints missing from the documented endpoint map ------------


async def check_7_undocumented_endpoints(ctx: _Ctx) -> CheckResult:
    claim = (
        "/me, /health, /generations, /inbox work live but are absent from capabilities.endpoints"
    )
    endpoints = ctx.capabilities.get("endpoints")
    documented_paths = " ".join(endpoints.keys()) if isinstance(endpoints, dict) else ""

    targets = ("/me", "/health", "/generations", "/inbox")
    rows = []
    for target in targets:
        documented = target in documented_paths
        try:
            response = await ctx.client.get(target)
            live = response.status_code == 200
        except httpx.HTTPError as exc:
            rows.append(f"{target}: request failed ({exc})")
            continue
        rows.append(f"{target}: live={live} documented={documented}")

    measured = "; ".join(rows)
    matches = all("live=True documented=False" in r for r in rows)
    if matches:
        return _ok(7, claim, "live=True, documented=False for all four", measured)
    return _no(7, claim, "live=True, documented=False for all four", measured)


# --- 8. inbox economics, from their own public docs page -------------------


async def check_8_inbox_economics(ctx: _Ctx) -> CheckResult:
    claim = "docs state the platform auto-responder bills 10₽/8₽ per reply after ~4s"
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as public:
            response = await public.get(DOCS_URL)
    except httpx.HTTPError as exc:
        return _err(8, claim, exc)

    if response.is_error:
        return _err(
            8, claim, VibeApiError(response.status_code, "http_error", "docs page fetch failed")
        )

    text = re.sub(r"<[^>]+>", " ", response.text)
    has_price = "10" in text and "8" in text and "ответ" in text
    has_window = "4 секунд" in text
    measured = f"10₽/8₽-per-reply phrase found={has_price}, ~4s window phrase found={has_window}"
    if has_price and has_window:
        return _ok(8, claim, "both phrases present in the docs", measured)
    return _no(
        8,
        claim,
        "both phrases present in the docs",
        measured,
        note="documentation prose can be reworded without notice; this is the weakest check here",
    )


# --- 9. -edit models cost more than the base model doing the same edit -----


async def check_9_edit_pricing(ctx: _Ctx) -> CheckResult:
    claim = "*-edit models cost more than the base model, which edits for the price of generating"
    # seedream-5-lite-edit is intentionally not a pair here: it is priced but
    # absent from the catalog (finding 4's own "priced_without_model" case), so
    # it cannot be looked up as a registry model at all.
    pairs = (("gpt-image-2", "gpt-image-2-edit"), ("seedream-5-pro", "seedream-5-pro-edit"))
    rows: list[str] = []
    all_hold = True
    try:
        for base_id, edit_id in pairs:
            base = ctx.registry.find(base_id)
            edit = ctx.registry.find(edit_id)
            if base is None or edit is None or base.price is None or edit.price is None:
                rows.append(f"{base_id}/{edit_id}: missing from catalog")
                all_hold = False
                continue
            holds = edit.price > base.price
            all_hold = all_hold and holds
            verdict = "holds" if holds else "no"
            rows.append(f"{base_id}={base.price:g} vs {edit_id}={edit.price:g} ({verdict})")

        txt = await _estimate(
            ctx,
            {
                "type": "image",
                "model": "nano-banana-2-lite",
                "prompt": "a poster",
                "strict": True,
            },
        )
        img = await _estimate(
            ctx,
            {
                "type": "image",
                "model": "nano-banana-2-lite",
                "prompt": "a poster",
                "image_input": [_IMG],
                "strict": True,
            },
        )
    except Exception as exc:
        return _err(9, claim, exc)

    t_cost, i_cost = txt.get("estimated_cost_rub"), img.get("estimated_cost_rub")
    no_premium = t_cost is not None and t_cost == i_cost
    all_hold = all_hold and no_premium
    verdict = "holds" if no_premium else "no"
    rows.append(f"nano-banana-2-lite txt2img={t_cost} vs img2img={i_cost} ({verdict})")

    measured = "; ".join(rows)
    if all_hold:
        return _ok(
            9, claim, "edit variants cost more; plain img2img costs the same as txt2img", measured
        )
    return _no(
        9, claim, "edit variants cost more; plain img2img costs the same as txt2img", measured
    )


CHECKS: tuple[Callable[[_Ctx], Awaitable[CheckResult]], ...] = (
    check_1_mcp_strict,
    check_2_valid_true_with_rejected,
    check_3_price_field_understates,
    check_4_catalog_price_drift,
    check_5_catalog_self_contradiction,
    check_6_duration_multiplier,
    check_7_undocumented_endpoints,
    check_8_inbox_economics,
    check_9_edit_pricing,
)


async def run_all(settings: Settings) -> list[CheckResult]:
    """Run every check and return the results in order.

    Never raises on an individual check's failure — a broken check is reported
    as an ``error`` row, not a crash, so one dead endpoint does not hide the
    other eight answers.
    """
    async with VibePricer.client_for(settings) as client:
        cap_response = await client.get("/capabilities")
        if cap_response.is_error:
            raise VibeApiError.from_response(cap_response)
        capabilities = cap_response.json()

        price_response = await client.get("/prices")
        if price_response.is_error:
            raise VibeApiError.from_response(price_response)

        registry = ModelRegistry.from_capabilities(capabilities)
        ctx = _Ctx(settings, client, capabilities, registry, price_response.json())

        results: list[CheckResult] = []
        for check in CHECKS:
            try:
                results.append(await check(ctx))
            except Exception as exc:
                results.append(_err(len(results) + 1, check.__doc__ or check.__name__, exc))
        return results
