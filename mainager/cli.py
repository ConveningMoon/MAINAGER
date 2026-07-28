"""Command line entry point."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from mainager.config import Settings, load_settings
from mainager.postflight.overspend import overspend_report, parse_generations
from mainager.preflight.reconcile import flatten_prices, reconcile, unlisted_tiers
from mainager.preflight.registry import ModelRegistry
from mainager.providers.vibemarketolog.capabilities import (
    fetch_capabilities,
    read_snapshot,
    write_snapshot,
)
from mainager.providers.vibemarketolog.errors import VibeApiError
from mainager.providers.vibemarketolog.pricing import VibePricer


def _summarise_models(payload: dict[str, Any]) -> str:
    """One-line breakdown of the catalog, which is keyed by model type."""
    models = payload.get("models")
    if not isinstance(models, dict):
        return "no model catalog in payload"
    by_type = {
        name: len(entries) for name, entries in models.items() if isinstance(entries, dict | list)
    }
    if not by_type:
        return "no model catalog in payload"
    breakdown = ", ".join(f"{name} {count}" for name, count in sorted(by_type.items()))
    return f"{sum(by_type.values())} models ({breakdown})"


async def _dump_capabilities(settings: Settings) -> int:
    payload = await fetch_capabilities(settings)
    destination = write_snapshot(payload, settings.data_dir)
    print(f"wrote {destination} — {_summarise_models(payload)}")
    return 0


async def _report_drift(settings: Settings, as_json: bool = False) -> int:
    """Compare the live catalog against the live price list.

    Text mode exits 1 when the two disagree, so it can gate a build. JSON mode
    always exits 0: it is a reading, not a verdict, and a monitor that fails on
    a finding stops recording exactly when the finding appears.
    """
    registry = ModelRegistry.from_capabilities(await fetch_capabilities(settings))
    async with VibePricer.client_for(settings) as client:
        response = await client.get("/prices")
        if response.is_error:
            raise VibeApiError.from_response(response)
        payload = response.json()
        prices = flatten_prices(payload)

    drift = reconcile(registry, prices)

    if as_json:
        unlisted, (parent, gap) = unlisted_tiers(registry, prices)
        print(
            json.dumps(
                {
                    "observed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    # Includes tiers expanded into their own callable ids, so this is
                    # larger than the 45 models the catalog lists as models.
                    "routable_ids": len(registry.ids()),
                    "price_keys": len(prices),
                    "hidden_non_generation_keys": payload.get("hidden_non_generation_keys"),
                    "generatable_without_price": list(drift.generatable_without_price),
                    "priced_without_model": list(drift.priced_without_model),
                    "category_mismatch": [list(m) for m in drift.category_mismatch],
                    "unlisted_tiers": [list(u) for u in unlisted],
                    "ceiling_gap_rub": gap,
                    "ceiling_gap_model": parent or None,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    print(drift.summary())
    for model_id in drift.generatable_without_price:
        print(f"  no price       {model_id}")
    for model_id in drift.priced_without_model:
        print(f"  no model       {model_id}")
    for model_id, ours, theirs in drift.category_mismatch:
        print(f"  category       {model_id}: catalog says {ours}, prices say {theirs}")

    unlisted, (parent, gap) = unlisted_tiers(registry, prices)
    for parent_id, tier, price in unlisted:
        print(f"  unlisted tier  {tier} at {price:g} RUB (not declared by {parent_id})")
    if gap:
        print(f"\na ceiling built from the catalog alone is {gap:g} RUB low on {parent}")
    return 0 if drift.is_clean else 1


async def _report_overspend(settings: Settings, from_file: Path | None) -> int:
    registry = ModelRegistry.from_capabilities(read_snapshot(settings.data_dir))

    async with VibePricer.client_for(settings) as client:
        if from_file is not None:
            payload = json.loads(from_file.read_text(encoding="utf-8"))
            print(f"reading history from {from_file}")
        else:
            response = await client.get("/generations")
            if response.is_error:
                raise VibeApiError.from_response(response)
            payload = response.json()

        history = parse_generations(payload)
        if not history:
            print("no generations on this account; nothing to analyse")
            return 0

        report = await overspend_report(history, registry, VibePricer(client))

    print(
        f"\n{'id':10} {'model paid':16} {'paid':>8} {'cheapest today':16} {'best':>8} {'saving':>8}"
    )
    for item in report.analysed:
        print(
            f"{item.generation_id:10} {item.model_id:16} {item.paid_rub:>8g} "
            f"{item.cheapest_model_id or '-':16} {item.cheapest_rub or 0:>8g} "
            f"{item.saving_rub:>8g}"
        )
    for item in report.skipped:
        print(
            f"{item.generation_id:10} {item.model_id:16} {item.paid_rub:>8g} "
            f"-- skipped: {item.unanalysable_reason}"
        )

    print(f"\n{report.summary()}")
    print("compared against today's prices; no generation was re-run and nothing was spent")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mainager", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser(
        "capabilities",
        help="fetch GET /capabilities and write the snapshot to the data directory",
    )
    drift_cmd = subcommands.add_parser(
        "drift",
        help="report disagreements between the model catalog and the price list",
    )
    drift_cmd.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit one JSON object for appending to a drift log; always exits 0",
    )
    overspend = subcommands.add_parser(
        "overspend",
        help="cost the account's generation history against today's cheapest routes",
    )
    overspend.add_argument(
        "--from-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="read history from a JSON file instead of GET /generations",
    )
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ValidationError as exc:
        print("configuration is incomplete; copy .env.example to .env", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2

    try:
        if args.command == "capabilities":
            return asyncio.run(_dump_capabilities(settings))
        if args.command == "drift":
            return asyncio.run(_report_drift(settings, as_json=args.as_json))
        if args.command == "overspend":
            return asyncio.run(_report_overspend(settings, args.from_file))
    except VibeApiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if exc.required_scope:
            granted = ", ".join(exc.granted_scopes) or "none"
            print(
                f"the token needs the '{exc.required_scope}' scope; it holds: {granted}",
                file=sys.stderr,
            )
        if exc.request_id:
            print(f"request_id: {exc.request_id}", file=sys.stderr)
        return 1

    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
