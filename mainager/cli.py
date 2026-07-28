"""Command line entry point."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from pydantic import ValidationError

from mainager.config import Settings, load_settings
from mainager.providers.vibemarketolog.capabilities import fetch_capabilities, write_snapshot
from mainager.providers.vibemarketolog.errors import VibeApiError


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mainager", description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser(
        "capabilities",
        help="fetch GET /capabilities and write the snapshot to the data directory",
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
    except VibeApiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if exc.required_scope:
            print(f"the token is missing the '{exc.required_scope}' scope", file=sys.stderr)
        return 1

    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
