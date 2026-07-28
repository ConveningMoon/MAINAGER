"""Reconcile the model catalog against the published price list.

The two disagree. Measured on 2026-07-28: five models are generatable with no
entry in the price list, six priced identifiers are absent from the catalog, and
two sit in different categories in each. The most expensive tier the platform
will bill for is one of the six missing from the catalog, so a spend ceiling
derived from the catalog alone is too low.

Reporting the drift at startup is the point. It is a provider-side inconsistency
that this project cannot fix, only refuse to be surprised by.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from mainager.preflight.registry import ModelRegistry


class CatalogDrift(BaseModel):
    """Differences between the catalog and the price list."""

    model_config = ConfigDict(frozen=True)

    generatable_without_price: tuple[str, ...] = ()
    priced_without_model: tuple[str, ...] = ()
    category_mismatch: tuple[tuple[str, str, str], ...] = ()

    @property
    def is_clean(self) -> bool:
        return not (
            self.generatable_without_price or self.priced_without_model or self.category_mismatch
        )

    def summary(self) -> str:
        if self.is_clean:
            return "catalog and price list agree"
        return (
            f"{len(self.generatable_without_price)} model(s) with no price, "
            f"{len(self.priced_without_model)} price(s) with no model, "
            f"{len(self.category_mismatch)} category mismatch(es)"
        )


def flatten_prices(payload: dict[str, Any]) -> dict[str, tuple[str, float]]:
    """Map model id to (category, price) from a `/prices` response."""
    prices = payload.get("prices")
    if not isinstance(prices, dict):
        raise ValueError("prices payload has no 'prices' object")
    flat: dict[str, tuple[str, float]] = {}
    for category, entries in prices.items():
        if not isinstance(entries, dict):
            continue
        for model_id, value in entries.items():
            if isinstance(value, int | float):
                flat[str(model_id)] = (str(category), float(value))
    return flat


def reconcile(registry: ModelRegistry, prices: dict[str, tuple[str, float]]) -> CatalogDrift:
    """Compare a built registry against a flattened price list."""
    catalog_ids = set(registry.ids())
    price_ids = set(prices)

    mismatches: list[tuple[str, str, str]] = []
    for model_id in sorted(catalog_ids & price_ids):
        spec = registry.get(model_id)
        category = prices[model_id][0]
        # "other" is a catch-all bucket in the price list, not a real category.
        if category != "other" and category != spec.media_type:
            mismatches.append((model_id, spec.media_type, category))

    return CatalogDrift(
        generatable_without_price=tuple(sorted(catalog_ids - price_ids)),
        priced_without_model=tuple(sorted(price_ids - catalog_ids)),
        category_mismatch=tuple(mismatches),
    )


def unlisted_tiers(
    registry: ModelRegistry, prices: dict[str, tuple[str, float]]
) -> tuple[tuple[tuple[str, str, float], ...], tuple[str, float]]:
    """Priced tiers of known models that the catalog never declares.

    Returns the unlisted tiers as ``(parent, tier, price)`` and the resulting
    ceiling gap as ``(parent, understatement)``, where the understatement is how
    far the dearest tier the catalog knows sits below the dearest the platform
    will bill for.

    Measured case: ``grok-itv`` declares tiers up to 316 RUB, while the price
    list sells ``grok-itv-30`` at 436 — a ceiling built from the catalog alone
    is 38% low.
    """
    unlisted: list[tuple[str, str, float]] = []
    worst: tuple[str, float] = ("", 0.0)

    for parent_id in registry.ids():
        parent = registry.get(parent_id)
        if not parent.tier_prices or parent.is_tier_of is not None:
            continue
        declared_max = max(parent.tier_prices.values())
        actual_max = declared_max
        for priced_id, (_, price) in prices.items():
            if priced_id.startswith(f"{parent_id}-") and priced_id not in registry:
                unlisted.append((parent_id, priced_id, price))
                actual_max = max(actual_max, price)
        gap = actual_max - declared_max
        if gap > worst[1]:
            worst = (parent_id, gap)

    return tuple(sorted(unlisted)), worst
