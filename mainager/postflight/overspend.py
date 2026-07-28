"""What a past generation would cost if it were routed today.

Billing is per generation and the history endpoint is free, so an account can be
asked a question the platform never answers on its own: of everything already
spent, how much was avoidable?

Two honesty constraints are built in rather than papered over.

First, a history record does not always carry the parameters the generation was
made with, and for some models the price depends on them — duration for the grok
tiers, prompt length for the per-1000-character voices. Where the request cannot
be reconstructed the entry is reported as unanalysable and excluded from the
total. It is never guessed.

Second, the comparison is against *today's* prices. A generation made last month
is compared with what the same output would cost now, which is the actionable
number but not a historical audit. The report says so.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from mainager.preflight.compiler import Intent
from mainager.preflight.pricing import Pricer
from mainager.preflight.registry import ModelRegistry
from mainager.preflight.router import NoViableModelError, route

_MEDIA_TYPES: frozenset[str] = frozenset({"image", "video", "voice", "music", "text"})

#: Parameters whose absence makes a price non-reconstructible for that model.
_PRICE_BEARING = ("duration", "prompt")


class PastGeneration(BaseModel):
    """One entry from the account's generation history."""

    model_config = ConfigDict(frozen=True)

    generation_id: str
    media_type: str
    model_id: str
    cost_rub: float
    status: str = "unknown"
    refunded: bool = False
    params: dict[str, Any] = {}


class ItemVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    generation_id: str
    model_id: str
    paid_rub: float
    cheapest_model_id: str | None = None
    cheapest_rub: float | None = None
    unanalysable_reason: str | None = None

    @property
    def saving_rub(self) -> float:
        if self.cheapest_rub is None:
            return 0.0
        return max(0.0, self.paid_rub - self.cheapest_rub)


class OverspendReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: tuple[ItemVerdict, ...]

    @property
    def analysed(self) -> tuple[ItemVerdict, ...]:
        return tuple(i for i in self.items if i.unanalysable_reason is None)

    @property
    def skipped(self) -> tuple[ItemVerdict, ...]:
        return tuple(i for i in self.items if i.unanalysable_reason is not None)

    @property
    def paid_rub(self) -> float:
        return sum(i.paid_rub for i in self.analysed)

    @property
    def optimal_rub(self) -> float:
        return sum(i.cheapest_rub or 0.0 for i in self.analysed)

    @property
    def overspend_rub(self) -> float:
        return sum(i.saving_rub for i in self.analysed)

    def summary(self) -> str:
        if not self.analysed:
            return f"nothing analysable in {len(self.items)} generation(s)"
        share = self.overspend_rub / self.paid_rub * 100 if self.paid_rub else 0.0
        line = (
            f"{len(self.analysed)} generation(s): paid {self.paid_rub:g} RUB, "
            f"cheapest equivalent today {self.optimal_rub:g} RUB, "
            f"avoidable {self.overspend_rub:g} RUB ({share:.0f}%)"
        )
        if self.skipped:
            line += f"; {len(self.skipped)} not analysable"
        return line


def parse_generations(payload: dict[str, Any]) -> tuple[PastGeneration, ...]:
    """Read `GET /generations`, tolerating fields the endpoint may not send."""
    raw = payload.get("generations")
    if not isinstance(raw, list):
        return ()

    parsed: list[PastGeneration] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        cost = entry.get("cost", entry.get("cost_rub"))
        params = entry.get("params")
        parsed.append(
            PastGeneration(
                generation_id=str(entry.get("id") or entry.get("generation_id") or index),
                media_type=str(entry.get("type") or ""),
                model_id=str(entry.get("model") or ""),
                cost_rub=float(cost) if isinstance(cost, int | float) else 0.0,
                status=str(entry.get("status") or "unknown"),
                refunded=bool(entry.get("refunded")),
                params=params if isinstance(params, dict) else {},
            )
        )
    return tuple(parsed)


def reconstruct_intent(
    generation: PastGeneration, registry: ModelRegistry
) -> tuple[Intent | None, str | None]:
    """Rebuild the intent behind a past generation, or explain why it cannot be."""
    if generation.media_type not in _MEDIA_TYPES:
        return None, f"unknown media type {generation.media_type!r}"

    spec = registry.find(generation.model_id)
    if spec is None:
        return None, f"model {generation.model_id!r} is no longer in the catalog"

    params = generation.params
    prompt = params.get("prompt")
    if not isinstance(prompt, str):
        return None, "history record carries no prompt, so length-priced models cannot be compared"

    for name in _PRICE_BEARING:
        if spec.accepts(name) and name not in params and name != "prompt":
            return None, f"history record carries no {name!r}, which affects this model's price"

    duration = params.get("duration")
    source = _first_media_url(params)

    intent = Intent(
        media_type=generation.media_type,  # type: ignore[arg-type]
        prompt=prompt,
        source_image_url=source,
        duration_s=int(duration) if isinstance(duration, int) else None,
        aspect_ratio=params.get("aspect_ratio")
        if isinstance(params.get("aspect_ratio"), str)
        else None,
        resolution=params.get("resolution") if isinstance(params.get("resolution"), str) else None,
    )
    return intent, None


def _first_media_url(params: dict[str, Any]) -> str | None:
    for key in ("image_input", "image_urls", "first_frame_url", "character_image_url", "image_url"):
        value = params.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value and isinstance(value[0], str):
            return str(value[0])
    return None


async def overspend_report(
    generations: tuple[PastGeneration, ...],
    registry: ModelRegistry,
    pricer: Pricer,
) -> OverspendReport:
    """Re-route every analysable generation and total what could have been saved."""
    items: list[ItemVerdict] = []

    for generation in generations:
        if generation.refunded:
            items.append(
                ItemVerdict(
                    generation_id=generation.generation_id,
                    model_id=generation.model_id,
                    paid_rub=generation.cost_rub,
                    unanalysable_reason="refunded, so nothing was actually paid",
                )
            )
            continue

        intent, reason = reconstruct_intent(generation, registry)
        if intent is None:
            items.append(
                ItemVerdict(
                    generation_id=generation.generation_id,
                    model_id=generation.model_id,
                    paid_rub=generation.cost_rub,
                    unanalysable_reason=reason,
                )
            )
            continue

        try:
            decision = await route(intent, registry, pricer)
        except NoViableModelError:
            items.append(
                ItemVerdict(
                    generation_id=generation.generation_id,
                    model_id=generation.model_id,
                    paid_rub=generation.cost_rub,
                    unanalysable_reason="no model can serve this intent today",
                )
            )
            continue

        items.append(
            ItemVerdict(
                generation_id=generation.generation_id,
                model_id=generation.model_id,
                paid_rub=generation.cost_rub,
                cheapest_model_id=decision.spec.model_id,
                cheapest_rub=decision.cost_rub,
            )
        )

    return OverspendReport(items=tuple(items))
