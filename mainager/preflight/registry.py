"""Typed model registry, generated from the provider catalog.

Nothing about which parameter a model expects is written down here. The catalog
already carries `required`, `optional`, `enums`, `limits`, `tiers` and
`tier_prices` per model, so the registry is a parse of that, not a transcription
of the prose documentation. When the provider changes a field, this file does not
need to.

One thing the catalog does not model explicitly: a tier (`grok-itv-10`) is a
callable model id in its own right, not a parameter on its parent. That was
established by dry-run — `model: "grok-itv-10"` prices at 196 RUB while
`model: "grok-itv"` prices at 36 — so tiers are expanded into first-class
entries here.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelSpec(BaseModel):
    """One callable model, as the catalog describes it."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    model_id: str
    media_type: str
    description: str = ""
    hint: str | None = None

    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    enums: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    limits: dict[str, Any] = Field(default_factory=dict)

    price: float | None = None
    per_second: float | None = None
    price_formula: str | None = None

    tiers: tuple[str, ...] = ()
    tier_prices: dict[str, float] = Field(default_factory=dict)

    #: True when this entry was derived from a parent's ``tiers`` list.
    is_tier_of: str | None = None

    @property
    def accepted_params(self) -> frozenset[str]:
        return frozenset(self.required) | frozenset(self.optional)

    def accepts(self, param: str) -> bool:
        return param in self.accepted_params

    def allowed_values(self, param: str) -> tuple[str, ...] | None:
        """Enum for a parameter, or None when the catalog does not constrain it."""
        return self.enums.get(param)

    def limit(self, name: str) -> Any:
        return self.limits.get(name)

    @property
    def mutually_exclusive(self) -> tuple[frozenset[str], ...]:
        """Parameter groups the catalog says cannot be combined."""
        raw = self.limits.get("mutually_exclusive")
        if not isinstance(raw, list):
            return ()
        return tuple(frozenset(str(p) for p in group) for group in raw if isinstance(group, list))


def _spec_from_entry(model_id: str, media_type: str, entry: dict[str, Any]) -> ModelSpec:
    enums_raw = entry.get("enums")
    enums: dict[str, tuple[str, ...]] = {}
    if isinstance(enums_raw, dict):
        for key, values in enums_raw.items():
            if isinstance(values, list):
                enums[str(key)] = tuple(str(v) for v in values)

    tier_prices_raw = entry.get("tier_prices")
    tier_prices: dict[str, float] = {}
    if isinstance(tier_prices_raw, dict):
        for key, value in tier_prices_raw.items():
            if isinstance(value, int | float):
                tier_prices[str(key)] = float(value)

    def _strs(key: str) -> tuple[str, ...]:
        value = entry.get(key)
        return tuple(str(v) for v in value) if isinstance(value, list) else ()

    limits = entry.get("limits")
    price = entry.get("price")

    return ModelSpec(
        model_id=model_id,
        media_type=media_type,
        description=str(entry.get("description") or ""),
        hint=str(entry["hint"]) if entry.get("hint") else None,
        required=_strs("required"),
        optional=_strs("optional"),
        enums=enums,
        limits=limits if isinstance(limits, dict) else {},
        price=float(price) if isinstance(price, int | float) else None,
        per_second=(
            float(entry["per_second"]) if isinstance(entry.get("per_second"), int | float) else None
        ),
        price_formula=str(entry["price_formula"]) if entry.get("price_formula") else None,
        tiers=_strs("tiers"),
        tier_prices=tier_prices,
    )


class ModelRegistry:
    """All callable models, keyed by the id that goes into a request body."""

    def __init__(self, specs: dict[str, ModelSpec]) -> None:
        self._specs = specs

    @classmethod
    def from_capabilities(cls, payload: dict[str, Any]) -> ModelRegistry:
        models = payload.get("models")
        if not isinstance(models, dict):
            raise ValueError("capabilities payload has no 'models' object")

        specs: dict[str, ModelSpec] = {}
        for media_type, entries in models.items():
            if not isinstance(entries, dict):
                continue
            for model_id, entry in entries.items():
                if not isinstance(entry, dict):
                    continue
                parent = _spec_from_entry(str(model_id), str(media_type), entry)
                specs[parent.model_id] = parent
                specs.update(_expand_tiers(parent))
        return cls(specs)

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._specs[model_id]
        except KeyError:
            raise UnknownModelError(model_id) from None

    def find(self, model_id: str) -> ModelSpec | None:
        return self._specs.get(model_id)

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def by_type(self, media_type: str) -> tuple[ModelSpec, ...]:
        return tuple(
            spec for _, spec in sorted(self._specs.items()) if spec.media_type == media_type
        )


def _expand_tiers(parent: ModelSpec) -> dict[str, ModelSpec]:
    """Promote each declared tier to its own callable entry.

    A tier inherits the parent's parameter contract and carries its own price.

    ``tiers`` and ``tier_prices`` are unioned rather than trusting either alone.
    The catalog has already been seen to disagree with itself about which tiers
    exist — ``getting_started`` advertises ``grok-itv-10|20|30`` while
    ``models.video["grok-itv"].tiers`` lists only the first two — so taking one
    list as authoritative is not safe. Tiers that appear in neither field are a
    separate problem, handled by reconciling against the price list.
    """
    names = dict.fromkeys([*parent.tiers, *parent.tier_prices])
    expanded: dict[str, ModelSpec] = {}
    for name in names:
        if name == parent.model_id:
            continue
        expanded[name] = parent.model_copy(
            update={
                "model_id": name,
                "price": parent.tier_prices.get(name),
                "tiers": (),
                "tier_prices": {},
                "is_tier_of": parent.model_id,
            }
        )
    return expanded


class UnknownModelError(KeyError):
    """Raised when a model id is not in the catalog."""

    def __init__(self, model_id: str) -> None:
        super().__init__(model_id)
        self.model_id = model_id

    def __str__(self) -> str:
        return f"model {self.model_id!r} is not in the catalog"
