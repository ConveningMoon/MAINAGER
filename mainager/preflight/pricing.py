"""The pricing port.

Prices are never computed locally. The catalog's `price` field is a
default-configuration figure, the published price list is incomplete, and for
several models the cost depends on duration in ways neither source expresses.
The provider's dry-run endpoint is the only complete source, and it is free, so
the plane asks it rather than guessing.

The protocol keeps the planes provider-agnostic; the implementation lives with
the adapter.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class Estimate(BaseModel):
    """What a dry-run says about one request body."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    model_id: str
    cost_rub: float
    #: The provider's own verdict. Treated as advisory: it has been observed to
    #: report true for a body carrying a parameter the provider will drop, and
    #: false for a well-formed body that merely exceeds the spend limit.
    provider_valid: bool
    rejected: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    balance_after: float | None = None
    within_daily_limit: bool | None = None
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)

    @property
    def body_is_clean(self) -> bool:
        """True when the provider kept every parameter that was sent.

        This is the question `provider_valid` should answer and does not, so it
        is derived from `rejected` instead.
        """
        return not self.rejected

    @property
    def affordable(self) -> bool:
        """False when the provider warned about balance or the daily ceiling."""
        if self.within_daily_limit is False:
            return False
        return not any(
            "insufficient_balance" in w or "daily_spend_limit" in w for w in self.warnings
        )


class Pricer(Protocol):
    """Anything that can price a request body without spending."""

    async def estimate(self, body: dict[str, Any]) -> Estimate: ...
