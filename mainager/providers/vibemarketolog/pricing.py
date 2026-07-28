"""Dry-run pricing against the Vibe Agent API."""

from __future__ import annotations

from typing import Any

import httpx

from mainager.config import Settings
from mainager.preflight.pricing import Estimate
from mainager.providers.vibemarketolog.errors import VibeApiError

ESTIMATE_PATH = "/generate/estimate"


def _strs(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    return tuple(str(v) for v in value) if isinstance(value, list) else ()


def parse_estimate(payload: dict[str, Any], fallback_model: str) -> Estimate:
    cost = payload.get("estimated_cost_rub")
    balance = payload.get("balance")
    daily = payload.get("daily_spend")
    return Estimate(
        model_id=str(payload.get("model") or fallback_model),
        cost_rub=float(cost) if isinstance(cost, int | float) else 0.0,
        provider_valid=bool(payload.get("valid")),
        rejected=_strs(payload, "rejected"),
        warnings=_strs(payload, "warnings"),
        balance_after=(
            float(balance["after"])
            if isinstance(balance, dict) and isinstance(balance.get("after"), int | float)
            else None
        ),
        within_daily_limit=(
            bool(daily["within_limit"])
            if isinstance(daily, dict) and "within_limit" in daily
            else None
        ),
        raw=payload,
    )


class VibePricer:
    """Prices a body through `POST /generate/estimate`, which never debits."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    @classmethod
    def client_for(cls, settings: Settings) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=settings.api_base_url,
            headers=settings.auth_header,
            timeout=settings.http_timeout_s,
        )

    async def estimate(self, body: dict[str, Any]) -> Estimate:
        response = await self._client.post(ESTIMATE_PATH, json=body)
        if response.is_error:
            raise VibeApiError.from_response(response)
        payload = response.json()
        if not isinstance(payload, dict):
            raise VibeApiError(
                response.status_code, "unexpected_payload", "estimate did not return an object"
            )
        return parse_estimate(payload, str(body.get("model", "")))
