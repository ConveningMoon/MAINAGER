"""Typed view over the Agent API's error envelope."""

from __future__ import annotations

from typing import Any

import httpx


class VibeApiError(RuntimeError):
    """An error response from the Agent API.

    The API returns a machine-readable ``error`` code alongside the HTTP status.
    Keeping both lets callers branch on the code instead of the status, which is
    what the documented codes are for.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{status_code} {code}: {message}")
        self.status_code = status_code
        self.code = code
        self.message = message
        self.payload = payload or {}

    @property
    def required_scope(self) -> str | None:
        """Scope the token is missing, when the API says so on a 403."""
        required = self.payload.get("required")
        return required if isinstance(required, str) else None

    @property
    def granted_scopes(self) -> list[str]:
        """Scopes the token does hold. Returned alongside ``required`` on a 403."""
        granted = self.payload.get("granted")
        return [s for s in granted if isinstance(s, str)] if isinstance(granted, list) else []

    @property
    def request_id(self) -> str | None:
        """Server-side correlation id. Worth carrying into the audit log."""
        request_id = self.payload.get("request_id")
        return request_id if isinstance(request_id, str) else None

    @classmethod
    def from_response(cls, response: httpx.Response) -> VibeApiError:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        code = str(payload.get("error") or f"http_{response.status_code}")
        message = str(payload.get("message") or response.reason_phrase or "request failed")
        return cls(response.status_code, code, message, payload)
