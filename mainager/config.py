"""Process configuration. Every value comes from the environment."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, loaded from the environment or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_token: SecretStr = Field(alias="VIBE_API_TOKEN")
    api_base_url: str = Field(
        default="https://lk.vibemarketolog.ru/api/agent",
        alias="VIBE_API_BASE_URL",
    )
    http_timeout_s: float = Field(default=30.0, alias="VIBE_HTTP_TIMEOUT_S")

    plan_ceiling_rub: float = Field(default=50.0, alias="MAINAGER_PLAN_CEILING_RUB")
    daily_ceiling_rub: float = Field(default=100.0, alias="MAINAGER_DAILY_CEILING_RUB")

    data_dir: Path = Field(default=Path("data"), alias="MAINAGER_DATA_DIR")

    log_format: Literal["text", "json"] = Field(default="text", alias="MAINAGER_LOG_FORMAT")
    log_level: str = Field(default="INFO", alias="MAINAGER_LOG_LEVEL")

    @property
    def auth_header(self) -> dict[str, str]:
        """Authorization header for the Agent API."""
        return {"Authorization": f"Bearer {self.api_token.get_secret_value()}"}


def load_settings() -> Settings:
    """Read settings from the environment.

    Raises ``pydantic.ValidationError`` when the token is absent, which is the
    correct failure: nothing in this project works without it.
    """
    return Settings()  # type: ignore[call-arg]
