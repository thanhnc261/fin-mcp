from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the free tier market data toolbox.

    All settings can be configured via environment variables with the
    FREE_TIER_ prefix. For example:
        FREE_TIER_LOG_LEVEL=DEBUG
        FREE_TIER_TRANSPORT_TYPE=http
        FREE_TIER_HTTP_PORT=8000
    """

    # Logging configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )

    # Transport configuration
    transport_type: Literal["stdio", "http", "sse"] = Field(
        default="stdio",
        description="MCP transport protocol to use.",
    )
    http_host: str = Field(
        default="127.0.0.1",
        description="Host address for HTTP/SSE transport.",
    )
    http_port: int = Field(
        default=8000,
        description="Port number for HTTP/SSE transport.",
        ge=1,
        le=65535,
    )

    # Yahoo Finance configuration
    yahoo_region: str = Field(
        default="us",
        description="Market region to use for Yahoo Finance screeners.",
    )
    yahoo_timeout: float = Field(
        default=10.0,
        description="HTTP timeout (seconds) applied to Yahoo Finance requests.",
        gt=0.0,
        le=60.0,
    )

    # Data limits configuration
    intraday_max_days: int = Field(
        default=30,
        description="Maximum number of days allowed for intraday price history.",
        ge=1,
        le=60,
    )
    history_max_years: int = Field(
        default=10,
        description="Maximum number of years allowed for historical price lookups.",
        ge=1,
        le=20,
    )

    model_config = SettingsConfigDict(
        env_prefix="FREE_TIER_",
        case_sensitive=False,
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return upper_v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()


settings = get_settings()
