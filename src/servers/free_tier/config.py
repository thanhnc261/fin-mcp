from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the free tier market data toolbox."""

    yahoo_region: str = Field(
        default="us",
        description="Market region to use for Yahoo Finance screeners.",
    )
    yahoo_timeout: float = Field(
        default=10.0,
        description="HTTP timeout (seconds) applied to Yahoo Finance requests.",
    )
    intraday_max_days: int = Field(
        default=30,
        description="Maximum number of days allowed for intraday price history.",
    )
    history_max_years: int = Field(
        default=10,
        description="Maximum number of years allowed for historical price lookups.",
    )

    model_config = SettingsConfigDict(
        env_prefix="FREE_TIER_",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()  # type: ignore[arg-type]


settings = get_settings()
