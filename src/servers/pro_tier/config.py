"""Configuration settings for the Pro Tier MCP server.

All settings are loaded from environment variables with the PRO_TIER_ prefix.
Sensitive API keys must be provided for sentiment and analytics features.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProTierSettings(BaseSettings):
    """Pro Tier server configuration with validation and defaults."""

    model_config = SettingsConfigDict(
        env_prefix="PRO_TIER_",
        case_sensitive=False,
        extra="ignore",
    )

    # Logging configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging verbosity level",
    )

    # Transport configuration
    transport_type: Literal["stdio", "http", "sse"] = Field(
        default="stdio",
        description="MCP transport protocol",
    )
    http_host: str = Field(
        default="127.0.0.1",
        description="Host address for HTTP/SSE transport",
    )
    http_port: int = Field(
        default=8001,
        description="Port number for HTTP/SSE transport",
    )

    # Redis cache configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for caching",
    )
    cache_ttl_sentiment: int = Field(
        default=300,  # 5 minutes
        description="Cache TTL for sentiment data in seconds",
    )
    cache_ttl_analytics: int = Field(
        default=900,  # 15 minutes
        description="Cache TTL for analytics data in seconds",
    )

    # External API configuration
    news_api_key: str = Field(
        default="",
        description="NewsAPI.org API key for news sentiment analysis",
    )
    alpha_vantage_key: str = Field(
        default="",
        description="Alpha Vantage API key for additional market data",
    )

    # Sentiment analysis configuration
    sentiment_lookback_days: int = Field(
        default=7,
        description="Default lookback period for sentiment trends",
    )
    news_max_articles: int = Field(
        default=100,
        description="Maximum news articles to analyze per query",
    )

    # Trading analytics configuration
    backtest_max_years: int = Field(
        default=5,
        description="Maximum years of historical data for backtesting",
    )
    portfolio_max_positions: int = Field(
        default=100,
        description="Maximum positions in portfolio analytics",
    )


# Global settings instance
settings = ProTierSettings()
