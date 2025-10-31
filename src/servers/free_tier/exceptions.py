from __future__ import annotations

from fastmcp.exceptions import ToolError, ValidationError


class MarketDataError(ToolError):
    """Base exception for Free Tier market data tools."""


class MarketDataValidationError(ValidationError, MarketDataError):
    """Raised when user input fails validation."""


class MarketDataProviderError(MarketDataError):
    """Raised when the upstream market data provider returns an error."""
