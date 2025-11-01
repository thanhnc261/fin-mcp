"""Custom exception types for Pro Tier server operations.

These exceptions provide clear error messages and help distinguish between
data provider issues, validation errors, and internal server problems.
"""

from __future__ import annotations


class ProTierError(Exception):
    """Base exception for all Pro Tier server errors."""

    pass


class SentimentProviderError(ProTierError):
    """Raised when external sentiment data providers fail or return invalid data."""

    pass


class AnalyticsError(ProTierError):
    """Raised when analytics calculations fail or produce invalid results."""

    pass


class CacheError(ProTierError):
    """Raised when Redis cache operations fail."""

    pass


class APIKeyMissingError(ProTierError):
    """Raised when required API keys are not configured."""

    pass
