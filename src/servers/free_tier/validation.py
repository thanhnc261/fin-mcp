from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal

from .config import settings
from .exceptions import MarketDataValidationError

TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")

CATEGORY_ALIASES: dict[str, Literal["gainers", "losers", "active"]] = {
    "gainers": "gainers",
    "day_gainers": "gainers",
    "top_gainers": "gainers",
    "losers": "losers",
    "day_losers": "losers",
    "top_losers": "losers",
    "active": "active",
    "most_active": "active",
    "top_active": "active",
}

CATEGORY_SCREENERS: dict[Literal["gainers", "losers", "active"], str] = {
    "gainers": "day_gainers",
    "losers": "day_losers",
    "active": "most_actives",
}

VALID_INTERVALS = {
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
}

INTRADAY_INTERVALS = {
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
}


def normalize_ticker(raw_ticker: str) -> str:
    """Validate and normalize a ticker symbol."""

    if not raw_ticker:
        raise MarketDataValidationError("Ticker symbol is required.")

    ticker = raw_ticker.strip().upper()
    if not ticker:
        raise MarketDataValidationError("Ticker symbol cannot be blank.")

    if not TICKER_PATTERN.fullmatch(ticker):
        raise MarketDataValidationError(
            "Ticker symbols must be 1-10 characters using letters, numbers, '.', or '-'."
        )

    return ticker


def resolve_category(raw_category: str) -> Literal["gainers", "losers", "active"]:
    """Normalize the requested market movers category."""

    category_key = raw_category.strip().lower()
    if not category_key:
        raise MarketDataValidationError("Category must be provided.")

    if category_key not in CATEGORY_ALIASES:
        allowed = "', '".join(sorted({*CATEGORY_ALIASES.values()}))
        raise MarketDataValidationError(
            f"Unknown category '{raw_category}'. Choose one of '{allowed}'."
        )

    return CATEGORY_ALIASES[category_key]


def get_screener_name(category: Literal["gainers", "losers", "active"]) -> str:
    return CATEGORY_SCREENERS[category]


def validate_limit(limit: int) -> int:
    """Ensure the requested limit is within safe bounds."""

    if limit is None:
        raise MarketDataValidationError("Limit must be provided.")

    if not 1 <= limit <= 25:
        raise MarketDataValidationError(
            "Limit must be between 1 and 25 to stay within Yahoo Finance constraints."
        )

    return limit


def validate_interval(interval: str) -> str:
    """Ensure the interval is supported by Yahoo Finance."""

    if not interval:
        raise MarketDataValidationError("Interval is required.")

    interval_key = interval.strip().lower()
    if interval_key not in VALID_INTERVALS:
        allowed = "', '".join(sorted(VALID_INTERVALS))
        raise MarketDataValidationError(
            f"Unsupported interval '{interval}'. Choose one of '{allowed}'."
        )

    return interval_key


def ensure_interval_range(start_dt: datetime, end_dt: datetime, interval: str) -> None:
    """Validate the allowed lookback for the given interval."""

    delta = end_dt - start_dt
    if delta <= timedelta(0):
        raise MarketDataValidationError("Start date must be earlier than the end date.")

    max_years = settings.history_max_years
    if delta.days > max_years * 366:
        raise MarketDataValidationError(
            f"Date range is too long. Reduce the window to less than {max_years} years."
        )

    if interval in INTRADAY_INTERVALS and delta.days > settings.intraday_max_days:
        raise MarketDataValidationError(
            "Intraday intervals only support up to "
            f"{settings.intraday_max_days} days of history."
        )


def normalize_date(value: date | datetime | str, field_name: str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except ValueError as exc:
            raise MarketDataValidationError(
                f"{field_name} must be an ISO formatted date (YYYY-MM-DD)."
            ) from exc

    raise MarketDataValidationError(f"Unsupported {field_name} value: {value!r}")


def validate_date_range(
    start_date: date,
    end_date: date,
) -> tuple[datetime, datetime]:
    """Normalize and validate the supplied date range."""

    start = normalize_date(start_date, "start_date")
    end = normalize_date(end_date, "end_date")

    if start > end:
        raise MarketDataValidationError("Start date must be on or before the end date.")

    start_dt = datetime.combine(start, time.min, tzinfo=timezone.utc)
    end_dt = datetime.combine(end, time.max, tzinfo=timezone.utc)

    return start_dt, end_dt
