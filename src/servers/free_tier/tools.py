"""Market data tool implementations for Yahoo Finance integration.

This module provides three core tools for retrieving market data:
- get_market_movers: Top gainers, losers, and most active stocks
- get_ticker_data: Real-time quote snapshot for a single symbol
- get_price_history: Historical OHLCV candles for technical analysis

All tools handle data validation, error handling, and type coercion to ensure
reliable data quality even when Yahoo Finance returns inconsistent formats.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Any, Literal

from fastmcp import Context
from pydantic import Field
from yfinance import Ticker, screen

# Note: yahoo_timeout from settings could be applied via a custom curl_cffi session
# passed to Ticker(session=...), but yfinance's internal session management is complex.
# For now, we rely on yfinance's default timeout behavior.
# TODO: Implement custom session with timeout configuration
from .exceptions import MarketDataProviderError
from .schemas import (
    MarketMover,
    MarketMoversResponse,
    PriceBar,
    PriceHistoryResponse,
    TickerQuote,
)
from .validation import (
    ensure_interval_range,
    get_screener_name,
    normalize_ticker,
    resolve_category,
    validate_date_range,
    validate_interval,
    validate_limit,
)


def _safe_datetime_from_epoch(epoch: int | float | None, tz_name: str | None) -> datetime | None:
    """Convert Unix epoch timestamp to timezone-aware datetime.

    Yahoo Finance returns timestamps as Unix epochs along with exchange timezone names.
    This helper safely converts them to proper datetime objects with timezone info.

    Args:
        epoch: Unix timestamp (seconds since 1970-01-01 UTC), may be None
        tz_name: IANA timezone name (e.g., "America/New_York"), may be None

    Returns:
        Timezone-aware datetime in the exchange's timezone, or None if epoch is missing.
        Falls back to UTC if timezone conversion fails.
    """
    if not epoch:
        return None

    # Start with UTC timezone
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)

    # Attempt to localize to the exchange's timezone
    if tz_name:
        try:
            from zoneinfo import ZoneInfo

            return dt.astimezone(ZoneInfo(tz_name))
        except Exception:
            # If timezone is invalid or unavailable, keep UTC
            # This can happen with exotic exchanges or stale data
            pass
    return dt


def _coerce_float(value: Any) -> float | None:
    """Safely convert a value to float, handling NaN and invalid types.

    Yahoo Finance sometimes returns NaN, inf, or non-numeric strings for
    missing or invalid price data. This helper ensures clean float values.

    Args:
        value: Any value that might be numeric (float, int, string, None)

    Returns:
        float value if valid, None otherwise (including NaN)
    """
    if value is None:
        return None
    try:
        coerced = float(value)
        # NaN is a valid float but meaningless for financial data
        if math.isnan(coerced):
            return None
        return coerced
    except (TypeError, ValueError):
        # Invalid type or unparseable string
        return None


def _coerce_int(value: Any) -> int | None:
    """Safely convert a value to int, handling invalid types.

    Args:
        value: Any value that might be numeric (int, float, string, None)

    Returns:
        int value if valid, None otherwise
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def get_market_movers(
    ctx: Context,
    category: Annotated[
        Literal["gainers", "losers", "active"],
        Field(
            description="Market movers category to query.",
            examples=["gainers", "losers", "active"],
        ),
    ] = "gainers",
    limit: Annotated[
        int,
        Field(
            ge=1,
            le=25,
            description="Maximum number of symbols to return (1-25).",
            examples=[5, 10],
        ),
    ] = 10,
) -> MarketMoversResponse:
    """Retrieve the leading gainers, losers, or most active stocks from Yahoo Finance.

    This tool queries Yahoo Finance's predefined screeners to find stocks with the
    largest price movements or trading volumes. Results are filtered to the US market
    and limited to 25 items to stay within API rate limits and Claude's context window.

    Args:
        ctx: FastMCP context for logging and progress updates
        category: Which type of movers to retrieve (gainers, losers, active)
        limit: Maximum number of stocks to return (1-25)

    Returns:
        MarketMoversResponse containing the list of matching stocks with price,
        volume, and change data
    """
    # Validate and normalize user input
    normalized_category = resolve_category(category)
    validated_limit = validate_limit(limit)
    screener_id = get_screener_name(normalized_category)

    await ctx.info(
        f"Fetching {normalized_category} market movers: screener={screener_id}, limit={validated_limit}"
    )

    try:
        # Yahoo Finance screener API call
        response = screen(screener_id, count=validated_limit, start=0)
    except Exception as exc:
        await ctx.error(f"Yahoo Finance screen request failed: category={screener_id}, error={exc}")
        raise MarketDataProviderError(
            "Unable to retrieve market movers from Yahoo Finance right now. "
            "Please try again in a moment."
        ) from exc

    # Extract quotes array from response
    quotes = response.get("quotes") if isinstance(response, dict) else None
    if not quotes:
        raise MarketDataProviderError(
            "Yahoo Finance returned no results for the requested market movers."
        )

    # Parse each quote into a structured MarketMover object
    movers: list[MarketMover] = []
    for quote in quotes[:validated_limit]:
        if not isinstance(quote, dict):
            continue

        # Yahoo Finance field names are inconsistent; use coercion helpers
        movers.append(
            MarketMover(
                symbol=quote.get("symbol", ""),
                name=quote.get("shortName") or quote.get("longName"),
                price=_coerce_float(quote.get("regularMarketPrice")),
                change=_coerce_float(quote.get("regularMarketChange")),
                change_percent=_coerce_float(quote.get("regularMarketChangePercent")),
                volume=_coerce_int(quote.get("regularMarketVolume")),
                currency=quote.get("currency"),
                market_state=quote.get("marketState"),
                timestamp=_safe_datetime_from_epoch(
                    quote.get("regularMarketTime"), quote.get("exchangeTimezoneName")
                ),
            )
        )

    await ctx.info(f"Successfully retrieved {len(movers)} {normalized_category}")
    return MarketMoversResponse(
        category=normalized_category,
        as_of=datetime.now(tz=timezone.utc),
        items=movers,
    )


async def get_ticker_data(
    ctx: Context,
    ticker: Annotated[
        str,
        Field(
            description="Ticker symbol to lookup (letters, numbers, '.', or '-').",
            examples=["AAPL", "MSFT"],
        ),
    ],
) -> TickerQuote:
    """Fetch the latest quote snapshot for a single ticker using Yahoo Finance.

    This tool retrieves real-time market data including price, volume, moving averages,
    and fundamental metrics for a specific ticker symbol. Data is sourced from Yahoo
    Finance's fast_info API for quick responses.

    Args:
        ctx: FastMCP context for logging and progress updates
        ticker: Stock symbol to lookup (e.g., "AAPL", "MSFT", "BRK-B")

    Returns:
        TickerQuote with current price, changes, volumes, and technical indicators
    """
    # Normalize ticker to uppercase and validate format
    normalized_ticker = normalize_ticker(ticker)
    await ctx.info(f"Fetching ticker snapshot: ticker={normalized_ticker}")

    try:
        # Yahoo Finance Ticker client with configurable timeout
        ticker_client = Ticker(normalized_ticker, session=None)
        # fast_info provides cached, quick-access data
        fast_info = dict(ticker_client.fast_info.items())
        # info provides comprehensive metadata (slower)
        info = ticker_client.info
    except Exception as exc:
        await ctx.error(f"Failed to load ticker info: ticker={normalized_ticker}, error={exc}")
        raise MarketDataProviderError(
            f"Yahoo Finance could not provide data for '{normalized_ticker}'. "
            "The symbol may be invalid or temporarily unavailable."
        ) from exc

    # Price is required; fail fast if missing
    price = _coerce_float(fast_info.get("lastPrice"))
    if price is None:
        raise MarketDataProviderError(
            f"Yahoo Finance did not return a last trade price for '{normalized_ticker}'."
        )

    # Calculate change metrics from previous close
    # Yahoo Finance may provide previousClose under different keys
    previous_close = _coerce_float(
        fast_info.get("previousClose") or fast_info.get("regularMarketPreviousClose")
    )
    change = price - previous_close if previous_close is not None else None
    change_percent = (
        (change / previous_close) * 100 if change is not None and previous_close else None
    )

    # Extract timestamp and timezone for accurate quote time
    timestamp = info.get("regularMarketTime")
    tz_name = info.get("exchangeTimezoneName")

    change_str = f"{change:.2f}" if change is not None else "N/A"
    await ctx.info(f"Successfully fetched {normalized_ticker}: price={price}, change={change_str}")

    return TickerQuote(
        symbol=normalized_ticker,
        name=info.get("shortName") or info.get("longName"),
        price=price,
        change=change,
        change_percent=change_percent,
        previous_close=previous_close,
        open=_coerce_float(fast_info.get("open")),
        day_high=_coerce_float(fast_info.get("dayHigh")),
        day_low=_coerce_float(fast_info.get("dayLow")),
        volume=_coerce_int(fast_info.get("lastVolume")),
        average_volume_10d=_coerce_int(fast_info.get("tenDayAverageVolume")),
        average_volume_3m=_coerce_int(fast_info.get("threeMonthAverageVolume")),
        market_cap=_coerce_float(fast_info.get("marketCap")),
        currency=fast_info.get("currency"),
        exchange=fast_info.get("exchange"),
        market_state=info.get("marketState"),
        fifty_day_average=_coerce_float(fast_info.get("fiftyDayAverage")),
        two_hundred_day_average=_coerce_float(fast_info.get("twoHundredDayAverage")),
        year_high=_coerce_float(fast_info.get("yearHigh")),
        year_low=_coerce_float(fast_info.get("yearLow")),
        timestamp=_safe_datetime_from_epoch(timestamp, tz_name),
    )


# Default lookback window when start_date is not specified
DEFAULT_HISTORY_DAYS = 30


async def get_price_history(
    ctx: Context,
    ticker: Annotated[
        str,
        Field(
            description="Ticker symbol to query (letters, numbers, '.', or '-').",
            examples=["AAPL", "SPY"],
        ),
    ],
    start_date: Annotated[
        date | None,
        Field(
            description="Start date for the price history (YYYY-MM-DD). "
            "Defaults to 30 days ago.",
        ),
    ] = None,
    end_date: Annotated[
        date | None,
        Field(description="Inclusive end date for the price history. Defaults to today."),
    ] = None,
    interval: Annotated[
        str,
        Field(
            description="Sampling interval supported by Yahoo Finance.",
            examples=["1d", "1h", "1wk"],
        ),
    ] = "1d",
) -> PriceHistoryResponse:
    """Retrieve historical OHLCV candles for the requested ticker and date range.

    This tool fetches historical price data (Open, High, Low, Close, Volume) along
    with corporate actions (dividends, stock splits) for technical analysis and
    backtesting. Supports intraday intervals (1m, 5m, 1h) with limited history
    and daily+ intervals for longer lookbacks.

    Args:
        ctx: FastMCP context for logging and progress updates
        ticker: Stock symbol to query (e.g., "AAPL", "SPY", "^GSPC")
        start_date: Beginning of date range (defaults to 30 days ago)
        end_date: End of date range (defaults to today)
        interval: Sampling frequency (1m, 5m, 15m, 1h, 1d, 1wk, 1mo, etc.)

    Returns:
        PriceHistoryResponse containing the list of price bars and metadata

    Note:
        Intraday intervals are limited to 30 days of history due to Yahoo Finance
        API constraints. Daily intervals can go back up to 10 years.
    """
    # Validate and normalize ticker symbol
    normalized_ticker = normalize_ticker(ticker)
    interval_key = validate_interval(interval)

    # Apply default date range if not specified
    today = date.today()
    start = start_date or (today - timedelta(days=DEFAULT_HISTORY_DAYS))
    end = end_date or today

    # Validate date range and interval compatibility
    start_dt, end_dt = validate_date_range(start, end)
    ensure_interval_range(start_dt, end_dt, interval_key)

    await ctx.info(
        f"Fetching price history: ticker={normalized_ticker}, interval={interval_key}, "
        f"start={start_dt.date()}, end={end_dt.date()}"
    )

    # Yahoo Finance uses exclusive end dates, so add 1 second to include the final day
    yf_end = end_dt + timedelta(seconds=1)

    try:
        ticker_client = Ticker(normalized_ticker, session=None)
        # Fetch historical data with corporate actions
        history = ticker_client.history(
            start=start_dt,
            end=yf_end,
            interval=interval_key,
            auto_adjust=False,  # Keep raw prices (not adjusted for splits/dividends)
            actions=True,  # Include dividend and split events
        )
        fast_info = dict(ticker_client.fast_info.items())
    except Exception as exc:
        await ctx.error(f"Failed to fetch price history: ticker={normalized_ticker}, error={exc}")
        raise MarketDataProviderError(
            f"Unable to load price history for '{normalized_ticker}'. "
            "Please try again with a smaller window or later."
        ) from exc

    # Check for empty result (invalid ticker or no trading during period)
    if history.empty:
        raise MarketDataProviderError(
            "Yahoo Finance returned an empty price series. "
            "Try a broader date range or a different interval."
        )

    # Convert pandas DataFrame rows to structured PriceBar objects
    points: list[PriceBar] = []
    for ts, row in history.iterrows():
        # Ensure timezone-aware timestamps
        timestamp = ts.to_pydatetime()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        points.append(
            PriceBar(
                timestamp=timestamp,
                # Fallback to 0.0 for OHLC if missing (rare but possible for halted stocks)
                open=_coerce_float(row.get("Open")) or 0.0,
                high=_coerce_float(row.get("High")) or 0.0,
                low=_coerce_float(row.get("Low")) or 0.0,
                close=_coerce_float(row.get("Close")) or 0.0,
                volume=_coerce_float(row.get("Volume")),
                dividends=_coerce_float(row.get("Dividends")),
                stock_splits=_coerce_float(row.get("Stock Splits")),
            )
        )

    await ctx.info(f"Successfully retrieved {len(points)} price bars for {normalized_ticker}")

    return PriceHistoryResponse(
        ticker=normalized_ticker,
        interval=interval_key,
        start=start_dt,
        end=end_dt,
        points=points,
        currency=fast_info.get("currency"),
    )
