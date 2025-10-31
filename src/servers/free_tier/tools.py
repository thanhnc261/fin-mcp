from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Any, Literal

from pydantic import Field
from yfinance import Ticker, screen

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

logger = logging.getLogger(__name__)


def _safe_datetime_from_epoch(epoch: int | float | None, tz_name: str | None) -> datetime | None:
    if not epoch:
        return None

    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    if tz_name:
        try:
            from zoneinfo import ZoneInfo

            return dt.astimezone(ZoneInfo(tz_name))
        except Exception:
            logger.debug("Unable to apply timezone %s, falling back to UTC", tz_name)
    return dt


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        coerced = float(value)
        if math.isnan(coerced):
            return None
        return coerced
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_market_movers(
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
    """
    Retrieve the leading gainers, losers, or most active stocks from Yahoo Finance.
    """

    normalized_category = resolve_category(category)
    validated_limit = validate_limit(limit)
    screener_id = get_screener_name(normalized_category)

    logger.info(
        "Fetching market movers: category=%s screener=%s limit=%d",
        normalized_category,
        screener_id,
        validated_limit,
    )

    try:
        response = screen(screener_id, count=validated_limit, start=0)
    except Exception as exc:
        logger.exception("Yahoo Finance screen request failed: category=%s", screener_id)
        raise MarketDataProviderError(
            "Unable to retrieve market movers from Yahoo Finance right now. "
            "Please try again in a moment."
        ) from exc

    quotes = response.get("quotes") if isinstance(response, dict) else None
    if not quotes:
        raise MarketDataProviderError(
            "Yahoo Finance returned no results for the requested market movers."
        )

    movers: list[MarketMover] = []
    for quote in quotes[:validated_limit]:
        if not isinstance(quote, dict):
            continue

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

    return MarketMoversResponse(
        category=normalized_category,
        as_of=datetime.now(tz=timezone.utc),
        items=movers,
    )


def get_ticker_data(
    ticker: Annotated[
        str,
        Field(
            description="Ticker symbol to lookup (letters, numbers, '.', or '-').",
            examples=["AAPL", "MSFT"],
        ),
    ],
) -> TickerQuote:
    """
    Fetch the latest quote snapshot for a single ticker using Yahoo Finance.
    """

    normalized_ticker = normalize_ticker(ticker)
    logger.info("Fetching ticker snapshot: ticker=%s", normalized_ticker)

    try:
        ticker_client = Ticker(normalized_ticker)
        fast_info = dict(ticker_client.fast_info.items())
        info = ticker_client.info
    except Exception as exc:
        logger.exception("Failed to load ticker info: ticker=%s", normalized_ticker)
        raise MarketDataProviderError(
            f"Yahoo Finance could not provide data for '{normalized_ticker}'. "
            "The symbol may be invalid or temporarily unavailable."
        ) from exc

    price = _coerce_float(fast_info.get("lastPrice"))
    if price is None:
        raise MarketDataProviderError(
            f"Yahoo Finance did not return a last trade price for '{normalized_ticker}'."
        )

    previous_close = _coerce_float(
        fast_info.get("previousClose") or fast_info.get("regularMarketPreviousClose")
    )
    change = price - previous_close if previous_close is not None else None
    change_percent = (
        (change / previous_close) * 100 if change is not None and previous_close else None
    )

    timestamp = info.get("regularMarketTime")
    tz_name = info.get("exchangeTimezoneName")

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


DEFAULT_HISTORY_DAYS = 30


def get_price_history(
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
    """
    Retrieve historical candles for the requested ticker and date range.
    """

    normalized_ticker = normalize_ticker(ticker)
    interval_key = validate_interval(interval)

    today = date.today()
    start = start_date or (today - timedelta(days=DEFAULT_HISTORY_DAYS))
    end = end_date or today

    start_dt, end_dt = validate_date_range(start, end)
    ensure_interval_range(start_dt, end_dt, interval_key)

    logger.info(
        "Fetching price history: ticker=%s interval=%s start=%s end=%s",
        normalized_ticker,
        interval_key,
        start_dt.isoformat(),
        end_dt.isoformat(),
    )

    # yfinance uses an exclusive end date, so push forward slightly to include the day
    yf_end = end_dt + timedelta(seconds=1)

    try:
        ticker_client = Ticker(normalized_ticker)
        history = ticker_client.history(
            start=start_dt,
            end=yf_end,
            interval=interval_key,
            auto_adjust=False,
            actions=True,
        )
        fast_info = dict(ticker_client.fast_info.items())
    except Exception as exc:
        logger.exception("Failed to fetch price history: ticker=%s", normalized_ticker)
        raise MarketDataProviderError(
            f"Unable to load price history for '{normalized_ticker}'. "
            "Please try again with a smaller window or later."
        ) from exc

    if history.empty:
        raise MarketDataProviderError(
            "Yahoo Finance returned an empty price series. "
            "Try a broader date range or a different interval."
        )

    points: list[PriceBar] = []
    for ts, row in history.iterrows():
        timestamp = ts.to_pydatetime()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        points.append(
            PriceBar(
                timestamp=timestamp,
                open=_coerce_float(row.get("Open")) or 0.0,
                high=_coerce_float(row.get("High")) or 0.0,
                low=_coerce_float(row.get("Low")) or 0.0,
                close=_coerce_float(row.get("Close")) or 0.0,
                volume=_coerce_float(row.get("Volume")),
                dividends=_coerce_float(row.get("Dividends")),
                stock_splits=_coerce_float(row.get("Stock Splits")),
            )
        )

    return PriceHistoryResponse(
        ticker=normalized_ticker,
        interval=interval_key,
        start=start_dt,
        end=end_dt,
        points=points,
        currency=fast_info.get("currency"),
    )
