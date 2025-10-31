from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pytest

from servers.free_tier import tools
from servers.free_tier.exceptions import (
    MarketDataProviderError,
    MarketDataValidationError,
)


# Mock FastMCP Context for testing
class MockContext:
    """Mock Context object for testing tools that require FastMCP Context."""

    async def info(self, message: str, logger_name: str | None = None, extra: Any = None) -> None:
        """Mock info logging."""
        pass

    async def error(self, message: str, logger_name: str | None = None, extra: Any = None) -> None:
        """Mock error logging."""
        pass

    async def debug(self, message: str, logger_name: str | None = None, extra: Any = None) -> None:
        """Mock debug logging."""
        pass


class DummyFastInfo(dict):
    def items(self):  # pragma: no cover - delegated to dict
        return super().items()


class DummyTicker:
    def __init__(
        self,
        symbol: str,
        *,
        info: dict[str, Any],
        fast_info: dict[str, Any],
        history: pd.DataFrame | None = None,
    ):
        self.symbol = symbol
        self._info = info
        self.fast_info = DummyFastInfo(fast_info)
        self._history = history if history is not None else pd.DataFrame()

    @property
    def info(self) -> dict[str, Any]:
        return self._info

    def history(self, *args, **kwargs) -> pd.DataFrame:
        return self._history


async def test_get_market_movers_success(monkeypatch):
    ctx = MockContext()
    sample_quotes = [
        {
            "symbol": "XYZ",
            "shortName": "XYZ Corp",
            "regularMarketPrice": 12.34,
            "regularMarketChange": 1.23,
            "regularMarketChangePercent": 10.5,
            "regularMarketVolume": 123456,
            "currency": "USD",
            "marketState": "REGULAR",
            "regularMarketTime": 1_700_000_000,
            "exchangeTimezoneName": "America/New_York",
        }
    ]

    def fake_screen(query: str, count: int | None = None, start: int = 0):
        assert query == "day_gainers"
        assert count == 1
        return {"quotes": sample_quotes}

    monkeypatch.setattr(tools, "screen", fake_screen)

    result = await tools.get_market_movers(ctx, category="gainers", limit=1)

    assert result.category == "gainers"
    assert len(result.items) == 1
    mover = result.items[0]
    assert mover.symbol == "XYZ"
    assert mover.price == 12.34


async def test_get_market_movers_invalid_category():
    ctx = MockContext()
    with pytest.raises(MarketDataValidationError):
        await tools.get_market_movers(ctx, category="unknown", limit=5)


async def test_get_market_movers_provider_error(monkeypatch):
    ctx = MockContext()

    def failing_screen(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(tools, "screen", failing_screen)

    with pytest.raises(MarketDataProviderError):
        await tools.get_market_movers(ctx, category="gainers", limit=5)


async def test_get_ticker_data_success(monkeypatch):
    ctx = MockContext()
    fast_info = {
        "lastPrice": 101.0,
        "previousClose": 100.0,
        "open": 100.5,
        "dayHigh": 102.0,
        "dayLow": 99.5,
        "lastVolume": 987654,
        "tenDayAverageVolume": 876543,
        "threeMonthAverageVolume": 765432,
        "marketCap": 1_500_000_000,
        "currency": "USD",
        "exchange": "NMS",
        "fiftyDayAverage": 97.0,
        "twoHundredDayAverage": 90.0,
        "yearHigh": 110.0,
        "yearLow": 80.0,
    }
    info = {
        "shortName": "Demo Corp",
        "regularMarketTime": 1_700_000_000,
        "exchangeTimezoneName": "America/New_York",
        "marketState": "REGULAR",
    }

    dummy = DummyTicker("AAPL", info=info, fast_info=fast_info)
    monkeypatch.setattr(tools, "Ticker", lambda symbol, session=None: dummy)

    result = await tools.get_ticker_data(ctx, "aapl")

    assert result.symbol == "AAPL"
    assert result.price == 101.0
    assert result.previous_close == 100.0
    assert result.change == pytest.approx(1.0)
    assert result.change_percent == pytest.approx(1.0)


async def test_get_ticker_data_invalid_symbol():
    ctx = MockContext()
    with pytest.raises(MarketDataValidationError):
        await tools.get_ticker_data(ctx, "bad ticker!!")


async def test_get_ticker_data_missing_price(monkeypatch):
    ctx = MockContext()
    dummy = DummyTicker(
        "AAPL",
        info={"regularMarketTime": 1_700_000_000, "exchangeTimezoneName": "UTC"},
        fast_info={},
    )
    monkeypatch.setattr(tools, "Ticker", lambda symbol, session=None: dummy)

    with pytest.raises(MarketDataProviderError):
        await tools.get_ticker_data(ctx, "AAPL")


async def test_get_ticker_data_provider_failure(monkeypatch):
    ctx = MockContext()

    def failing_ticker(_symbol: str, session=None):
        raise RuntimeError("ticker failure")

    monkeypatch.setattr(tools, "Ticker", failing_ticker)

    with pytest.raises(MarketDataProviderError):
        await tools.get_ticker_data(ctx, "AAPL")


def create_history_dataframe() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            datetime(2024, 1, 1, 16, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 2, 16, 0, tzinfo=timezone.utc),
        ]
    )
    data = {
        "Open": [100.0, 102.0],
        "High": [105.0, 104.0],
        "Low": [99.0, 101.0],
        "Close": [104.0, 103.0],
        "Volume": [1_000_000, 900_000],
        "Dividends": [0.0, 0.0],
        "Stock Splits": [0.0, 0.0],
    }
    return pd.DataFrame(data, index=index)


async def test_get_price_history_success(monkeypatch):
    ctx = MockContext()
    history_df = create_history_dataframe()
    dummy = DummyTicker(
        "AAPL",
        info={},
        fast_info={"currency": "USD"},
        history=history_df,
    )

    monkeypatch.setattr(tools, "Ticker", lambda symbol, session=None: dummy)

    result = await tools.get_price_history(
        ctx, "AAPL", start_date=date(2024, 1, 1), end_date=date(2024, 1, 2)
    )

    assert result.ticker == "AAPL"
    assert len(result.points) == 2
    assert result.points[0].open == 100.0
    assert result.currency == "USD"


async def test_get_price_history_invalid_interval():
    ctx = MockContext()
    with pytest.raises(MarketDataValidationError):
        await tools.get_price_history(ctx, "AAPL", interval="invalid")


async def test_get_price_history_invalid_range():
    ctx = MockContext()
    with pytest.raises(MarketDataValidationError):
        await tools.get_price_history(
            ctx,
            "AAPL",
            start_date=date(2024, 2, 1),
            end_date=date(2024, 1, 1),
        )


async def test_get_price_history_intraday_window_too_large():
    ctx = MockContext()
    start = date.today() - timedelta(days=31)
    end = date.today()

    with pytest.raises(MarketDataValidationError):
        await tools.get_price_history(ctx, "AAPL", start_date=start, end_date=end, interval="1h")


async def test_get_price_history_provider_error(monkeypatch):
    ctx = MockContext()

    def failing_history(self, *args, **kwargs):
        raise RuntimeError("history failed")

    dummy = DummyTicker(
        "AAPL",
        info={},
        fast_info={"currency": "USD"},
        history=pd.DataFrame(),
    )

    monkeypatch.setattr(tools, "Ticker", lambda symbol, session=None: dummy)
    monkeypatch.setattr(DummyTicker, "history", failing_history)

    with pytest.raises(MarketDataProviderError):
        await tools.get_price_history(
            ctx, "AAPL", start_date=date(2024, 1, 1), end_date=date(2024, 1, 2)
        )


async def test_get_price_history_empty_result(monkeypatch):
    ctx = MockContext()
    dummy = DummyTicker(
        "AAPL",
        info={},
        fast_info={"currency": "USD"},
        history=pd.DataFrame(),
    )
    monkeypatch.setattr(tools, "Ticker", lambda symbol, session=None: dummy)

    with pytest.raises(MarketDataProviderError):
        await tools.get_price_history(
            ctx, "AAPL", start_date=date(2024, 1, 1), end_date=date(2024, 1, 2)
        )
