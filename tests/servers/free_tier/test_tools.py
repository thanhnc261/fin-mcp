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


class DummyFastInfo(dict):
    def items(self):  # pragma: no cover - delegated to dict
        return super().items()


class DummyTicker:
    def __init__(self, symbol: str, *, info: dict[str, Any], fast_info: dict[str, Any], history: pd.DataFrame | None = None):
        self.symbol = symbol
        self._info = info
        self.fast_info = DummyFastInfo(fast_info)
        self._history = history if history is not None else pd.DataFrame()

    @property
    def info(self) -> dict[str, Any]:
        return self._info

    def history(self, *args, **kwargs) -> pd.DataFrame:
        return self._history


def test_get_market_movers_success(monkeypatch):
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

    result = tools.get_market_movers(category="gainers", limit=1)

    assert result.category == "gainers"
    assert len(result.items) == 1
    mover = result.items[0]
    assert mover.symbol == "XYZ"
    assert mover.price == 12.34


def test_get_market_movers_invalid_category():
    with pytest.raises(MarketDataValidationError):
        tools.get_market_movers(category="unknown", limit=5)


def test_get_market_movers_provider_error(monkeypatch):
    def failing_screen(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(tools, "screen", failing_screen)

    with pytest.raises(MarketDataProviderError):
        tools.get_market_movers(category="gainers", limit=5)


def test_get_ticker_data_success(monkeypatch):
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
    monkeypatch.setattr(tools, "Ticker", lambda symbol: dummy)

    result = tools.get_ticker_data("aapl")

    assert result.symbol == "AAPL"
    assert result.price == 101.0
    assert result.previous_close == 100.0
    assert result.change == pytest.approx(1.0)
    assert result.change_percent == pytest.approx(1.0)


def test_get_ticker_data_invalid_symbol():
    with pytest.raises(MarketDataValidationError):
        tools.get_ticker_data("bad ticker!!")


def test_get_ticker_data_missing_price(monkeypatch):
    dummy = DummyTicker(
        "AAPL",
        info={"regularMarketTime": 1_700_000_000, "exchangeTimezoneName": "UTC"},
        fast_info={},
    )
    monkeypatch.setattr(tools, "Ticker", lambda symbol: dummy)

    with pytest.raises(MarketDataProviderError):
        tools.get_ticker_data("AAPL")


def test_get_ticker_data_provider_failure(monkeypatch):
    def failing_ticker(_symbol: str):
        raise RuntimeError("ticker failure")

    monkeypatch.setattr(tools, "Ticker", failing_ticker)

    with pytest.raises(MarketDataProviderError):
        tools.get_ticker_data("AAPL")


def create_history_dataframe() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [datetime(2024, 1, 1, 16, 0, tzinfo=timezone.utc), datetime(2024, 1, 2, 16, 0, tzinfo=timezone.utc)]
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


def test_get_price_history_success(monkeypatch):
    history_df = create_history_dataframe()
    dummy = DummyTicker(
        "AAPL",
        info={},
        fast_info={"currency": "USD"},
        history=history_df,
    )

    monkeypatch.setattr(tools, "Ticker", lambda symbol: dummy)

    result = tools.get_price_history("AAPL", start_date=date(2024, 1, 1), end_date=date(2024, 1, 2))

    assert result.ticker == "AAPL"
    assert len(result.points) == 2
    assert result.points[0].open == 100.0
    assert result.currency == "USD"


def test_get_price_history_invalid_interval():
    with pytest.raises(MarketDataValidationError):
        tools.get_price_history("AAPL", interval="invalid")


def test_get_price_history_invalid_range():
    with pytest.raises(MarketDataValidationError):
        tools.get_price_history(
            "AAPL",
            start_date=date(2024, 2, 1),
            end_date=date(2024, 1, 1),
        )


def test_get_price_history_intraday_window_too_large():
    start = date.today() - timedelta(days=31)
    end = date.today()

    with pytest.raises(MarketDataValidationError):
        tools.get_price_history("AAPL", start_date=start, end_date=end, interval="1h")


def test_get_price_history_provider_error(monkeypatch):
    def failing_history(self, *args, **kwargs):
        raise RuntimeError("history failed")

    dummy = DummyTicker(
        "AAPL",
        info={},
        fast_info={"currency": "USD"},
        history=pd.DataFrame(),
    )

    monkeypatch.setattr(tools, "Ticker", lambda symbol: dummy)
    monkeypatch.setattr(DummyTicker, "history", failing_history)

    with pytest.raises(MarketDataProviderError):
        tools.get_price_history("AAPL", start_date=date(2024, 1, 1), end_date=date(2024, 1, 2))


def test_get_price_history_empty_result(monkeypatch):
    dummy = DummyTicker(
        "AAPL",
        info={},
        fast_info={"currency": "USD"},
        history=pd.DataFrame(),
    )
    monkeypatch.setattr(tools, "Ticker", lambda symbol: dummy)

    with pytest.raises(MarketDataProviderError):
        tools.get_price_history("AAPL", start_date=date(2024, 1, 1), end_date=date(2024, 1, 2))
