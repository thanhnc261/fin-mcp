from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class _BaseModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=False,
    )


class MarketMover(_BaseModel):
    symbol: str
    name: str | None = None
    price: float | None = None
    change: float | None = None
    change_percent: float | None = None
    volume: int | None = None
    currency: str | None = None
    market_state: str | None = None
    timestamp: datetime | None = None


class MarketMoversResponse(_BaseModel):
    category: Literal["gainers", "losers", "active"]
    as_of: datetime
    items: list[MarketMover]
    source: Literal["yfinance"] = "yfinance"


class TickerQuote(_BaseModel):
    symbol: str
    name: str | None = None
    price: float
    change: float | None = None
    change_percent: float | None = None
    previous_close: float | None = None
    open: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    volume: int | None = None
    average_volume_10d: int | None = None
    average_volume_3m: int | None = None
    market_cap: float | None = None
    currency: str | None = None
    exchange: str | None = None
    market_state: str | None = None
    fifty_day_average: float | None = None
    two_hundred_day_average: float | None = None
    year_high: float | None = None
    year_low: float | None = None
    timestamp: datetime | None = None
    source: Literal["yfinance"] = "yfinance"


class PriceBar(_BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    dividends: float | None = None
    stock_splits: float | None = None


class PriceHistoryResponse(_BaseModel):
    ticker: str
    interval: str
    start: datetime
    end: datetime
    points: list[PriceBar]
    currency: str | None = None
    source: Literal["yfinance"] = "yfinance"
