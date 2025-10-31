"""Tests for Pro Tier trading analytics tools.

Tests cover:
- Strategy backtesting
- Portfolio analytics
- Paper trading simulation
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from servers.pro_tier.analytics_tools import (
    backtest_strategy,
    get_portfolio_analytics,
    init_cache,
    submit_paper_trade,
)
from servers.pro_tier.cache import CacheClient
from servers.pro_tier.schemas import (
    BacktestResponse,
    PaperTradeOrder,
    PortfolioAnalyticsResponse,
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


@pytest.fixture
def mock_cache():
    """Mock Redis cache client."""
    cache = MagicMock(spec=CacheClient)
    cache.is_available.return_value = False  # Disable caching for tests
    return cache


@pytest.fixture
def mock_price_history():
    """Mock historical price data for backtesting."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    data = {
        "Open": [100 + i * 0.1 for i in range(len(dates))],
        "High": [101 + i * 0.1 for i in range(len(dates))],
        "Low": [99 + i * 0.1 for i in range(len(dates))],
        "Close": [100 + i * 0.1 for i in range(len(dates))],
        "Volume": [1000000 for _ in range(len(dates))],
    }
    return pd.DataFrame(data, index=dates)


@pytest.mark.asyncio
async def test_backtest_buy_and_hold(mock_cache, mock_price_history):
    """Test buy-and-hold strategy backtest."""
    init_cache(mock_cache)

    with patch("servers.pro_tier.analytics_tools.Ticker") as mock_ticker:
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_price_history
        mock_ticker.return_value = mock_ticker_instance

        ctx = MockContext()
        result = await backtest_strategy(
            ctx,
            ticker="SPY",
            strategy="buy_and_hold",
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=10000.0,
        )

        assert isinstance(result, BacktestResponse)
        assert result.strategy_name == "buy_and_hold"
        assert result.ticker == "SPY"
        assert result.initial_capital == 10000.0
        assert result.final_capital > 0
        assert len(result.trades) > 0
        assert result.metrics.total_return is not None
        assert result.metrics.max_drawdown >= 0


@pytest.mark.asyncio
async def test_backtest_sma_crossover(mock_cache, mock_price_history):
    """Test SMA crossover strategy backtest."""
    init_cache(mock_cache)

    # Create price data with more variation for crossover signals
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    # Add some oscillation to trigger crossovers
    prices = [
        100 + 10 * ((i % 100) / 50 - 1) + i * 0.05 for i in range(len(dates))
    ]
    history = pd.DataFrame(
        {
            "Open": prices,
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.99 for p in prices],
            "Close": prices,
            "Volume": [1000000 for _ in range(len(dates))],
        },
        index=dates,
    )

    with patch("servers.pro_tier.analytics_tools.Ticker") as mock_ticker:
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = history
        mock_ticker.return_value = mock_ticker_instance

        ctx = MockContext()
        result = await backtest_strategy(
            ctx,
            ticker="AAPL",
            strategy="sma_crossover",
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=10000.0,
        )

        assert isinstance(result, BacktestResponse)
        assert result.strategy_name == "sma_crossover"
        assert result.ticker == "AAPL"
        assert result.metrics.total_trades >= 0


@pytest.mark.asyncio
async def test_submit_paper_trade(mock_cache):
    """Test paper trade submission."""
    init_cache(mock_cache)

    mock_fast_info = {"lastPrice": 150.0}

    with patch("servers.pro_tier.analytics_tools.Ticker") as mock_ticker:
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.fast_info = mock_fast_info
        mock_ticker.return_value = mock_ticker_instance

        ctx = MockContext()
        result = await submit_paper_trade(
            ctx,
            ticker="AAPL",
            action="buy",
            quantity=10,
            portfolio_id="test_portfolio",
        )

        assert isinstance(result, PaperTradeOrder)
        assert result.ticker == "AAPL"
        assert result.action == "buy"
        assert result.quantity == 10
        assert result.status == "filled"
        assert result.filled_price == 150.0
        assert result.order_id is not None


@pytest.mark.asyncio
async def test_get_portfolio_analytics_empty(mock_cache):
    """Test portfolio analytics with no positions."""
    init_cache(mock_cache)

    ctx = MockContext()
    result = await get_portfolio_analytics(ctx, portfolio_id="empty_portfolio")

    assert isinstance(result, PortfolioAnalyticsResponse)
    assert result.portfolio_id == "empty_portfolio"
    assert len(result.positions) == 0
    assert result.metrics.num_positions == 0


@pytest.mark.asyncio
async def test_backtest_invalid_date_range(mock_cache):
    """Test backtest with invalid date range."""
    init_cache(mock_cache)

    with patch("servers.pro_tier.analytics_tools.Ticker") as mock_ticker:
        mock_ticker_instance = MagicMock()
        # Return empty DataFrame for invalid date range
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance

        ctx = MockContext()

        with pytest.raises(Exception):  # Should raise AnalyticsError
            await backtest_strategy(
                ctx,
                ticker="INVALID",
                strategy="buy_and_hold",
                start_date="2023-01-01",
                end_date="2023-12-31",
                initial_capital=10000.0,
            )


@pytest.mark.asyncio
async def test_cache_backtest_result(mock_cache, mock_price_history):
    """Test that backtest results are cached."""
    # Enable cache
    mock_cache.is_available.return_value = True
    mock_cache.get.return_value = None  # First call, no cache
    init_cache(mock_cache)

    with patch("servers.pro_tier.analytics_tools.Ticker") as mock_ticker:
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_price_history
        mock_ticker.return_value = mock_ticker_instance

        ctx = MockContext()
        result = await backtest_strategy(
            ctx,
            ticker="SPY",
            strategy="buy_and_hold",
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=10000.0,
        )

        # Verify cache was written to
        mock_cache.set.assert_called_once()
        assert isinstance(result, BacktestResponse)
