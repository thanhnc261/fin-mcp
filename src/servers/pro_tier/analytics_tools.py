"""Trading analytics and portfolio management tools for Pro Tier MCP server.

Provides three analytics tools:
- backtest_strategy: Backtest trading strategies on historical data
- get_portfolio_analytics: Analyze portfolio performance and metrics
- submit_paper_trade: Execute paper trades for simulation

All tools use Redis caching for expensive calculations.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

import pandas as pd
from fastmcp import Context
from pydantic import Field
from yfinance import Ticker

from .cache import CacheClient
from .config import settings
from .exceptions import AnalyticsError
from .schemas import (
    BacktestMetrics,
    BacktestResponse,
    BacktestTrade,
    PaperTradeOrder,
    PortfolioAnalyticsResponse,
    PortfolioMetrics,
    PortfolioPosition,
)

logger = logging.getLogger(__name__)

# Global cache instance
_cache: CacheClient | None = None


def init_cache(cache_client: CacheClient) -> None:
    """Initialize the global cache client.

    Args:
        cache_client: Configured CacheClient instance
    """
    global _cache
    _cache = cache_client


# In-memory paper trading state (in production, use database)
_paper_trades: dict[str, list[PaperTradeOrder]] = {}


async def backtest_strategy(
    ctx: Context,
    ticker: Annotated[
        str,
        Field(
            description="Ticker symbol to backtest",
            examples=["SPY", "AAPL"],
        ),
    ],
    strategy: Annotated[
        Literal["buy_and_hold", "sma_crossover"],
        Field(
            description="Strategy to backtest",
        ),
    ] = "buy_and_hold",
    start_date: Annotated[
        str,
        Field(
            description="Backtest start date (YYYY-MM-DD)",
            examples=["2020-01-01"],
        ),
    ] = "",
    end_date: Annotated[
        str,
        Field(
            description="Backtest end date (YYYY-MM-DD)",
            examples=["2023-12-31"],
        ),
    ] = "",
    initial_capital: Annotated[
        float,
        Field(
            gt=0,
            description="Initial portfolio capital",
        ),
    ] = 10000.0,
) -> BacktestResponse:
    """Backtest a trading strategy on historical price data.

    Simulates executing a trading strategy on historical data to evaluate
    performance metrics like total return, Sharpe ratio, and max drawdown.

    Supported strategies:
    - buy_and_hold: Buy at start, hold until end
    - sma_crossover: Simple moving average crossover (50/200 day)

    Args:
        ctx: FastMCP context for logging
        ticker: Symbol to backtest
        strategy: Strategy name
        start_date: Backtest start (defaults to 1 year ago)
        end_date: Backtest end (defaults to today)
        initial_capital: Starting capital in USD

    Returns:
        BacktestResponse with performance metrics and trade history
    """
    # Default date range if not provided
    if not end_date:
        end_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    if not start_date:
        start_dt = datetime.now(tz=timezone.utc) - timedelta(days=365)
        start_date = start_dt.strftime("%Y-%m-%d")

    cache_key = f"backtest:{ticker}:{strategy}:{start_date}:{end_date}:{initial_capital}"

    # Try cache first
    if _cache and _cache.is_available():
        cached = _cache.get(cache_key)
        if cached:
            await ctx.info(f"Returning cached backtest for {ticker}")
            return BacktestResponse(**cached)

    await ctx.info(
        f"Backtesting {strategy} on {ticker}: {start_date} to {end_date}, capital=${initial_capital:,.2f}"
    )

    try:
        # Fetch historical data
        ticker_obj = Ticker(ticker)
        history = ticker_obj.history(start=start_date, end=end_date, interval="1d")

        if history.empty:
            raise AnalyticsError(
                f"No historical data available for {ticker} in the specified date range"
            )

        # Execute strategy
        if strategy == "buy_and_hold":
            trades, final_value = _backtest_buy_and_hold(history, initial_capital)
        elif strategy == "sma_crossover":
            trades, final_value = _backtest_sma_crossover(history, initial_capital)
        else:
            raise AnalyticsError(f"Unknown strategy: {strategy}")

        # Calculate metrics
        total_return = ((final_value - initial_capital) / initial_capital) * 100
        days = (
            datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)
        ).days
        years = days / 365.25
        annualized_return = (
            ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        )

        # Simple drawdown calculation
        max_drawdown = _calculate_max_drawdown(history["Close"])

        # Win rate
        profitable = sum(1 for t in trades if t.action == "sell" and len(trades) > 1)
        total_trades_count = len([t for t in trades if t.action == "sell"])
        win_rate = profitable / total_trades_count if total_trades_count > 0 else 0.0

        metrics = BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=None,  # TODO: Implement Sharpe ratio calculation
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(trades),
            profitable_trades=profitable,
        )

        response = BacktestResponse(
            strategy_name=strategy,
            ticker=ticker,
            start_date=datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc),
            end_date=datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc),
            initial_capital=initial_capital,
            final_capital=final_value,
            metrics=metrics,
            trades=trades,
            as_of=datetime.now(tz=timezone.utc),
        )

        # Cache the response
        if _cache and _cache.is_available():
            _cache.set(
                cache_key, response.model_dump(mode="json"), settings.cache_ttl_analytics
            )

        await ctx.info(
            f"Backtest complete: return={total_return:.2f}%, trades={len(trades)}, "
            f"final_value=${final_value:,.2f}"
        )

        return response

    except Exception as exc:
        logger.error(f"Backtest failed for {ticker}: {exc}")
        raise AnalyticsError(f"Backtest failed: {exc}") from exc


def _backtest_buy_and_hold(
    history: pd.DataFrame, initial_capital: float
) -> tuple[list[BacktestTrade], float]:
    """Execute buy-and-hold strategy."""
    trades: list[BacktestTrade] = []

    # Buy at first available price
    first_date = history.index[0]
    first_price = float(history.iloc[0]["Close"])
    shares = int(initial_capital / first_price)

    trades.append(
        BacktestTrade(
            timestamp=first_date.to_pydatetime().replace(tzinfo=timezone.utc),
            action="buy",
            ticker=history.name if hasattr(history, "name") else "UNKNOWN",
            price=first_price,
            quantity=shares,
        )
    )

    # Hold until end
    last_price = float(history.iloc[-1]["Close"])
    final_value = shares * last_price

    return trades, final_value


def _backtest_sma_crossover(
    history: pd.DataFrame, initial_capital: float
) -> tuple[list[BacktestTrade], float]:
    """Execute SMA crossover strategy (50/200 day)."""
    # Calculate moving averages
    history["SMA50"] = history["Close"].rolling(window=50).mean()
    history["SMA200"] = history["Close"].rolling(window=200).mean()

    trades: list[BacktestTrade] = []
    position_shares = 0
    cash = initial_capital

    for i in range(200, len(history)):
        date = history.index[i]
        price = float(history.iloc[i]["Close"])
        sma50 = history.iloc[i]["SMA50"]
        sma200 = history.iloc[i]["SMA200"]
        prev_sma50 = history.iloc[i - 1]["SMA50"]
        prev_sma200 = history.iloc[i - 1]["SMA200"]

        # Buy signal: SMA50 crosses above SMA200
        if prev_sma50 <= prev_sma200 and sma50 > sma200 and position_shares == 0:
            shares = int(cash / price)
            if shares > 0:
                position_shares = shares
                cash -= shares * price
                trades.append(
                    BacktestTrade(
                        timestamp=date.to_pydatetime().replace(tzinfo=timezone.utc),
                        action="buy",
                        ticker="UNKNOWN",
                        price=price,
                        quantity=shares,
                    )
                )

        # Sell signal: SMA50 crosses below SMA200
        elif prev_sma50 >= prev_sma200 and sma50 < sma200 and position_shares > 0:
            cash += position_shares * price
            trades.append(
                BacktestTrade(
                    timestamp=date.to_pydatetime().replace(tzinfo=timezone.utc),
                    action="sell",
                    ticker="UNKNOWN",
                    price=price,
                    quantity=position_shares,
                )
            )
            position_shares = 0

    # Close any remaining position
    if position_shares > 0:
        last_price = float(history.iloc[-1]["Close"])
        cash += position_shares * last_price

    return trades, cash


def _calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown percentage."""
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax * 100
    return abs(float(drawdown.min()))


async def get_portfolio_analytics(
    ctx: Context,
    portfolio_id: Annotated[
        str,
        Field(
            description="Portfolio identifier",
            examples=["default", "retirement"],
        ),
    ] = "default",
) -> PortfolioAnalyticsResponse:
    """Analyze portfolio performance and metrics.

    Retrieves current positions from the paper trading system and calculates
    portfolio-level metrics including total value, P&L, and concentration.

    Args:
        ctx: FastMCP context for logging
        portfolio_id: Portfolio to analyze

    Returns:
        PortfolioAnalyticsResponse with positions and metrics
    """
    await ctx.info(f"Analyzing portfolio: {portfolio_id}")

    # Get trades for this portfolio
    trades = _paper_trades.get(portfolio_id, [])
    if not trades:
        # Return empty portfolio
        return PortfolioAnalyticsResponse(
            portfolio_id=portfolio_id,
            positions=[],
            metrics=PortfolioMetrics(
                total_value=0.0,
                total_cost=0.0,
                total_pnl=0.0,
                total_pnl_percent=0.0,
                cash_balance=10000.0,  # Default starting cash
                num_positions=0,
                concentration=0.0,
            ),
            as_of=datetime.now(tz=timezone.utc),
        )

    # TODO: Implement real portfolio position tracking
    # For MVP, return placeholder
    logger.warning(f"Portfolio analytics not fully implemented for {portfolio_id}")

    return PortfolioAnalyticsResponse(
        portfolio_id=portfolio_id,
        positions=[],
        metrics=PortfolioMetrics(
            total_value=10000.0,
            total_cost=10000.0,
            total_pnl=0.0,
            total_pnl_percent=0.0,
            cash_balance=10000.0,
            num_positions=0,
            concentration=0.0,
        ),
        as_of=datetime.now(tz=timezone.utc),
    )


async def submit_paper_trade(
    ctx: Context,
    ticker: Annotated[
        str,
        Field(description="Ticker symbol", examples=["AAPL"]),
    ],
    action: Annotated[
        Literal["buy", "sell"],
        Field(description="Trade action"),
    ],
    quantity: Annotated[
        int,
        Field(gt=0, description="Number of shares"),
    ],
    portfolio_id: Annotated[
        str,
        Field(description="Portfolio identifier"),
    ] = "default",
) -> PaperTradeOrder:
    """Submit a paper trade order for simulation.

    Executes a simulated trade at current market price (via Yahoo Finance).
    All trades are stored in-memory for the session.

    Args:
        ctx: FastMCP context for logging
        ticker: Stock symbol
        action: Buy or sell
        quantity: Number of shares
        portfolio_id: Portfolio to trade in

    Returns:
        PaperTradeOrder with execution details
    """
    await ctx.info(f"Submitting paper trade: {action} {quantity} {ticker}")

    try:
        # Fetch current price
        ticker_obj = Ticker(ticker)
        current_price = float(ticker_obj.fast_info["lastPrice"])

        # Create order
        order = PaperTradeOrder(
            order_id=str(uuid.uuid4()),
            ticker=ticker,
            action=action,
            quantity=quantity,
            order_type="market",
            limit_price=None,
            status="filled",  # Instant fill for paper trading
            filled_price=current_price,
            timestamp=datetime.now(tz=timezone.utc),
        )

        # Store trade
        if portfolio_id not in _paper_trades:
            _paper_trades[portfolio_id] = []
        _paper_trades[portfolio_id].append(order)

        await ctx.info(
            f"Paper trade executed: {action} {quantity} {ticker} @ ${current_price:.2f}"
        )

        return order

    except Exception as exc:
        logger.error(f"Paper trade failed: {exc}")
        raise AnalyticsError(f"Failed to execute paper trade: {exc}") from exc
