"""Pydantic schemas for Pro Tier request/response validation.

These models ensure type safety and automatic validation for all Pro Tier
tools, including sentiment analysis, trading analytics, and portfolio management.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ============================================================================
# Sentiment Analysis Schemas
# ============================================================================


class SentimentScore(BaseModel):
    """Individual sentiment indicator with normalized score."""

    source: str = Field(description="Data source name (e.g., 'cnn_fear_greed', 'crypto_fear_greed')")
    raw_value: float | None = Field(description="Original raw value from the source")
    normalized_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Normalized sentiment score (0=extreme fear, 100=extreme greed)",
    )
    timestamp: datetime = Field(description="When this sentiment data was captured")
    status: Literal["active", "stale", "degraded"] = Field(
        default="active",
        description="Data freshness indicator",
    )


class BasicSentimentResponse(BaseModel):
    """Response for get_basic_sentiment() tool."""

    fear_greed_index: SentimentScore | None = Field(
        description="CNN Fear & Greed Index (0-100)"
    )
    crypto_fear_greed: SentimentScore | None = Field(
        description="Crypto Fear & Greed Index (0-100)"
    )
    composite_score: float | None = Field(
        ge=0.0,
        le=100.0,
        description="Weighted average of available sentiment indicators",
    )
    as_of: datetime = Field(description="Timestamp when this response was generated")


class NewsSentiment(BaseModel):
    """News article sentiment analysis result."""

    headline: str = Field(description="Article headline")
    source: str = Field(description="News source/publisher")
    published_at: datetime = Field(description="Publication timestamp")
    sentiment_score: float = Field(
        ge=-1.0,
        le=1.0,
        description="Sentiment polarity (-1=negative, 0=neutral, 1=positive)",
    )
    subjectivity: float = Field(
        ge=0.0,
        le=1.0,
        description="Subjectivity score (0=objective, 1=subjective)",
    )


class AdvancedSentimentResponse(BaseModel):
    """Response for get_advanced_sentiment() tool."""

    basic_sentiment: BasicSentimentResponse = Field(
        description="Basic fear/greed indicators"
    )
    google_trends_score: float | None = Field(
        ge=0.0,
        le=100.0,
        description="Google Trends interest score for the query term",
    )
    news_sentiment_score: float | None = Field(
        ge=-1.0,
        le=1.0,
        description="Aggregated news sentiment from recent articles",
    )
    news_articles: list[NewsSentiment] = Field(
        default_factory=list,
        description="Individual news article sentiments",
    )
    composite_score: float | None = Field(
        ge=0.0,
        le=100.0,
        description="Weighted composite of all sentiment signals",
    )
    as_of: datetime = Field(description="Response generation timestamp")


class SentimentTrendPoint(BaseModel):
    """Single point in a sentiment time series."""

    timestamp: datetime = Field(description="Time point")
    score: float = Field(ge=0.0, le=100.0, description="Sentiment score at this time")


class SentimentTrendsResponse(BaseModel):
    """Response for get_sentiment_trends() tool."""

    query: str = Field(description="Search query or ticker symbol")
    lookback_days: int = Field(description="Number of days in the trend")
    trend_points: list[SentimentTrendPoint] = Field(
        description="Time series of sentiment scores"
    )
    trend_direction: Literal["rising", "falling", "neutral"] = Field(
        description="Overall trend direction"
    )
    volatility: float = Field(ge=0.0, description="Sentiment volatility measure")
    as_of: datetime = Field(description="Response generation timestamp")


# ============================================================================
# Trading Analytics Schemas
# ============================================================================


class BacktestTrade(BaseModel):
    """Individual trade in a backtest simulation."""

    timestamp: datetime = Field(description="Trade execution timestamp")
    action: Literal["buy", "sell"] = Field(description="Trade action")
    ticker: str = Field(description="Ticker symbol")
    price: float = Field(gt=0, description="Execution price")
    quantity: int = Field(gt=0, description="Number of shares")
    commission: float = Field(ge=0, default=0, description="Trading commission paid")


class BacktestMetrics(BaseModel):
    """Performance metrics for a backtest run."""

    total_return: float = Field(description="Total return percentage")
    annualized_return: float = Field(description="Annualized return percentage")
    sharpe_ratio: float | None = Field(description="Risk-adjusted return measure")
    max_drawdown: float = Field(description="Maximum peak-to-trough decline percentage")
    win_rate: float = Field(ge=0, le=1, description="Fraction of profitable trades")
    total_trades: int = Field(ge=0, description="Number of trades executed")
    profitable_trades: int = Field(ge=0, description="Number of winning trades")


class BacktestResponse(BaseModel):
    """Response for backtest_strategy() tool."""

    strategy_name: str = Field(description="Name of the backtested strategy")
    ticker: str = Field(description="Ticker symbol backtested")
    start_date: datetime = Field(description="Backtest start date")
    end_date: datetime = Field(description="Backtest end date")
    initial_capital: float = Field(gt=0, description="Starting portfolio value")
    final_capital: float = Field(gt=0, description="Ending portfolio value")
    metrics: BacktestMetrics = Field(description="Performance metrics")
    trades: list[BacktestTrade] = Field(description="Trade history")
    as_of: datetime = Field(description="Response generation timestamp")


class PortfolioPosition(BaseModel):
    """Current position in a portfolio."""

    ticker: str = Field(description="Ticker symbol")
    quantity: int = Field(description="Number of shares held")
    avg_cost: float = Field(gt=0, description="Average cost basis per share")
    current_price: float = Field(gt=0, description="Current market price")
    market_value: float = Field(gt=0, description="Current position value")
    unrealized_pnl: float = Field(description="Unrealized profit/loss")
    unrealized_pnl_percent: float = Field(description="Unrealized P&L percentage")
    weight: float = Field(ge=0, le=1, description="Position weight in portfolio")


class PortfolioMetrics(BaseModel):
    """Aggregated portfolio analytics."""

    total_value: float = Field(ge=0, description="Total portfolio market value")
    total_cost: float = Field(ge=0, description="Total cost basis")
    total_pnl: float = Field(description="Total unrealized profit/loss")
    total_pnl_percent: float = Field(description="Total P&L percentage")
    cash_balance: float = Field(ge=0, description="Available cash")
    num_positions: int = Field(ge=0, description="Number of holdings")
    concentration: float = Field(
        ge=0, le=1, description="Largest position weight (concentration risk)"
    )


class PortfolioAnalyticsResponse(BaseModel):
    """Response for get_portfolio_analytics() tool."""

    portfolio_id: str = Field(description="Portfolio identifier")
    positions: list[PortfolioPosition] = Field(description="Current holdings")
    metrics: PortfolioMetrics = Field(description="Portfolio-level metrics")
    as_of: datetime = Field(description="Response generation timestamp")


class PaperTradeOrder(BaseModel):
    """Paper trading order submission."""

    order_id: str = Field(description="Unique order identifier")
    ticker: str = Field(description="Ticker symbol")
    action: Literal["buy", "sell"] = Field(description="Order action")
    quantity: int = Field(gt=0, description="Number of shares")
    order_type: Literal["market", "limit"] = Field(default="market", description="Order type")
    limit_price: float | None = Field(gt=0, description="Limit price (if limit order)")
    status: Literal["pending", "filled", "cancelled"] = Field(
        default="pending", description="Order status"
    )
    filled_price: float | None = Field(gt=0, description="Execution price (if filled)")
    timestamp: datetime = Field(description="Order submission timestamp")
