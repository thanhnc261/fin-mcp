"""Tests for Pro Tier sentiment analysis tools.

Tests cover:
- Basic sentiment fetching with degraded sources
- Advanced sentiment with news analysis
- Sentiment trends over time
- Cache behavior
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from servers.pro_tier.cache import CacheClient
from servers.pro_tier.schemas import (
    AdvancedSentimentResponse,
    BasicSentimentResponse,
    SentimentScore,
    SentimentTrendsResponse,
)
from servers.pro_tier.sentiment_tools import (
    get_advanced_sentiment,
    get_basic_sentiment,
    get_sentiment_trends,
    init_cache,
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
def mock_sentiment_scores():
    """Mock sentiment scores from Fear & Greed indices."""
    return {
        "cnn": SentimentScore(
            source="cnn_fear_greed",
            raw_value=65.0,
            normalized_score=65.0,
            timestamp=datetime.now(tz=timezone.utc),
            status="active",
        ),
        "crypto": SentimentScore(
            source="crypto_fear_greed",
            raw_value=72.0,
            normalized_score=72.0,
            timestamp=datetime.now(tz=timezone.utc),
            status="active",
        ),
    }


@pytest.mark.asyncio
async def test_get_basic_sentiment_all_sources_active(
    mock_cache, mock_sentiment_scores
):
    """Test basic sentiment when all sources are active."""
    init_cache(mock_cache)

    with patch(
        "servers.pro_tier.sentiment_tools.fetch_cnn_fear_greed",
        return_value=mock_sentiment_scores["cnn"],
    ), patch(
        "servers.pro_tier.sentiment_tools.fetch_crypto_fear_greed",
        return_value=mock_sentiment_scores["crypto"],
    ):
        ctx = MockContext()
        result = await get_basic_sentiment(ctx)

        assert isinstance(result, BasicSentimentResponse)
        assert result.fear_greed_index is not None
        assert result.crypto_fear_greed is not None
        assert result.composite_score is not None
        # Composite should be average of 65 and 72 = 68.5
        assert 68.0 <= result.composite_score <= 69.0


@pytest.mark.asyncio
async def test_get_basic_sentiment_degraded_source(mock_cache, mock_sentiment_scores):
    """Test basic sentiment when one source is degraded."""
    init_cache(mock_cache)

    with patch(
        "servers.pro_tier.sentiment_tools.fetch_cnn_fear_greed",
        return_value=mock_sentiment_scores["cnn"],
    ), patch(
        "servers.pro_tier.sentiment_tools.fetch_crypto_fear_greed",
        return_value=None,  # Degraded source
    ):
        ctx = MockContext()
        result = await get_basic_sentiment(ctx)

        assert isinstance(result, BasicSentimentResponse)
        assert result.fear_greed_index is not None
        assert result.crypto_fear_greed is None
        assert result.composite_score is not None
        # Composite should be just CNN score
        assert result.composite_score == 65.0


@pytest.mark.asyncio
async def test_get_basic_sentiment_all_sources_degraded(mock_cache):
    """Test basic sentiment when all sources are degraded."""
    init_cache(mock_cache)

    with patch(
        "servers.pro_tier.sentiment_tools.fetch_cnn_fear_greed",
        return_value=None,
    ), patch(
        "servers.pro_tier.sentiment_tools.fetch_crypto_fear_greed",
        return_value=None,
    ):
        ctx = MockContext()
        result = await get_basic_sentiment(ctx)

        assert isinstance(result, BasicSentimentResponse)
        assert result.fear_greed_index is None
        assert result.crypto_fear_greed is None
        assert result.composite_score is None


@pytest.mark.asyncio
async def test_get_advanced_sentiment_with_news(mock_cache, mock_sentiment_scores):
    """Test advanced sentiment including news analysis."""
    init_cache(mock_cache)

    mock_articles = [
        {
            "title": "Tech stocks soar on AI optimism",
            "description": "Market rallies as AI developments boost investor confidence",
            "source": {"name": "TechNews"},
            "publishedAt": "2023-12-01T12:00:00Z",
        },
        {
            "title": "Markets remain cautious amid economic uncertainty",
            "description": "Investors wary of potential headwinds",
            "source": {"name": "Finance Today"},
            "publishedAt": "2023-12-01T11:00:00Z",
        },
    ]

    with patch(
        "servers.pro_tier.sentiment_tools.fetch_cnn_fear_greed",
        return_value=mock_sentiment_scores["cnn"],
    ), patch(
        "servers.pro_tier.sentiment_tools.fetch_crypto_fear_greed",
        return_value=mock_sentiment_scores["crypto"],
    ), patch(
        "servers.pro_tier.sentiment_tools.fetch_news_articles",
        return_value=mock_articles,
    ), patch(
        "servers.pro_tier.sentiment_tools.fetch_google_trends",
        return_value=None,
    ):
        ctx = MockContext()
        result = await get_advanced_sentiment(ctx, "AAPL")

        assert isinstance(result, AdvancedSentimentResponse)
        assert result.basic_sentiment is not None
        assert result.news_articles is not None
        assert len(result.news_articles) > 0
        assert result.composite_score is not None


@pytest.mark.asyncio
async def test_get_sentiment_trends(mock_cache):
    """Test sentiment trends endpoint."""
    init_cache(mock_cache)

    # Mock the get_advanced_sentiment call
    mock_advanced = AdvancedSentimentResponse(
        basic_sentiment=BasicSentimentResponse(
            fear_greed_index=None,
            crypto_fear_greed=None,
            composite_score=60.0,
            as_of=datetime.now(tz=timezone.utc),
        ),
        google_trends_score=None,
        news_sentiment_score=0.2,
        news_articles=[],
        composite_score=62.0,
        as_of=datetime.now(tz=timezone.utc),
    )

    with patch(
        "servers.pro_tier.sentiment_tools.get_advanced_sentiment",
        return_value=mock_advanced,
    ):
        ctx = MockContext()
        result = await get_sentiment_trends(ctx, "TSLA", lookback_days=7)

        assert isinstance(result, SentimentTrendsResponse)
        assert result.query == "TSLA"
        assert result.lookback_days == 7
        assert len(result.trend_points) > 0
        assert result.trend_direction in ["rising", "falling", "neutral"]


@pytest.mark.asyncio
async def test_cache_hit(mock_cache, mock_sentiment_scores):
    """Test that cached responses are returned when available."""
    # Configure cache to return cached data
    cached_data = {
        "fear_greed_index": mock_sentiment_scores["cnn"].model_dump(mode="json"),
        "crypto_fear_greed": mock_sentiment_scores["crypto"].model_dump(mode="json"),
        "composite_score": 68.5,
        "as_of": datetime.now(tz=timezone.utc).isoformat(),
    }
    mock_cache.is_available.return_value = True
    mock_cache.get.return_value = cached_data

    init_cache(mock_cache)

    ctx = MockContext()
    result = await get_basic_sentiment(ctx)

    # Should return cached data without calling fetchers
    assert isinstance(result, BasicSentimentResponse)
    mock_cache.get.assert_called_once()
