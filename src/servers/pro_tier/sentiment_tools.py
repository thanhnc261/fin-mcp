"""Sentiment analysis tools for Pro Tier MCP server.

Provides three sentiment analysis tools:
- get_basic_sentiment: Fear & Greed indices only
- get_advanced_sentiment: Includes news sentiment and Google Trends
- get_sentiment_trends: Historical sentiment trends over time

All tools use Redis caching to minimize external API calls.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated

from fastmcp import Context
from pydantic import Field

from .cache import CacheClient
from .config import settings
from .schemas import (
    AdvancedSentimentResponse,
    BasicSentimentResponse,
    NewsSentiment,
    SentimentScore,
    SentimentTrendPoint,
    SentimentTrendsResponse,
)
from .sentiment_fetchers import (
    analyze_text_sentiment,
    compute_composite_sentiment,
    fetch_cnn_fear_greed,
    fetch_crypto_fear_greed,
    fetch_google_trends,
    fetch_news_articles,
    normalize_sentiment_to_100,
)

logger = logging.getLogger(__name__)

# Global cache instance (initialized by server)
_cache: CacheClient | None = None


def init_cache(cache_client: CacheClient) -> None:
    """Initialize the global cache client.

    Args:
        cache_client: Configured CacheClient instance
    """
    global _cache
    _cache = cache_client


async def get_basic_sentiment(ctx: Context) -> BasicSentimentResponse:
    """Retrieve basic market sentiment from Fear & Greed indices.

    Fetches the CNN Fear & Greed Index and Crypto Fear & Greed Index, both
    normalized to a 0-100 scale where:
    - 0-25: Extreme Fear
    - 25-45: Fear
    - 45-55: Neutral
    - 55-75: Greed
    - 75-100: Extreme Greed

    Results are cached for 5 minutes to reduce API load.

    Args:
        ctx: FastMCP context for logging

    Returns:
        BasicSentimentResponse with fear/greed indices and composite score
    """
    cache_key = "sentiment:basic"

    # Try cache first
    if _cache and _cache.is_available():
        cached = _cache.get(cache_key)
        if cached:
            await ctx.info("Returning cached basic sentiment data")
            return BasicSentimentResponse(**cached)

    await ctx.info("Fetching basic sentiment from Fear & Greed indices")

    # Fetch both indices concurrently
    cnn_score = await fetch_cnn_fear_greed()
    crypto_score = await fetch_crypto_fear_greed()

    # Compute composite if at least one source is available
    composite = compute_composite_sentiment([cnn_score, crypto_score])

    response = BasicSentimentResponse(
        fear_greed_index=cnn_score,
        crypto_fear_greed=crypto_score,
        composite_score=composite,
        as_of=datetime.now(tz=timezone.utc),
    )

    # Cache the response
    if _cache and _cache.is_available():
        _cache.set(cache_key, response.model_dump(mode="json"), settings.cache_ttl_sentiment)

    composite_str = f"{composite:.1f}" if composite is not None else "N/A"
    await ctx.info(
        f"Basic sentiment: composite={composite_str}, "
        f"cnn={'OK' if cnn_score else 'DEGRADED'}, "
        f"crypto={'OK' if crypto_score else 'DEGRADED'}"
    )

    return response


async def get_advanced_sentiment(
    ctx: Context,
    query: Annotated[
        str,
        Field(
            description="Search query (ticker symbol, company name, or market term)",
            examples=["AAPL", "Bitcoin", "AI stocks"],
        ),
    ],
) -> AdvancedSentimentResponse:
    """Retrieve advanced sentiment analysis including news and trends.

    Combines Fear & Greed indices with:
    - Google Trends interest score
    - News sentiment from recent articles (via NewsAPI + TextBlob)
    - Weighted composite score

    Results are cached per query for 5 minutes.

    Args:
        ctx: FastMCP context for logging
        query: Search term for news and trends analysis

    Returns:
        AdvancedSentimentResponse with all sentiment signals
    """
    cache_key = f"sentiment:advanced:{query}"

    # Try cache first
    if _cache and _cache.is_available():
        cached = _cache.get(cache_key)
        if cached:
            await ctx.info(f"Returning cached advanced sentiment for query={query}")
            return AdvancedSentimentResponse(**cached)

    await ctx.info(f"Fetching advanced sentiment for query={query}")

    # Get basic sentiment (already cached)
    basic = await get_basic_sentiment(ctx)

    # Fetch Google Trends (may return None if not implemented)
    trends_score = await fetch_google_trends(query, days=7)

    # Fetch and analyze news articles
    news_articles: list[NewsSentiment] = []
    news_sentiment_score: float | None = None

    try:
        articles = await fetch_news_articles(
            query, settings.news_api_key, max_results=settings.news_max_articles
        )

        if articles:
            sentiment_scores: list[float] = []

            for article in articles[:50]:  # Analyze top 50 for performance
                headline = article.get("title", "")
                description = article.get("description", "")
                text = f"{headline} {description}"

                polarity, subjectivity = analyze_text_sentiment(text)
                sentiment_scores.append(polarity)

                # Parse timestamp
                published_str = article.get("publishedAt", "")
                try:
                    published_at = datetime.fromisoformat(
                        published_str.replace("Z", "+00:00")
                    )
                except Exception:
                    published_at = datetime.now(tz=timezone.utc)

                news_articles.append(
                    NewsSentiment(
                        headline=headline,
                        source=article.get("source", {}).get("name", "Unknown"),
                        published_at=published_at,
                        sentiment_score=polarity,
                        subjectivity=subjectivity,
                    )
                )

            # Average news sentiment
            if sentiment_scores:
                news_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)

    except Exception as exc:
        logger.warning(f"News sentiment analysis failed for query={query}: {exc}")

    # Compute composite score from all available signals
    # Weight: 40% basic sentiment, 30% news, 30% trends
    composite_components: list[float] = []

    if basic.composite_score is not None:
        composite_components.append(basic.composite_score * 0.4)

    if news_sentiment_score is not None:
        # Convert news sentiment (-1 to 1) to 0-100 scale
        news_normalized = normalize_sentiment_to_100(news_sentiment_score)
        composite_components.append(news_normalized * 0.3)

    if trends_score is not None:
        composite_components.append(trends_score * 0.3)

    composite_score = sum(composite_components) if composite_components else None

    response = AdvancedSentimentResponse(
        basic_sentiment=basic,
        google_trends_score=trends_score,
        news_sentiment_score=news_sentiment_score,
        news_articles=news_articles,
        composite_score=composite_score,
        as_of=datetime.now(tz=timezone.utc),
    )

    # Cache the response
    if _cache and _cache.is_available():
        _cache.set(cache_key, response.model_dump(mode="json"), settings.cache_ttl_sentiment)

    composite_str = f"{composite_score:.1f}" if composite_score is not None else "N/A"
    await ctx.info(
        f"Advanced sentiment: query={query}, composite={composite_str}, "
        f"news_articles={len(news_articles)}"
    )

    return response


async def get_sentiment_trends(
    ctx: Context,
    query: Annotated[
        str,
        Field(
            description="Search query (ticker symbol, company name, or market term)",
            examples=["TSLA", "Ethereum", "tech stocks"],
        ),
    ],
    lookback_days: Annotated[
        int,
        Field(
            ge=1,
            le=30,
            description="Number of days to analyze (1-30)",
        ),
    ] = 7,
) -> SentimentTrendsResponse:
    """Retrieve historical sentiment trends over time.

    Note: This is a simplified implementation for MVP. Real trend analysis would
    require storing historical sentiment data in a time-series database. For now,
    we return a placeholder response demonstrating the API contract.

    Args:
        ctx: FastMCP context for logging
        query: Search term for trend analysis
        lookback_days: Number of days to analyze

    Returns:
        SentimentTrendsResponse with trend points and analysis
    """
    await ctx.info(f"Fetching sentiment trends: query={query}, days={lookback_days}")

    # TODO: Implement real historical trend tracking with time-series storage
    # For MVP, we'll return a simple placeholder with current sentiment
    current_sentiment = await get_advanced_sentiment(ctx, query)
    current_score = current_sentiment.composite_score or 50.0

    # Generate mock trend points (in production, query from time-series DB)
    trend_points = [
        SentimentTrendPoint(
            timestamp=datetime.now(tz=timezone.utc),
            score=current_score,
        )
    ]

    response = SentimentTrendsResponse(
        query=query,
        lookback_days=lookback_days,
        trend_points=trend_points,
        trend_direction="neutral",
        volatility=0.0,
        as_of=datetime.now(tz=timezone.utc),
    )

    logger.warning(
        f"Sentiment trends not fully implemented, returning placeholder for query={query}"
    )

    return response
