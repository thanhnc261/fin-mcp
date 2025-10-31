"""Sentiment data fetchers for Fear & Greed indices and news sources.

Provides utilities to fetch and normalize sentiment indicators from:
- CNN Fear & Greed Index
- Crypto Fear & Greed Index
- Google Trends
- News sentiment via NewsAPI + TextBlob

All fetchers handle failures gracefully and return None for degraded sources.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from .exceptions import SentimentProviderError
from .schemas import SentimentScore

logger = logging.getLogger(__name__)

# External API endpoints
CNN_FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
CRYPTO_FEAR_GREED_URL = "https://api.alternative.me/fng/"
NEWS_API_URL = "https://newsapi.org/v2/everything"


async def fetch_cnn_fear_greed() -> SentimentScore | None:
    """Fetch CNN Fear & Greed Index.

    The index ranges from 0 (extreme fear) to 100 (extreme greed) and is already
    normalized, so no additional transformation is needed.

    Returns:
        SentimentScore with normalized index or None if unavailable
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(CNN_FEAR_GREED_URL)
            response.raise_for_status()
            data = response.json()

        # Extract current value from the graphdata structure
        # CNN returns: {"fear_and_greed": {"score": float, "rating": str, ...}, ...}
        fear_greed = data.get("fear_and_greed", {})
        score = fear_greed.get("score")

        if score is None:
            logger.warning("CNN Fear & Greed API returned no score")
            return None

        raw_value = float(score)

        return SentimentScore(
            source="cnn_fear_greed",
            raw_value=raw_value,
            normalized_score=raw_value,  # Already 0-100
            timestamp=datetime.now(tz=timezone.utc),
            status="active",
        )

    except Exception as exc:
        logger.warning(f"Failed to fetch CNN Fear & Greed Index: {exc}")
        return None


async def fetch_crypto_fear_greed() -> SentimentScore | None:
    """Fetch Crypto Fear & Greed Index from alternative.me.

    The index ranges from 0 (extreme fear) to 100 (extreme greed).

    Returns:
        SentimentScore with normalized index or None if unavailable
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(CRYPTO_FEAR_GREED_URL)
            response.raise_for_status()
            data = response.json()

        # Response format: {"data": [{"value": str, "value_classification": str, "timestamp": str}], ...}
        data_points = data.get("data", [])
        if not data_points:
            logger.warning("Crypto Fear & Greed API returned no data")
            return None

        latest = data_points[0]
        value = latest.get("value")
        timestamp_str = latest.get("timestamp")

        if value is None:
            logger.warning("Crypto Fear & Greed API returned no value")
            return None

        raw_value = float(value)
        timestamp = (
            datetime.fromtimestamp(int(timestamp_str), tz=timezone.utc)
            if timestamp_str
            else datetime.now(tz=timezone.utc)
        )

        return SentimentScore(
            source="crypto_fear_greed",
            raw_value=raw_value,
            normalized_score=raw_value,  # Already 0-100
            timestamp=timestamp,
            status="active",
        )

    except Exception as exc:
        logger.warning(f"Failed to fetch Crypto Fear & Greed Index: {exc}")
        return None


async def fetch_google_trends(query: str, days: int = 7) -> float | None:
    """Fetch Google Trends interest score for a query term.

    Note: This is a placeholder implementation. Real Google Trends access requires
    the pytrends library and may be subject to rate limiting. For now, we return
    a mock value to demonstrate the integration pattern.

    Args:
        query: Search term (e.g., ticker symbol, market term)
        days: Lookback period in days

    Returns:
        Interest score (0-100) or None if unavailable
    """
    # TODO: Implement real Google Trends integration with pytrends
    # For MVP, we'll log a warning and return None
    logger.warning(f"Google Trends integration not implemented, query={query}")
    return None


async def fetch_news_articles(
    query: str, api_key: str, max_results: int = 100
) -> list[dict[str, Any]]:
    """Fetch recent news articles from NewsAPI.

    Args:
        query: Search query (e.g., ticker symbol, company name)
        api_key: NewsAPI.org API key
        max_results: Maximum articles to retrieve

    Returns:
        List of article dictionaries with 'title', 'source', 'publishedAt', 'description'

    Raises:
        SentimentProviderError: If NewsAPI request fails
    """
    if not api_key:
        logger.warning("NewsAPI key not configured, skipping news sentiment")
        return []

    try:
        params = {
            "q": query,
            "apiKey": api_key,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(max_results, 100),  # NewsAPI limit is 100
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(NEWS_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

        if data.get("status") != "ok":
            logger.warning(f"NewsAPI returned non-ok status: {data}")
            return []

        articles = data.get("articles", [])
        logger.info(f"Fetched {len(articles)} news articles for query={query}")
        return articles

    except Exception as exc:
        logger.error(f"Failed to fetch news articles: {exc}")
        raise SentimentProviderError(
            f"NewsAPI request failed for query '{query}'"
        ) from exc


def analyze_text_sentiment(text: str) -> tuple[float, float]:
    """Analyze sentiment of text using TextBlob.

    Args:
        text: Text to analyze (headline, article snippet, etc.)

    Returns:
        Tuple of (polarity, subjectivity)
        - polarity: -1.0 (negative) to 1.0 (positive)
        - subjectivity: 0.0 (objective) to 1.0 (subjective)
    """
    try:
        from textblob import TextBlob

        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except Exception as exc:
        logger.warning(f"Text sentiment analysis failed: {exc}")
        return 0.0, 0.0  # Neutral fallback


def normalize_sentiment_to_100(polarity: float) -> float:
    """Normalize TextBlob polarity (-1 to 1) to 0-100 scale.

    Args:
        polarity: Sentiment polarity from -1.0 to 1.0

    Returns:
        Normalized score from 0 (very negative) to 100 (very positive)
    """
    # Map [-1, 1] to [0, 100]
    return (polarity + 1.0) * 50.0


def compute_composite_sentiment(scores: list[SentimentScore | None]) -> float | None:
    """Compute weighted average of available sentiment scores.

    Args:
        scores: List of SentimentScore objects (may contain None for degraded sources)

    Returns:
        Composite score (0-100) or None if no valid scores available
    """
    valid_scores = [s for s in scores if s is not None and s.status == "active"]

    if not valid_scores:
        return None

    # Simple equal-weighted average for MVP
    # TODO: Implement weighted averaging based on source reliability
    total = sum(s.normalized_score for s in valid_scores)
    return total / len(valid_scores)
