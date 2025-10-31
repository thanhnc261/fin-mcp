"""FastMCP server for Pro Tier analytics and sentiment tools.

This server provides advanced market sentiment analysis and trading analytics:
- get_basic_sentiment: Fear & Greed indices
- get_advanced_sentiment: News sentiment + trends
- get_sentiment_trends: Historical sentiment analysis
- backtest_strategy: Strategy backtesting
- get_portfolio_analytics: Portfolio performance analysis
- submit_paper_trade: Paper trading simulation

The server requires Redis for caching and external API keys for news sentiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastmcp import FastMCP

# Handle both direct execution and package import
if __package__ in {None, ""}:
    project_src = Path(__file__).resolve().parents[2]
    if project_src.is_dir():
        sys.path.insert(0, project_src.as_posix())
    from servers.pro_tier.analytics_tools import (
        backtest_strategy,
        get_portfolio_analytics,
        init_cache as init_analytics_cache,
        submit_paper_trade,
    )
    from servers.pro_tier.cache import CacheClient
    from servers.pro_tier.config import settings
    from servers.pro_tier.logging_config import configure_logging
    from servers.pro_tier.sentiment_tools import (
        get_advanced_sentiment,
        get_basic_sentiment,
        get_sentiment_trends,
        init_cache as init_sentiment_cache,
    )
else:
    from .analytics_tools import (
        backtest_strategy,
        get_portfolio_analytics,
        init_cache as init_analytics_cache,
        submit_paper_trade,
    )
    from .cache import CacheClient
    from .config import settings
    from .logging_config import configure_logging
    from .sentiment_tools import (
        get_advanced_sentiment,
        get_basic_sentiment,
        get_sentiment_trends,
        init_cache as init_sentiment_cache,
    )

# Initialize FastMCP application
app = FastMCP(
    name="pro-tier-analytics",
    instructions=(
        "Advanced market sentiment and trading analytics tools for professional traders. "
        "Provides Fear & Greed indices, news sentiment analysis, strategy backtesting, "
        "portfolio analytics, and paper trading simulation. All data is cached via Redis "
        "to minimize external API calls and improve performance."
    ),
    version="0.1.0",
)

# Register sentiment tools
app.tool(get_basic_sentiment)
app.tool(get_advanced_sentiment)
app.tool(get_sentiment_trends)

# Register analytics tools
app.tool(backtest_strategy)
app.tool(get_portfolio_analytics)
app.tool(submit_paper_trade)


def main() -> None:
    """Main entry point for the Pro Tier MCP server.

    Configures logging, initializes Redis cache, and starts the server with
    the appropriate transport based on settings (STDIO, HTTP, or SSE).
    """
    # Configure logging before any other operations
    configure_logging(settings)

    # Initialize Redis cache
    cache = CacheClient(settings.redis_url)
    init_sentiment_cache(cache)
    init_analytics_cache(cache)

    # Determine transport type from settings
    transport = settings.transport_type

    if transport == "stdio":
        # Default MCP transport for local tool integration
        app.run()
    elif transport == "http":
        # HTTP transport for remote access
        app.run(transport="http", host=settings.http_host, port=settings.http_port)
    elif transport == "sse":
        # Server-Sent Events transport for push notifications
        app.run(transport="sse", host=settings.http_host, port=settings.http_port)
    else:
        # This should never happen due to Pydantic validation, but be defensive
        raise ValueError(f"Unsupported transport type: {transport}")


if __name__ == "__main__":
    main()
