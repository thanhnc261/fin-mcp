"""FastMCP server for free-tier Yahoo Finance market data tools.

This server provides three tools for accessing market data without API keys:
- get_market_movers: Top gainers, losers, and most active stocks
- get_ticker_data: Real-time quote snapshot for a ticker
- get_price_history: Historical OHLCV candles

The server supports multiple transports (STDIO, HTTP, SSE) configurable
via environment variables. See config.py for available settings.
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
    from servers.free_tier.config import settings
    from servers.free_tier.logging_config import configure_logging
    from servers.free_tier.tools import (
        get_market_movers,
        get_price_history,
        get_ticker_data,
    )
else:
    from .config import settings
    from .logging_config import configure_logging
    from .tools import get_market_movers, get_price_history, get_ticker_data

# Initialize FastMCP application
app = FastMCP(
    name="free-tier-market-data",
    instructions=(
        "Tools that provide Yahoo Finance market data suitable for Claude's free tier, "
        "including market movers, ticker snapshots, and historical prices. "
        "All data is sourced from Yahoo Finance's public API without requiring API keys."
    ),
    version="0.1.0",
)

# Register tools using decorator pattern
# FastMCP automatically handles Pydantic model serialization
app.tool(get_market_movers)
app.tool(get_ticker_data)
app.tool(get_price_history)


def main() -> None:
    """Main entry point for the MCP server.

    Configures logging and starts the server with the appropriate transport
    based on settings (STDIO, HTTP, or SSE).
    """
    # Configure logging before any other operations
    configure_logging(settings)

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
