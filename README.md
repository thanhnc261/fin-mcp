# Fin-MCP: Financial Market Data & Analytics MCP Servers

High-quality Model Context Protocol (MCP) servers providing market data, sentiment analysis, and trading analytics for AI assistants.

## Servers

### Free Tier - Market Data

Yahoo Finance market data without requiring API keys.

**Features:**
- **Market Movers**: Top gainers, losers, and most active stocks
- **Ticker Quotes**: Real-time price snapshots with technical indicators
- **Price History**: Historical OHLCV candles for technical analysis
- **Multi-Transport**: STDIO, HTTP, and SSE support
- **Type-Safe**: Full type hints with Pydantic validation
- **Well-Tested**: Comprehensive test suite with async support

**Tools:**
- `get_market_movers` – Top gainers, losers, or most active equities
- `get_ticker_data` – Real-time quote snapshot for a single symbol
- `get_price_history` – Historical OHLCV candles for technical analysis

### Pro Tier - Analytics & Sentiment

Advanced sentiment analysis and trading analytics requiring API keys and Redis.

**Features:**
- **Sentiment Analysis**: CNN & Crypto Fear/Greed indices, news sentiment, trends
- **Trading Analytics**: Strategy backtesting with performance metrics
- **Portfolio Management**: Position tracking and P&L analytics
- **Paper Trading**: Simulated trade execution for testing
- **Redis Caching**: 5-minute TTL for sentiment, 15-minute for analytics
- **Graceful Degradation**: Continues operating with partial data sources

**Tools:**
- `get_basic_sentiment` – Fear & Greed indices (0-100 scale)
- `get_advanced_sentiment` – News sentiment + Google Trends
- `get_sentiment_trends` – Historical sentiment analysis
- `backtest_strategy` – Backtest buy-and-hold or SMA crossover strategies
- `get_portfolio_analytics` – Portfolio performance metrics
- `submit_paper_trade` – Execute simulated trades

## Getting Started

```bash
# Install dependencies
make dev

# Run the server (STDIO mode by default)
python -m servers.free_tier.server
```

## Configuration

All settings are configurable via environment variables with the `FREE_TIER_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `FREE_TIER_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `FREE_TIER_TRANSPORT_TYPE` | `stdio` | Transport protocol (stdio, http, sse) |
| `FREE_TIER_HTTP_HOST` | `127.0.0.1` | Host address for HTTP/SSE transport |
| `FREE_TIER_HTTP_PORT` | `8000` | Port number for HTTP/SSE transport |
| `FREE_TIER_YAHOO_TIMEOUT` | `10.0` | HTTP timeout for Yahoo Finance (seconds) |
| `FREE_TIER_INTRADAY_MAX_DAYS` | `30` | Max days for intraday intervals |
| `FREE_TIER_HISTORY_MAX_YEARS` | `10` | Max years for historical data |

### Examples

```bash
# Debug logging
export FREE_TIER_LOG_LEVEL=DEBUG

# HTTP server mode
export FREE_TIER_TRANSPORT_TYPE=http
export FREE_TIER_HTTP_PORT=8080
python -m servers.free_tier.server
```

## Running the Servers

### Free Tier Server

```bash
source .venv/bin/activate
fastmcp run src/servers/free_tier/server.py
```

To iterate with the MCP Inspector UI:

```bash
source .venv/bin/activate
fastmcp dev src/servers/free_tier/server.py
```

Pre-built Inspector scripts covering success, validation, and failure paths for
each tool live under `scripts/mcp_inspector/free_tier/`.

### Pro Tier Server

**Prerequisites:**
1. Install and start Redis:
   ```bash
   # macOS
   brew install redis
   brew services start redis

   # Linux
   sudo apt-get install redis-server
   sudo systemctl start redis
   ```

2. Set required API keys (optional, server degrades gracefully):
   ```bash
   export PRO_TIER_NEWS_API_KEY="your-newsapi-key"
   export PRO_TIER_ALPHA_VANTAGE_KEY="your-alpha-vantage-key"
   ```

**Run the server:**

```bash
source .venv/bin/activate
fastmcp run src/servers/pro_tier/server.py
```

To iterate with the MCP Inspector UI:

```bash
source .venv/bin/activate
fastmcp dev src/servers/pro_tier/server.py
```

**Configuration:**

All Pro Tier settings use the `PRO_TIER_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `PRO_TIER_LOG_LEVEL` | `INFO` | Logging level |
| `PRO_TIER_TRANSPORT_TYPE` | `stdio` | Transport protocol |
| `PRO_TIER_HTTP_HOST` | `127.0.0.1` | HTTP/SSE host |
| `PRO_TIER_HTTP_PORT` | `8001` | HTTP/SSE port |
| `PRO_TIER_REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `PRO_TIER_CACHE_TTL_SENTIMENT` | `300` | Sentiment cache TTL (seconds) |
| `PRO_TIER_CACHE_TTL_ANALYTICS` | `900` | Analytics cache TTL (seconds) |
| `PRO_TIER_NEWS_API_KEY` | `""` | NewsAPI.org API key |
| `PRO_TIER_ALPHA_VANTAGE_KEY` | `""` | Alpha Vantage API key |

## Test Suite

```bash
make lint
make typecheck
make test
```

Use `make test-all` when you need the full suite, including slow and integration
tests.

Tests rely on monkeypatched Yahoo Finance responses, so they run offline and do
not hit the live API.
