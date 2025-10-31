# Fin-MCP: Free-Tier Market Data MCP Server

A high-quality Model Context Protocol (MCP) server providing Yahoo Finance market data without requiring API keys.

## Features

- **Market Movers**: Top gainers, losers, and most active stocks
- **Ticker Quotes**: Real-time price snapshots with technical indicators
- **Price History**: Historical OHLCV candles for technical analysis
- **Multi-Transport**: STDIO, HTTP, and SSE support
- **Type-Safe**: Full type hints with Pydantic validation
- **Well-Tested**: Comprehensive test suite with async support

## Tools

- `get_market_movers` – Top gainers, losers, or most active equities
- `get_ticker_data` – Real-time quote snapshot for a single symbol
- `get_price_history` – Historical OHLCV candles for technical analysis

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

## Running the Free Tier server

```bash
source .venv/bin/activate
fastmcp run src/servers/free_tier/server.py
```

To iterate with the MCP Inspector UI instead:

```bash
source .venv/bin/activate
fastmcp dev src/servers/free_tier/server.py
```

Pre-built Inspector scripts covering success, validation, and failure paths for
each tool live under `scripts/mcp_inspector/free_tier/`.

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
