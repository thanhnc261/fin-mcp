# Free Tier Market Data MCP Tools

This repository hosts a FastMCP server that exposes three free-tier market data
tools backed by Yahoo Finance:

- `get_market_movers` – top gainers, losers, or most active equities
- `get_ticker_data` – real-time quote snapshot for a single symbol
- `get_price_history` – historical OHLCV candles for a given ticker

Each tool performs strict validation and returns descriptive errors when a
request cannot be fulfilled. The server is packaged for use with Claude Desktop
or the MCP Inspector.

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
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
source .venv/bin/activate
pytest
```

Tests rely on monkeypatched Yahoo Finance responses, so they run offline and do
not hit the live API.
