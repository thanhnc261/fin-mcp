from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools import Tool
from pydantic import BaseModel

from .tools import get_market_movers, get_price_history, get_ticker_data

logger = logging.getLogger(__name__)


def _serialize_tool_result(result: Any) -> Any:
    """Convert tool results into JSON serialisable primitives."""

    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")

    if isinstance(result, list):
        return [_serialize_tool_result(item) for item in result]

    if isinstance(result, dict):
        return {key: _serialize_tool_result(value) for key, value in result.items()}

    return result


app = FastMCP(
    name="free-tier-market-data",
    instructions=(
        "Tools that provide Yahoo Finance market data suitable for Claude's free tier, "
        "including market movers, ticker snapshots, and historical prices."
    ),
    version="0.1.0",
    tool_serializer=_serialize_tool_result,
)

app.add_tool(Tool.from_function(get_market_movers))
app.add_tool(Tool.from_function(get_ticker_data))
app.add_tool(Tool.from_function(get_price_history))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger.info("Starting Free Tier market data server")
    app.run()


if __name__ == "__main__":
    main()
