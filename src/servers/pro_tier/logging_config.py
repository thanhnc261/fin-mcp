"""Logging configuration for the Pro Tier MCP server.

Provides structured logging with configurable levels and formats optimized
for debugging sentiment analysis and analytics operations.
"""

from __future__ import annotations

import logging
import sys

from .config import ProTierSettings


def configure_logging(settings: ProTierSettings) -> None:
    """Configure logging for the Pro Tier server.

    Sets up structured logging with the specified verbosity level and format.
    Logs are written to stderr to avoid interfering with MCP protocol on stdout.

    Args:
        settings: Pro Tier configuration containing log level
    """
    # Map string level to logging constant
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,  # Keep stdout clean for MCP protocol
        force=True,  # Override any existing configuration
    )

    # Silence noisy third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Pro Tier logging configured: level={settings.log_level}")
