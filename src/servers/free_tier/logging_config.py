"""Centralized logging configuration for the MCP server.

This module provides a single source of truth for logging setup, ensuring
consistent logging behavior across all entry points (CLI, tests, imports).
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure structured logging for the MCP server.

    Sets up logging with:
    - Configurable log level from settings
    - Consistent format with timestamps, levels, and logger names
    - Stream handler to stderr for proper MCP communication
    - UTF-8 encoding for international symbols

    Args:
        settings: Application settings containing log_level configuration
    """
    # Parse log level from settings (e.g., "INFO", "DEBUG")
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,  # MCP uses stdout for protocol, stderr for logs
        force=True,  # Override any existing configuration
    )

    # Set specific log levels for noisy third-party libraries
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.

    This is a convenience wrapper around logging.getLogger() that ensures
    consistent logger naming conventions across the application.

    Args:
        name: Logger name, typically __name__ from the calling module

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
