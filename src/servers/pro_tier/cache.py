"""Redis caching utilities for Pro Tier server.

Provides a simple caching layer with TTL support to reduce load on external
sentiment and analytics APIs. Handles connection failures gracefully.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class CacheClient:
    """Simple Redis cache client with graceful degradation."""

    def __init__(self, redis_url: str) -> None:
        """Initialize Redis client.

        Args:
            redis_url: Redis connection URL (e.g., 'redis://localhost:6379/0')
        """
        self.redis_url = redis_url
        self._client: Any = None
        self._available = False
        self._init_client()

    def _init_client(self) -> None:
        """Attempt to initialize Redis client, degrading gracefully on failure."""
        try:
            import redis

            self._client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            # Test connection
            self._client.ping()
            self._available = True
            logger.info(f"Redis cache connected: {self.redis_url}")
        except Exception as exc:
            logger.warning(f"Redis cache unavailable, operating without cache: {exc}")
            self._available = False

    def get(self, key: str) -> Any | None:
        """Retrieve cached value by key.

        Args:
            key: Cache key

        Returns:
            Deserialized cached value or None if not found/unavailable
        """
        if not self._available:
            return None

        try:
            value = self._client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except Exception as exc:
            logger.warning(f"Cache read failed for key={key}: {exc}")
            return None

    def set(self, key: str, value: Any, ttl: int) -> None:
        """Store value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl: Time-to-live in seconds
        """
        if not self._available:
            return

        try:
            serialized = json.dumps(value)
            self._client.setex(key, ttl, serialized)
        except Exception as exc:
            logger.warning(f"Cache write failed for key={key}: {exc}")

    def delete(self, key: str) -> None:
        """Delete cached value.

        Args:
            key: Cache key to delete
        """
        if not self._available:
            return

        try:
            self._client.delete(key)
        except Exception as exc:
            logger.warning(f"Cache delete failed for key={key}: {exc}")

    def is_available(self) -> bool:
        """Check if cache is available.

        Returns:
            True if Redis is connected and responding
        """
        return self._available
