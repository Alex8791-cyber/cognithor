"""Jarvis · Rate-Limiter Middleware.

Einfacher Token-Bucket-Algorithmus fuer API-Endpoints.
"""
from __future__ import annotations

import time
import asyncio
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """Token-Bucket fuer einen Client."""
    rate: float          # tokens per second
    capacity: float      # max burst
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self):
        self.tokens = self.capacity
        self.last_refill = time.monotonic()

    def consume(self, tokens: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimiter:
    """In-Memory Rate-Limiter mit per-IP Buckets."""

    def __init__(
        self,
        rate: float = 10.0,       # requests per second
        capacity: float = 50.0,   # burst capacity
        cleanup_interval: float = 300.0,  # cleanup every 5 min
    ) -> None:
        self._rate = rate
        self._capacity = capacity
        self._cleanup_interval = cleanup_interval
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()
        self._last_cleanup = time.monotonic()

    async def check(self, client_id: str) -> bool:
        async with self._lock:
            now = time.monotonic()
            if now - self._last_cleanup > self._cleanup_interval:
                self._cleanup(now)
                self._last_cleanup = now

            bucket = self._buckets.get(client_id)
            if bucket is None:
                bucket = TokenBucket(rate=self._rate, capacity=self._capacity)
                self._buckets[client_id] = bucket
            return bucket.consume()

    def _cleanup(self, now: float) -> None:
        stale = [k for k, v in self._buckets.items()
                 if now - v.last_refill > self._cleanup_interval]
        for k in stale:
            del self._buckets[k]
