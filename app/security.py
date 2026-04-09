from collections import deque
from dataclasses import dataclass
from threading import Lock
from time import time

from fastapi import HTTPException, Request, status

from .config import RATE_LIMIT_PER_MINUTE


@dataclass
class RateLimitWindow:
    entries: deque[float]


class InMemoryRateLimiter:
    def __init__(self, limit_per_minute: int) -> None:
        self.limit_per_minute = max(0, limit_per_minute)
        self.window_seconds = 60
        self._buckets: dict[str, RateLimitWindow] = {}
        self._lock = Lock()

    def check(self, key: str) -> None:
        if self.limit_per_minute <= 0:
            return

        now = time()
        cutoff = now - self.window_seconds

        with self._lock:
            bucket = self._buckets.setdefault(key, RateLimitWindow(entries=deque()))
            while bucket.entries and bucket.entries[0] < cutoff:
                bucket.entries.popleft()

            if len(bucket.entries) >= self.limit_per_minute:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later.",
                )

            bucket.entries.append(now)


rate_limiter = InMemoryRateLimiter(RATE_LIMIT_PER_MINUTE)


def build_rate_limit_key(request: Request, user_id: str, bucket: str) -> str:
    client_host = request.client.host if request.client else "unknown"
    return f"{bucket}:{user_id}:{client_host}"
