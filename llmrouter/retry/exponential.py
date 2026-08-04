import asyncio
import time
from typing import Dict, Optional


class HTTPError(Exception):
    def __init__(self, status_code: int, message: str = "HTTP error", headers: Optional[Dict] = None):
        super().__init__(f"{status_code}: {message}")
        self.status_code = status_code
        self.headers = headers or {}


class ExponentialRetry:
    def __init__(self, max_retries: int = 3, base: float = 0.5, factor: float = 2.0, max_backoff: float = 60.0):
        self.max_retries = max_retries
        self.base = base
        self.factor = factor
        self.max_backoff = max_backoff

    def should_retry(self, exc: Exception, attempt: int) -> bool:
        if attempt >= self.max_retries:
            return False

        # HTTP retryable status codes
        if isinstance(exc, HTTPError):
            return exc.status_code in (429, 500, 502, 503, 504)

        # Network/timeouts
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True

        return False

    def get_backoff(self, exc: Exception, attempt: int) -> float:
        # If HTTP 429 and Retry-After header present, honor it
        if isinstance(exc, HTTPError) and exc.status_code == 429:
            retry_after = exc.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(float(retry_after), self.max_backoff)
                except Exception:
                    pass

        # jitterless exponential backoff (attempt is 1-based)
        backoff = self.base * (self.factor ** (attempt - 1))
        return min(backoff, self.max_backoff)

    async def wait(self, exc: Exception, attempt: int):
        await asyncio.sleep(self.get_backoff(exc, attempt))
