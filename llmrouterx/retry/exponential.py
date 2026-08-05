from __future__ import annotations

import asyncio
import random
from typing import Any


class HTTPError(Exception):
    """Generic HTTP error."""

    def __init__(
        self,
        status_code: int,
        message: str = "HTTP error",
        *,
        headers: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{status_code}: {message}")
        self.status_code = status_code
        self.headers = headers or {}


class ExponentialRetry:
    """
    Exponential backoff retry policy.

    Supports:
    - Retry-After header
    - Exponential backoff
    - Jitter
    - Max backoff
    """

    RETRYABLE_HTTP_STATUS = frozenset(
        {
            429,
            500,
            502,
            503,
            504,
        }
    )

    def __init__(
        self,
        *,
        max_retries: int = 3,
        base: float = 0.5,
        factor: float = 2.0,
        max_backoff: float = 60.0,
        jitter: bool = True,
    ) -> None:
        self.max_retries = max_retries
        self.base = base
        self.factor = factor
        self.max_backoff = max_backoff
        self.jitter = jitter

    def should_retry(
        self,
        exc: Exception,
        attempt: int,
    ) -> bool:
        if attempt >= self.max_retries:
            return False

        if isinstance(exc, HTTPError):
            return exc.status_code in self.RETRYABLE_HTTP_STATUS

        return isinstance(
            exc,
            (
                asyncio.TimeoutError,
                ConnectionError,
            ),
        )

    def get_backoff(
        self,
        exc: Exception,
        attempt: int,
    ) -> float:
        """
        Compute delay before next retry.
        """

        if isinstance(exc, HTTPError):
            retry_after = exc.headers.get("Retry-After")

            if retry_after is not None:
                try:
                    return min(
                        float(retry_after),
                        self.max_backoff,
                    )
                except ValueError:
                    pass

        delay = min(
            self.base * (self.factor ** (attempt - 1)),
            self.max_backoff,
        )

        if self.jitter:
            delay *= random.uniform(0.5, 1.5)

        return delay

    async def wait(
        self,
        exc: Exception,
        attempt: int,
    ) -> None:
        await asyncio.sleep(
            self.get_backoff(
                exc,
                attempt,
            )
        )