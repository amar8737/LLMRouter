import asyncio
import math


class ExponentialRetry:
    def __init__(self, max_retries: int = 3, base: float = 0.5, factor: float = 2.0, max_backoff: float = 10.0):
        self.max_retries = max_retries
        self.base = base
        self.factor = factor
        self.max_backoff = max_backoff

    def should_retry(self, exc: Exception, attempt: int) -> bool:
        return attempt < self.max_retries

    def get_backoff(self, attempt: int) -> float:
        # jitterless exponential backoff
        backoff = self.base * (self.factor ** attempt)
        return min(backoff, self.max_backoff)

    async def wait(self, attempt: int):
        await asyncio.sleep(self.get_backoff(attempt))
