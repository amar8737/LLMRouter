from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod


class BaseRetry(ABC):
    """
    Base class for retry policies.

    A retry policy answers two questions:

    - Should this exception be retried at this attempt number?
    - How long should the router wait before the next attempt?

    ``attempt`` is 1-based: it is ``1`` immediately after the first failure.
    """

    @abstractmethod
    def should_retry(
        self,
        exc: Exception,
        attempt: int,
    ) -> bool:
        """
        Return ``True`` if the router should try again.
        """
        raise NotImplementedError

    def get_backoff(
        self,
        exc: Exception,
        attempt: int,
    ) -> float:
        """
        Return the delay in seconds before the next attempt.

        Defaults to no delay so that simple policies only need to
        implement :meth:`should_retry`.
        """
        return 0.0

    async def wait(
        self,
        exc: Exception,
        attempt: int,
    ) -> None:
        """
        Sleep for :meth:`get_backoff` seconds.
        """
        delay = self.get_backoff(exc, attempt)

        if delay > 0:
            await asyncio.sleep(delay)
