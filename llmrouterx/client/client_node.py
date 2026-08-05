import asyncio
from datetime import datetime, timedelta
from typing import Any


class ClientNode:
    """
    Represents a single API key/client.

    Responsibilities:
    - Health tracking
    - Failure tracking
    - Cooldown management
    - Concurrency limiting
    - Sending requests
    """

    FAILURE_THRESHOLD = 5
    COOLDOWN_SECONDS = 30

    def __init__(
        self,
        api_key: str,
        client: Any,
        *,
        max_concurrent: int = 100,
        timeout: float | None = None,
    ):
        self.api_key = api_key
        self.client = client

        self.timeout = timeout

        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)

        self.active_requests = 0

        self.failures = 0
        self.last_success: datetime | None = None
        self.last_failure: datetime | None = None
        self.cooldown_until: datetime | None = None

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def is_healthy(self) -> bool:
        async with self._lock:
            now = datetime.now(datetime.timezone.utc)

            # Recover automatically after cooldown
            if self.cooldown_until is not None and now >= self.cooldown_until:
                self.cooldown_until = None
                self.failures = 0

            return self.cooldown_until is None

    # ------------------------------------------------------------------
    # Reservation API (future schedulers)
    # ------------------------------------------------------------------

    async def acquire(self):
        """
        Reserve one concurrency slot.
        """
        await self._semaphore.acquire()

        async with self._lock:
            self.active_requests += 1

    async def release(self):
        """
        Release one concurrency slot.
        """
        async with self._lock:
            self.active_requests = max(0, self.active_requests - 1)

        self._semaphore.release()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def _record_success(self):
        async with self._lock:
            self.failures = 0
            self.cooldown_until = None
            self.last_success = datetime.now(datetime.timezone.utc)

    async def _record_failure(self):
        async with self._lock:
            self.failures += 1
            self.last_failure = datetime.now(datetime.timezone.utc)

            if self.failures >= self.FAILURE_THRESHOLD:
                self.cooldown_until = datetime.now(datetime.timezone.utc) + timedelta(
                    seconds=self.COOLDOWN_SECONDS
                )

    # ------------------------------------------------------------------
    # Request
    # ------------------------------------------------------------------

    async def send(self, op: str, payload: dict, **kwargs):
        """
        Send request to underlying provider.
        """

        await self.acquire()

        try:
            coro = self.client.request(
                op,
                payload,
                api_key=self.api_key,
                **kwargs,
            )

            if self.timeout:
                response = await asyncio.wait_for(
                    coro,
                    timeout=self.timeout,
                )
            else:
                response = await coro

            await self._record_success()

            return response

        except asyncio.CancelledError:
            raise

        except Exception:
            await self._record_failure()
            raise

        finally:
            await self.release()
