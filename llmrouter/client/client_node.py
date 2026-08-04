import asyncio
from datetime import datetime, timedelta


class ClientNode:
    """Represents a single API key / client instance with basic health and concurrency control."""

    FAILURE_THRESHOLD = 5
    COOLDOWN_SECONDS = 30

    def __init__(self, api_key: str, client, max_concurrent: int = 100):
        self.api_key = api_key
        self.client = client
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)

        self.active_requests = 0
        self.last_success = None
        self.last_failure = None
        self.failures = 0
        self.cooldown_until = None

    async def is_healthy(self) -> bool:
        async with self._lock:
            if self.cooldown_until and datetime.utcnow() < self.cooldown_until:
                return False
            return self.failures < self.FAILURE_THRESHOLD

    async def _record_success(self):
        async with self._lock:
            self.last_success = datetime.utcnow()
            # On success, fully recover failure count and clear cooldown so
            # healthy clients rapidly return to service.
            self.failures = 0
            self.cooldown_until = None

    async def _record_failure(self):
        async with self._lock:
            self.last_failure = datetime.utcnow()
            self.failures += 1
            if self.failures >= self.FAILURE_THRESHOLD:
                self.cooldown_until = datetime.utcnow() + timedelta(seconds=self.COOLDOWN_SECONDS)

    async def send(self, op: str, prompt: str, **kwargs):
        # enforce max concurrent requests per key
        async with self._semaphore:
            async with self._lock:
                self.active_requests += 1

            try:
                resp = await self.client.request(op, prompt, api_key=self.api_key, **kwargs)
                await self._record_success()
                return resp
            except Exception as e:
                # Do not treat cancellation as a failure metric — re-raise immediately
                if isinstance(e, asyncio.CancelledError):
                    raise
                await self._record_failure()
                raise
            finally:
                async with self._lock:
                    self.active_requests = max(0, self.active_requests - 1)
