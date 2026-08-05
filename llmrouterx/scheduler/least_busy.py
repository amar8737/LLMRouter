from .base import BaseScheduler
import asyncio


class LeastBusyScheduler(BaseScheduler):
    def __init__(self):
        self._lock = asyncio.Lock()

    async def select(self, provider_router):
        async with self._lock:
            candidates = [c for c in provider_router.clients if await c.is_healthy()]
            if not candidates:
                return None
            # choose client with fewest active requests (snapshot under lock)
            candidates.sort(key=lambda c: getattr(c, "active_requests", 0))
            return candidates[0]
