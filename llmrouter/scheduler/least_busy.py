from .base import BaseScheduler


class LeastBusyScheduler(BaseScheduler):
    async def select(self, provider_router):
        candidates = [c for c in provider_router.clients if await c.is_healthy()]
        if not candidates:
            return None
        # choose client with fewest active requests
        candidates.sort(key=lambda c: getattr(c, "active_requests", 0))
        return candidates[0]
