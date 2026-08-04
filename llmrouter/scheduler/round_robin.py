from .base import BaseScheduler


class RoundRobinScheduler(BaseScheduler):
    def __init__(self):
        self._indices = {}

    async def select(self, provider_router):
        clients = provider_router.clients
        if not clients:
            return None
        # Use provider name if present, otherwise use object id to key rotation state
        key = getattr(provider_router, "name", None) or f"id:{id(provider_router)}"
        idx = self._indices.get(key, 0) % len(clients)
        for i in range(len(clients)):
            c = clients[(idx + i) % len(clients)]
            try:
                healthy = await c.is_healthy()
            except Exception:
                healthy = False
            if healthy:
                self._indices[key] = (idx + i + 1) % len(clients)
                return c
        return None
