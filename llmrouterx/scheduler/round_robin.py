import asyncio
import logging

from .base import BaseScheduler


class RoundRobinScheduler(BaseScheduler):
    def __init__(self):
        self._indices = {}
        self._locks = {}

    async def select(self, provider_router):
        clients = provider_router.clients
        if not clients:
            return None
        # Use provider name if present, otherwise use object id to key rotation state
        key = getattr(provider_router, "name", None) or f"id:{id(provider_router)}"
        # ensure there's a lock per provider
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            idx = self._indices.get(key, 0) % len(clients)
            for i in range(len(clients)):
                c = clients[(idx + i) % len(clients)]
                try:
                    healthy = await c.is_healthy()
                except (AttributeError, RuntimeError, OSError, TypeError) as e:
                    logging.debug("round-robin: client health check failed: %s", e)
                    healthy = False
                if healthy:
                    self._indices[key] = (idx + i + 1) % len(clients)
                    return c
        return None
