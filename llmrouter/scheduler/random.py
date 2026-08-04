import random
from .base import BaseScheduler


class RandomScheduler(BaseScheduler):
    async def select(self, provider_router):
        healthy = [c for c in provider_router.clients if await c.is_healthy()]
        if not healthy:
            return None
        return random.choice(healthy)
