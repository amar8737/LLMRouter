import random
from .base import BaseScheduler


class WeightedScheduler(BaseScheduler):
    """Choose a healthy client randomly weighted by `weight` attribute (default 1)."""

    async def select(self, provider_router):
        clients = provider_router.clients
        if not clients:
            return None
        weighted = []
        for c in clients:
            try:
                if await c.is_healthy():
                    w = getattr(c, "weight", 1) or 1
                    weighted.append((c, float(w)))
            except Exception:
                continue
        if not weighted:
            return None
        total = sum(w for _, w in weighted)
        r = random.random() * total
        upto = 0.0
        for c, w in weighted:
            upto += w
            if r <= upto:
                return c
        return weighted[-1][0]
