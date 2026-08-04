import logging

from .base import BaseScheduler


class PriorityScheduler(BaseScheduler):
    """Select the healthiest client with the highest priority.

    Clients may expose a `priority` attribute (lower number == higher priority).
    Among equal priority, choose the least-busy client.
    """

    async def select(self, provider_router):
        clients = provider_router.clients
        if not clients:
            return None
        candidates = []
        for c in clients:
            try:
                if await c.is_healthy():
                    pr = getattr(c, "priority", 100)
                    candidates.append((pr, c))
            except (AttributeError, RuntimeError, OSError, TypeError) as e:
                logging.debug("skipping client in priority scheduler due to error: %s", e)
                continue
        if not candidates:
            return None
        # sort by priority asc, then by active_requests asc
        candidates.sort(key=lambda pc: (pc[0], getattr(pc[1], "active_requests", 0)))
        return candidates[0][1]
