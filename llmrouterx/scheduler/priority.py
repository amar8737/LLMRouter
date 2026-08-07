from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .base import BaseScheduler

if TYPE_CHECKING:
    from ..providers.provider_router import ProviderRouter


class PriorityScheduler(BaseScheduler):
    """Select the healthiest client with the highest priority.

    Clients may expose a `priority` attribute (lower number == higher priority).
    Among equal priority, choose the least-busy client.
    """

    async def select(self, provider_router: ProviderRouter) -> Any:
        clients = provider_router.clients
        if not clients:
            return None
        candidates = []
        for c in clients:
            try:
                if await c.is_healthy() and not getattr(c, "is_saturated", False):
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
