from __future__ import annotations

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
        healthy_clients = await self._healthy_clients(provider_router)
        if not healthy_clients:
            return None

        candidates = []
        for c in healthy_clients:
            pr = getattr(c, "priority", 100)
            candidates.append((pr, c))

        # sort by priority asc, then by active_requests asc
        candidates.sort(key=lambda pc: (pc[0], getattr(pc[1], "active_requests", 0)))
        return candidates[0][1]
