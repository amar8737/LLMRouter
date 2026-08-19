from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from .base import BaseScheduler

if TYPE_CHECKING:
    from ..providers.provider_router import ProviderRouter


class LeastBusyScheduler(BaseScheduler):
    """
    Choose a healthy client with the fewest active requests.

    Skips unhealthy and saturated clients.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def select(self, provider_router: ProviderRouter) -> Any:
        async with self._lock:
            healthy_clients = await self._healthy_clients(provider_router)
            if not healthy_clients:
                return None

            best = None
            best_load = float("inf")

            for client in healthy_clients:
                load = client.active_requests
                if load < best_load:
                    best = client
                    best_load = load

            return best
