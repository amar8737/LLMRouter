from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from .base import BaseScheduler

if TYPE_CHECKING:
    from ..providers.provider_router import ProviderRouter


class RoundRobinScheduler(BaseScheduler):
    def __init__(self) -> None:
        self._indices: dict[str, int] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def select(self, provider_router: ProviderRouter) -> Any:
        healthy_clients = await self._healthy_clients(provider_router)
        if not healthy_clients:
            return None

        # Use provider name if present, otherwise use object id to key rotation state
        key = getattr(provider_router, "name", None) or f"id:{id(provider_router)}"
        lock = self._locks.setdefault(key, asyncio.Lock())

        async with lock:
            idx = self._indices.get(key, 0) % len(healthy_clients)
            for i in range(len(healthy_clients)):
                c = healthy_clients[(idx + i) % len(healthy_clients)]
                self._indices[key] = (idx + i + 1) % len(healthy_clients)
                return c
        return None
