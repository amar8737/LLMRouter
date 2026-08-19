from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from .base import BaseScheduler

if TYPE_CHECKING:
    from ..providers.provider_router import ProviderRouter


class RandomScheduler(BaseScheduler):
    async def select(self, provider_router: ProviderRouter) -> Any:
        healthy = await self._healthy_clients(provider_router)
        if not healthy:
            return None
        return random.choice(healthy)
