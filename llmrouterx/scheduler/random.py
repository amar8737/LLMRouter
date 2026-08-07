from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from .base import BaseScheduler

if TYPE_CHECKING:
    from ..providers.provider_router import ProviderRouter


class RandomScheduler(BaseScheduler):
    async def select(self, provider_router: ProviderRouter) -> Any:
        healthy = [c for c in provider_router.clients if await c.is_healthy()]
        if not healthy:
            return None
        return random.choice(healthy)
