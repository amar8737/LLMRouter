from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from ..exceptions import ConfigurationError
from .base import BaseScheduler

if TYPE_CHECKING:
    from ..providers.provider_router import ProviderRouter


class WeightedScheduler(BaseScheduler):
    """Choose a healthy client randomly weighted by `weight` attribute (default 1)."""

    async def select(self, provider_router: ProviderRouter) -> Any:
        healthy_clients = await self._healthy_clients(provider_router)
        if not healthy_clients:
            return None

        weighted = []
        for c in healthy_clients:
            w = getattr(c, "weight", 1)
            if w < 0:
                raise ConfigurationError(f"Negative weight {w} for client {c}")
            if w == 0:
                continue
            weighted.append((c, float(w)))

        if not weighted:
            return None
        total = sum(w for _, w in weighted)
        if total <= 0:
            return None
        r = random.random() * total
        upto = 0.0
        for c, w in weighted:
            upto += w
            if r <= upto:
                return c
        return weighted[-1][0]
