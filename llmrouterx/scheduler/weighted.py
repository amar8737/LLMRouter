from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

from ..exceptions import ConfigurationError
from .base import BaseScheduler

if TYPE_CHECKING:
    from ..providers.provider_router import ProviderRouter


class WeightedScheduler(BaseScheduler):
    """Choose a healthy client randomly weighted by `weight` attribute (default 1)."""

    async def select(self, provider_router: ProviderRouter) -> Any:
        clients = provider_router.clients
        if not clients:
            return None
        weighted = []
        for c in clients:
            try:
                if await c.is_healthy() and not getattr(c, "is_saturated", False):
                    w = getattr(c, "weight", 1)
                    if w < 0:
                        raise ConfigurationError(
                            f"Negative weight {w} for client {c}"
                        )
                    if w == 0:
                        # Zero-weight clients are excluded from selection.
                        continue
                    weighted.append((c, float(w)))
            except (AttributeError, RuntimeError, OSError, TypeError) as e:
                logging.debug("weighted scheduler: skipping client due to error: %s", e)
                continue
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
