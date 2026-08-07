from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from .base import BaseScheduler

if TYPE_CHECKING:
    from ..providers.provider_router import ProviderRouter

logger = logging.getLogger(__name__)


class LeastBusyScheduler(BaseScheduler):
    """
    Selects the healthy client with the fewest active requests.

    Skips unhealthy and saturated clients.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def select(self, provider_router: ProviderRouter) -> Any:
        async with self._lock:
            best = None
            best_load = float("inf")

            for client in provider_router.clients:
                try:
                    if not await client.is_healthy():
                        continue

                    if client.is_saturated:
                        continue

                    load = client.active_requests

                    if load < best_load:
                        best = client
                        best_load = load

                except asyncio.CancelledError:
                    raise

                except Exception:
                    logger.exception(
                        "Failed to inspect client '%s'",
                        getattr(client, "api_key", "<unknown>"),
                    )

            return best
