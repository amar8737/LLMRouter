from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..client.client_node import ClientNode
    from ..providers.provider_router import ProviderRouter


class BaseScheduler(ABC):
    """Base class for provider client schedulers."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def select(self, provider_router: ProviderRouter) -> ClientNode | None:
        """Return a ClientNode or None"""

    async def _healthy_clients(self, provider_router: ProviderRouter) -> list[ClientNode]:
        """Return list of healthy, non-saturated clients.

        Subclasses should use this helper to filter clients consistently.
        """
        healthy: list[ClientNode] = []
        for client in provider_router.clients:
            try:
                if await client.is_healthy() and not getattr(client, "is_saturated", False):
                    healthy.append(client)
            except (AttributeError, RuntimeError, OSError, TypeError) as e:
                self.logger.debug("Scheduler health check failed for client %s: %s", client, e)
                continue
        return healthy
