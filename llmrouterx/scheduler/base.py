from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..providers.provider_router import ProviderRouter


class BaseScheduler(ABC):
    @abstractmethod
    async def select(self, provider_router: ProviderRouter) -> Any:
        """Return a ClientNode or None"""
