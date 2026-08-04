from abc import ABC, abstractmethod


class BaseScheduler(ABC):
    @abstractmethod
    async def select(self, provider_router):
        """Return a ClientNode or None"""
