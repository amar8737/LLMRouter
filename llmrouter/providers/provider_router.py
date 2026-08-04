import logging

from ..exceptions import NoHealthyClientError


class ProviderRouter:
    """Manages a single provider and its ClientNode instances."""

    def __init__(self, name: str, clients: list, scheduler=None):
        self.name = name
        self.clients = clients or []
        self.scheduler = scheduler

    async def is_healthy(self) -> bool:
        for c in self.clients:
            try:
                if await c.is_healthy():
                    return True
            except (AttributeError, RuntimeError, OSError, TypeError) as e:
                logging.debug("client health check failed for %s: %s", getattr(c, "api_key", "<client>"), e)
                continue
        return False

    async def handle(self, op: str, *args, **kwargs):
        # Use attached scheduler if available
        if self.scheduler is not None:
            client = await self.scheduler.select(self)
            if client:
                return await client.send(op, *args, **kwargs)
            raise NoHealthyClientError(f"No healthy clients for provider {self.name}")

        # Fallback simple scheduler: pick first healthy client
        for c in self.clients:
            if await c.is_healthy():
                return await c.send(op, *args, **kwargs)
        raise NoHealthyClientError(f"No healthy clients for provider {self.name}")
