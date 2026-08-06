import asyncio
import logging

from ..exceptions import NoHealthyClientError

logger = logging.getLogger(__name__)


class ProviderRouter:
    """
    Routes requests to one of the provider's ClientNodes.
    """

    def __init__(self, name: str, clients: list, scheduler=None):
        self.name = name
        self.clients = clients or []
        self.scheduler = scheduler

    async def is_healthy(self) -> bool:
        """
        Returns True if at least one client is healthy.
        """
        for client in self.clients:
            try:
                if await client.is_healthy():
                    return True
            except Exception as exc:
                logger.exception(
                    "Health check failed for client %s",
                    getattr(client, "api_key", "<unknown>"),
                    exc_info=exc,
                )

        return False

    async def handle(self, op: str, payload: dict, **kwargs) -> dict:
        """
        Route with retry logic for TOCTOU races.
        """

        if self.scheduler is not None:
            last_exception = None
            max_scheduler_retries = 3

            for attempt in range(max_scheduler_retries):
                try:
                    client = await self.scheduler.select(self)
                    
                    if client is None:
                        raise NoHealthyClientError(
                            f"No healthy client for provider '{self.name}'"
                        )
                    
                    return await client.send(op, payload, **kwargs)
                
                except asyncio.CancelledError:
                    raise
                
                except NoHealthyClientError:
                    raise
                
                except Exception as exc:
                    logger.warning(
                        "Client '%s' failed (attempt %d/%d): %s",
                        getattr(client, "api_key", "<unknown>"),
                        attempt + 1,
                        max_scheduler_retries,
                        str(exc),
                    )
                    
                    last_exception = exc
                    
                    if attempt == max_scheduler_retries - 1:
                        raise

        for client in self.clients:
            try:
                if not await client.is_healthy():
                    continue

                return await client.send(op, payload, **kwargs)
            
            except asyncio.CancelledError:
                raise
            
            except Exception:
                logger.exception("Client '%s' failed", client.api_key)
                continue

        raise NoHealthyClientError(
            f"No healthy client available for provider '{self.name}'"
        )

    async def is_healthy(self) -> bool:
        """
        Returns True if at least one client is healthy.
        """
        for client in self.clients:
            try:
                if await client.is_healthy():
                    return True
            except Exception as exc:
                logger.exception(
                    "Health check failed for client %s",
                    getattr(client, "api_key", "<unknown>"),
                    exc_info=exc,
                )

        return False

    async def handle(self, op: str, payload: dict, **kwargs) -> dict:
        """
        Route with retry logic for TOCTOU races.
        """

        if self.scheduler is not None:
            last_exception = None
            max_scheduler_retries = 3

            for attempt in range(max_scheduler_retries):
                try:
                    client = await self.scheduler.select(self)
                    
                    if client is None:
                        raise NoHealthyClientError(
                            f"No healthy client for provider '{self.name}'"
                        )
                    
                    return await client.send(op, payload, **kwargs)
                
                except asyncio.CancelledError:
                    raise
                
                except NoHealthyClientError:
                    raise
                
                except Exception as exc:
                    logger.warning(
                        "Client '%s' failed (attempt %d/%d): %s",
                        getattr(client, "api_key", "<unknown>"),
                        attempt + 1,
                        max_scheduler_retries,
                        str(exc),
                    )
                    
                    last_exception = exc
                    
                    if attempt == max_scheduler_retries - 1:
                        raise

        for client in self.clients:
            try:
                if not await client.is_healthy():
                    continue

                return await client.send(op, payload, **kwargs)
            
            except asyncio.CancelledError:
                raise
            
            except Exception:
                logger.exception("Client '%s' failed", client.api_key)
                continue

        raise NoHealthyClientError(
            f"No healthy client available for provider '{self.name}'"
        )
