from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from ..exceptions import NoHealthyClientError

logger = logging.getLogger(__name__)


class ProviderRouter:
    """
    Routes requests to one of the provider's ClientNodes.
    """

    def __init__(
        self,
        name: str,
        clients: list[Any],
        scheduler: Any | None = None,
    ) -> None:
        self.name = name
        self.clients = clients or []
        self.scheduler = scheduler

    async def is_healthy(self) -> bool:
        """Returns True if at least one client is healthy."""
        for client in self.clients:
            try:
                if await client.is_healthy():
                    return True
            except Exception:
                logger.exception(
                    "Health check failed for client %s",
                    getattr(client, "api_key", "<unknown>"),
                )
        return False

    async def handle(
        self,
        op: str,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Route with retry logic for TOCTOU races.
        """
        if self.scheduler is not None:
            max_scheduler_retries = 3

            for attempt in range(max_scheduler_retries):
                client = None
                try:
                    client = await self.scheduler.select(self)

                    if client is None:
                        raise NoHealthyClientError(f"No healthy client for provider '{self.name}'")

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

                    if attempt == max_scheduler_retries - 1:
                        raise

                    # Back off briefly before re-selecting so a transient
                    # failure does not hammer the same client.
                    await asyncio.sleep(0.1 * (attempt + 1))

        for client in self.clients:
            if not await client.is_healthy():
                continue

            return await client.send(op, payload, **kwargs)

        raise NoHealthyClientError(f"No healthy client available for provider '{self.name}'")

    async def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream from the first healthy client."""
        for client in self.clients:
            try:
                if not await client.is_healthy():
                    continue

                async for token in client.stream(prompt, **kwargs):
                    yield token
                return

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception(
                    "Stream failed for client %s",
                    getattr(client, "api_key", "<unknown>"),
                )
                continue

        raise NoHealthyClientError(
            f"No healthy client available for streaming on provider '{self.name}'"
        )
