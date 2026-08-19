from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from ..exceptions import NoHealthyClientError
from ..utils.masking import mask_api_key

if TYPE_CHECKING:
    from ..context.request_context import RequestContext

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
        self.last_api_key: str | None = None

    async def is_healthy(self) -> bool:
        """Returns True if at least one client is healthy."""
        for client in self.clients:
            try:
                if await client.is_healthy():
                    return True
            except Exception:
                logger.exception(
                    "Health check failed for client %s",
                    mask_api_key(getattr(client, "api_key", None)),
                )
        return False

    async def handle(
        self,
        op: str,
        payload: dict[str, Any],
        *,
        context: RequestContext | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Route a request to a healthy client, with per-client error isolation.

        When a scheduler is configured it is consulted up to three times as a
        TOCTOU guard (a client can die between selection and dispatch). If the
        scheduler path is exhausted, we fall through to a linear scan of any
        remaining healthy clients so a bad scheduler pick cannot starve a
        provider that still has working clients.

        A single client's failure never aborts the whole provider: the next
        healthy client is tried. If every client fails, the last original
        exception is re-raised (not a ``NoHealthyClientError``) so the
        router-level retry policy can still classify and retry it.
        """
        last_exception: Exception | None = None
        attempted: set[Any] = set()

        if self.scheduler is not None:
            max_scheduler_retries = 3

            for attempt in range(max_scheduler_retries):
                client = None
                try:
                    client = await self.scheduler.select(self)

                    if client is None:
                        raise NoHealthyClientError(f"No healthy client for provider '{self.name}'")

                    attempted.add(client)
                    return await self._dispatch(client, op, payload, context, kwargs)

                except asyncio.CancelledError:
                    raise

                except NoHealthyClientError:
                    # Scheduler returned no healthy client; fall through to
                    # the linear scan so a bad scheduler pick cannot starve
                    # a provider that still has working clients.
                    pass

                except Exception as exc:
                    last_exception = exc
                    logger.warning(
                        "Client '%s' failed (attempt %d/%d): %s",
                        mask_api_key(getattr(client, "api_key", None)),
                        attempt + 1,
                        max_scheduler_retries,
                        str(exc),
                    )

        for client in self.clients:
            if client in attempted:
                continue

            try:
                if not await client.is_healthy():
                    continue
                return await self._dispatch(client, op, payload, context, kwargs)

            except asyncio.CancelledError:
                raise

            except NoHealthyClientError:
                raise

            except Exception as exc:
                last_exception = exc
                logger.warning(
                    "Client '%s' failed: %s",
                    mask_api_key(getattr(client, "api_key", None)),
                    str(exc),
                )

        if last_exception is not None:
            raise last_exception

        raise NoHealthyClientError(f"No healthy client available for provider '{self.name}'")

    async def _dispatch(
        self,
        client: Any,
        op: str,
        payload: dict[str, Any],
        context: RequestContext | None,
        kwargs: dict[str, Any],
    ) -> Any:
        """Send to a client and record attribution for the request context."""
        result = await client.send(op, payload, context=context, **kwargs)
        self.last_api_key = getattr(client, "api_key", None)
        if context is not None:
            context.api_key = self.last_api_key
        return result

    async def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream from the first healthy client.

        Failover only happens *before* the first token is emitted. Once a
        token has been delivered the stream is committed to that client;
        a mid-stream failure is surfaced to the caller instead of re-routing.
        """
        for client in self.clients:
            started = False
            try:
                if not await client.is_healthy():
                    continue

                async for token in client.stream(prompt, **kwargs):
                    started = True
                    yield token
                return

            except asyncio.CancelledError:
                raise

            except Exception:
                if started:
                    raise
                logger.exception(
                    "Stream failed before first token for client %s",
                    mask_api_key(getattr(client, "api_key", None)),
                )
                continue

        raise NoHealthyClientError(
            f"No healthy client available for streaming on provider '{self.name}'"
        )
