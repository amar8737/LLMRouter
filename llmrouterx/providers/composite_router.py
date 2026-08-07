from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from ..exceptions import NoHealthyClientError

if TYPE_CHECKING:
    from ..context.request_context import RequestContext

logger = logging.getLogger(__name__)


class CompositeRouter:
    """
    Routes requests across multiple providers.

    Strategy:
        Provider 1
            ↓
        Provider 2
            ↓
        Provider 3
    """

    def __init__(
        self,
        providers: list[Any],
        metrics: Any | None = None,
    ) -> None:
        self.providers = providers
        self.metrics = metrics
        self.last_provider: str | None = None

    @property
    def last_api_key(self) -> str | None:
        """API key of the provider that most recently handled a request."""
        if self.last_provider is None:
            return None
        for provider in self.providers:
            if getattr(provider, "name", None) == self.last_provider:
                return getattr(provider, "last_api_key", None)
        return None

    async def handle(
        self,
        op: str,
        payload: dict[str, Any],
        *,
        context: RequestContext | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:

        exceptions: list[Exception] = []

        for provider in self.providers:
            provider_name = getattr(
                provider,
                "name",
                provider.__class__.__name__,
            )

            try:
                response = await provider.handle(
                    op,
                    payload,
                    context=context,
                    **kwargs,
                )

                self.last_provider = provider_name
                if context is not None:
                    context.provider = provider_name

                if self.metrics:
                    self.metrics.incr(f"provider.success.{provider_name}")

                return response

            except asyncio.CancelledError:
                raise

            except NoHealthyClientError as exc:
                logger.debug(
                    "Provider '%s' has no healthy clients.",
                    provider_name,
                )

                exceptions.append(exc)

                if self.metrics:
                    self.metrics.incr(f"provider.no_healthy.{provider_name}")

                continue

            except Exception as exc:
                logger.exception(
                    "Provider '%s' failed.",
                    provider_name,
                )

                exceptions.append(exc)

                if self.metrics:
                    self.metrics.incr(f"provider.error.{provider_name}")

                continue

        # Surface the full failure sequence. If any underlying error is
        # transient the router's retry policy unwraps it (see LLMRouter), so
        # wrapping here does not prevent transient errors from being retried.
        raise NoHealthyClientError(
            "All providers failed to serve the request.",
            errors=exceptions,
        )

    async def health(
        self,
        *,
        timeout: float | None = 5.0,
    ) -> dict[str, bool]:
        """Return a ``{provider_name: is_healthy}`` map."""
        result: dict[str, bool] = {}
        for provider in self.providers:
            name = getattr(provider, "name", provider.__class__.__name__)
            try:
                coro = provider.is_healthy()
                if timeout is not None:
                    coro = asyncio.wait_for(coro, timeout=timeout)
                result[name] = await coro
            except Exception:
                result[name] = False
        return result

    async def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Stream from the first healthy provider.

        Failover only happens *before* the first token is emitted. Once a
        token has been delivered the stream is committed to that provider;
        a mid-stream failure is surfaced to the caller instead of silently
        re-routing, which would interleave output from two providers.
        """
        for provider in self.providers:
            name = getattr(provider, "name", provider.__class__.__name__)
            started = False
            try:
                async for token in provider.stream(prompt, **kwargs):
                    started = True
                    yield token
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if started:
                    raise
                logger.warning(
                    "Provider '%s' stream failed before first token: %s",
                    name,
                    str(exc),
                )
                continue
        raise NoHealthyClientError("No healthy providers available for streaming.")
