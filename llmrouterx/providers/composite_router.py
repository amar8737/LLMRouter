from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from ..exceptions import NoHealthyClientError

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

    async def handle(
        self,
        op: str,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:

        last_exception: Exception | None = None

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
                    **kwargs,
                )

                if self.metrics:
                    self.metrics.incr(
                        f"provider.success.{provider_name}"
                    )

                return response

            except asyncio.CancelledError:
                raise

            except NoHealthyClientError as exc:

                logger.debug(
                    "Provider '%s' has no healthy clients.",
                    provider_name,
                )

                last_exception = exc

                if self.metrics:
                    self.metrics.incr(
                        f"provider.no_healthy.{provider_name}"
                    )

                continue

            except Exception as exc:

                logger.exception(
                    "Provider '%s' failed.",
                    provider_name,
                )

                last_exception = exc

                if self.metrics:
                    self.metrics.incr(
                        f"provider.error.{provider_name}"
                    )

                continue

        if last_exception is not None:
            raise last_exception

        raise NoHealthyClientError(
            "No healthy providers available."
        )

    async def health(self) -> dict[str, bool]:
        """Return a ``{provider_name: is_healthy}`` map."""
        result: dict[str, bool] = {}
        for provider in self.providers:
            name = getattr(provider, "name", provider.__class__.__name__)
            try:
                result[name] = await provider.is_healthy()
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

        Failover only happens before the first token is emitted.
        """
        for provider in self.providers:
            name = getattr(provider, "name", provider.__class__.__name__)
            try:
                async for token in provider.stream(prompt, **kwargs):
                    yield token
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Provider '%s' stream failed: %s",
                    name,
                    str(exc),
                )
                continue
        raise NoHealthyClientError("No healthy providers available for streaming.")