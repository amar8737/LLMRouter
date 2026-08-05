from __future__ import annotations

import asyncio
import logging
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

        raise RuntimeError(
            "No providers configured."
        )