from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from typing import Any

from ..client.client_node import ClientNode
from ..config.config import RouterConfig
from ..metrics.metrics import MetricsCollector
from ..middleware.base import BaseMiddleware
from ..providers.composite_router import CompositeRouter
from ..providers.provider_router import ProviderRouter
from ..retry.exponential import ExponentialRetry

logger = logging.getLogger(__name__)


class LLMRouter:
    """
    Public interface for all LLM operations.
    """

    def __init__(
        self,
        composite_router: CompositeRouter,
        *,
        retry: ExponentialRetry | None = None,
        metrics: MetricsCollector | None = None,
        middleware: list[BaseMiddleware] | None = None,
        max_retries: int = 3,
    ) -> None:
        self._router = composite_router

        self.retry = retry or ExponentialRetry(
            max_retries=max_retries,
        )

        self.metrics = metrics or MetricsCollector()

        self.middleware = middleware or []

    # --------------------------------------------------------
    # Middleware
    # --------------------------------------------------------

    async def _before(
        self,
        op: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:

        for middleware in self.middleware:
            payload = (
                await middleware.before_request(
                    op,
                    payload,
                )
                or payload
            )

        return payload

    async def _after(
        self,
        op: str,
        payload: dict[str, Any],
        response: dict[str, Any],
    ) -> dict[str, Any]:

        for middleware in self.middleware:
            response = (
                await middleware.after_response(
                    op,
                    payload,
                    response,
                )
                or response
            )

        return response

    # --------------------------------------------------------
    # Core routing
    # --------------------------------------------------------

    async def _execute(
        self,
        op: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:

        payload = await self._before(
            op,
            payload,
        )

        started = time.perf_counter()

        attempt = 0

        while True:

            try:

                response = await self._router.handle(
                    op,
                    payload,
                )

                response = await self._after(
                    op,
                    payload,
                    response,
                )

                elapsed = time.perf_counter() - started

                self.metrics.incr(f"requests.{op}")

                self.metrics.timing(
                    f"latency.{op}",
                    elapsed,
                )

                return response

            except asyncio.CancelledError:
                raise

            except Exception as exc:

                attempt += 1

                self.metrics.incr(
                    f"errors.{op}"
                )

                if not self.retry.should_retry(
                    exc,
                    attempt,
                ):
                    raise

                delay = self.retry.get_backoff(
                    exc,
                    attempt,
                )

                logger.warning(
                    "Retrying %s attempt=%d delay=%.2fs",
                    op,
                    attempt,
                    delay,
                )

                with suppress(Exception):
                    self.metrics.timing(
                        f"retry.backoff.{op}",
                        delay,
                    )

                await self.retry.wait(
                    exc,
                    attempt,
                )

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    async def chat(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> dict[str, Any]:

        return await self._execute(
            "chat",
            {
                "prompt": prompt,
                **kwargs,
            },
        )

    async def embeddings(
        self,
        text: str,
        **kwargs: Any,
    ) -> dict[str, Any]:

        return await self._execute(
            "embeddings",
            {
                "text": text,
                **kwargs,
            },
        )

    async def responses(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:

        return await self._execute(
            "responses",
            {
                "args": args,
                **kwargs,
            },
        )

    # --------------------------------------------------------
    # Factory
    # --------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: RouterConfig,
    ) -> "LLMRouter":

        config.validate()

        providers: list[ProviderRouter] = []

        for provider in config.providers:

            if isinstance(
                provider,
                ProviderRouter,
            ):
                providers.append(provider)
                continue

            clients: list[ClientNode] = []

            for raw_client in provider.get(
                "clients",
                [],
            ):
                clients.append(
                    ClientNode(
                        api_key=raw_client["api_key"],
                        client=raw_client["client"],
                        max_concurrent=config.max_concurrent_per_key,
                        timeout=config.timeout,
                    )
                )

            providers.append(
                ProviderRouter(
                    name=provider["name"],
                    clients=clients,
                    scheduler=provider.get(
                        "scheduler",
                        config.scheduler,
                    ),
                )
            )

        return cls(
            CompositeRouter(providers),
            retry=config.retry,
            middleware=config.middleware,
            max_retries=config.max_retries,
        )