from __future__ import annotations

from ..adapters import AdapterFactory
from ..client.client_node import ClientNode
from ..config.config import RouterConfig
from ..metrics.metrics import MetricsCollector
from ..providers.composite_router import CompositeRouter
from ..providers.provider_router import ProviderRouter
from ..retry.exponential import ExponentialRetry
from ..streaming.manager import StreamingManager
from .llmrouter import LLMRouter


class RouterFactory:
    """
    Responsible for constructing a complete LLMRouter.

    Responsibilities
    ----------------
    - Build ClientNodes
    - Build ProviderRouters
    - Build CompositeRouter
    - Configure Metrics
    - Configure Retry
    """

    @staticmethod
    def build(
        config: RouterConfig,
    ) -> LLMRouter:

        config.validate()

        metrics = MetricsCollector()

        retry = config.retry or ExponentialRetry(
            max_retries=config.max_retries,
        )

        providers: list[ProviderRouter] = []

        for provider_cfg in config.providers:
            clients: list[ClientNode] = []

            for client_cfg in provider_cfg["clients"]:
                adapter = AdapterFactory.create(
                    provider=provider_cfg["name"],
                    client=client_cfg["client"],
                    default_model=client_cfg.get("default_model"),
                    embedding_model=client_cfg.get("embedding_model"),
                )

                stream_manager = StreamingManager(
                    adapter,
                )

                node = ClientNode(
                    api_key=client_cfg["api_key"],
                    client=adapter,
                    streaming=stream_manager,
                    timeout=config.timeout,
                    max_concurrent=config.max_concurrent_per_key,
                    failure_threshold=client_cfg.get(
                        "failure_threshold", config.circuit_breaker_threshold
                    ),
                    cooldown_seconds=client_cfg.get(
                        "cooldown_seconds", config.circuit_breaker_reset_timeout
                    ),
                    circuit_breaker_enabled=config.enable_circuit_breaker,
                )

                clients.append(node)

            providers.append(
                ProviderRouter(
                    name=provider_cfg["name"],
                    clients=clients,
                    scheduler=provider_cfg.get(
                        "scheduler",
                        config.scheduler,
                    ),
                )
            )

        composite = CompositeRouter(
            providers,
            metrics=metrics,
        )

        # Circuit breaking is handled per key inside ClientNode so that one
        # failing provider/key does not take down the whole router. The router
        # level circuit breaker is left unset by default; pass one explicitly
        # to LLMRouter if you want an additional global guard.
        return LLMRouter(
            composite_router=composite,
            retry=retry,
            metrics=metrics,
            middleware=config.middleware,
            max_retries=config.max_retries,
            circuit_breaker=None,
            max_concurrent_requests=config.max_concurrent_requests,
        )
