from __future__ import annotations

from ..adapters import AdapterFactory
from ..client.client_node import ClientNode
from ..config.config import RouterConfig
from ..metrics.metrics import MetricsCollector
from ..providers.composite_router import CompositeRouter
from ..providers.provider_router import ProviderRouter
from ..retry.circuit_breaker import CircuitBreaker
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
                    failure_threshold=client_cfg.get("failure_threshold"),
                    cooldown_seconds=client_cfg.get("cooldown_seconds"),
                )

                # Optional attachment for future use
                node.streaming = stream_manager

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

        circuit_breaker = None
        if config.enable_circuit_breaker:
            circuit_breaker = CircuitBreaker(
                failure_threshold=config.circuit_breaker_threshold,
                reset_timeout=config.circuit_breaker_reset_timeout,
            )

        return LLMRouter(
            composite_router=composite,
            retry=retry,
            metrics=metrics,
            middleware=config.middleware,
            max_retries=config.max_retries,
            circuit_breaker=circuit_breaker,
            max_concurrent_requests=config.max_concurrent_requests,
        )
