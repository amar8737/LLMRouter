from __future__ import annotations

from typing import Any

from ..client.client_node import ClientNode
from ..config.config import RouterConfig
from ..metrics.metrics import MetricsCollector
from ..providers.composite_router import CompositeRouter
from ..providers.provider_router import ProviderRouter
from ..retry.exponential import ExponentialRetry
from ..streaming.manager import StreamingManager
from ..streaming.provider_adapter import GenericProviderAdapter

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

                adapter = GenericProviderAdapter(
                    client_cfg["client"],
                    default_model=client_cfg.get(
                        "default_model"
                    ),
                    embedding_model=client_cfg.get(
                        "embedding_model"
                    ),
                )

                stream_manager = StreamingManager(
                    adapter,
                )

                node = ClientNode(
                    api_key=client_cfg["api_key"],
                    client=adapter,
                    timeout=config.timeout,
                    max_concurrent=config.max_concurrent_per_key,
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

        return LLMRouter(
            composite_router=composite,
            retry=retry,
            metrics=metrics,
            middleware=config.middleware,
            max_retries=config.max_retries,
        )