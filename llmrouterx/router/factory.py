from __future__ import annotations

from ..adapters import AdapterFactory
from ..client.client_node import ClientNode
from ..config.config import RouterConfig
from ..config.secrets import resolve_key
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
                    api_key=resolve_key(client_cfg),
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

        # Auto-enable Langfuse tracing when it is configured through the
        # environment. This is opt-in via env vars, so routing is unchanged
        # for users who do not use Langfuse.
        middleware = list(config.middleware or [])
        try:
            from ..middleware.langfuse_trace import LangfuseMiddleware, is_configured
        except ImportError:  # pragma: no cover - langfuse not installed
            pass
        else:
            if is_configured() and not any(type(m) is LangfuseMiddleware for m in middleware):
                middleware.append(LangfuseMiddleware())

        # Circuit breaking is handled per key inside ClientNode so that one
        # failing provider/key does not take down the whole router. When
        # ``enable_circuit_breaker`` is set we also install a router-level
        # guard that rejects all requests if the *entire* router keeps
        # failing, giving ``enable_circuit_breaker`` a global effect.
        router_breaker = (
            CircuitBreaker(
                failure_threshold=config.circuit_breaker_threshold,
                reset_timeout=config.circuit_breaker_reset_timeout,
            )
            if config.enable_circuit_breaker
            else None
        )

        return LLMRouter(
            composite_router=composite,
            retry=retry,
            metrics=metrics,
            middleware=middleware,
            max_retries=config.max_retries,
            circuit_breaker=router_breaker,
            max_concurrent_requests=config.max_concurrent_requests,
            total_timeout=config.total_timeout,
        )
