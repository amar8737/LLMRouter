"""LLMRouter package

Expose the public router and package version.
"""

from .adapters import AdapterFactory
from .adapters.base import BaseProviderAdapter
from .client import ClientNode
from .config.config import RouterConfig
from .metrics.metrics import MetricsCollector
from .middleware import BaseMiddleware
from .providers.composite_router import CompositeRouter
from .providers.provider_router import ProviderRouter
from .retry.exponential import ExponentialRetry
from .router.llmrouter import LLMRouter
from .scheduler import (
    BaseScheduler,
    LeastBusyScheduler,
    PriorityScheduler,
    RandomScheduler,
    RoundRobinScheduler,
    WeightedScheduler,
)
from .sync import LLMRouterSync
from .types import (
    BaseProviderAdapterProtocol,
    ChatCompletionDict,
    ChatPayload,
    CircuitBreakerProtocol,
    ClientConfigDict,
    ClientFactoryProtocol,
    CompositeRouterProtocol,
    EmbeddingPayload,
    EmbeddingResponseDict,
    MetricsCollectorProtocol,
    MiddlewareProtocol,
    OperationPayload,
    ProviderConfigDict,
    ProviderInfoDict,
    ProviderRouterProtocol,
    RerankPayload,
    RerankResultDict,
    RetryPolicyProtocol,
    RouterConfigDict,
    SchedulerProtocol,
    SDKClient,
    StreamChunkDict,
    StreamPayload,
    UsageDict,
)

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("llmrouterx")
    except PackageNotFoundError:
        __version__ = "0.1.31"
except ImportError:  # pragma: no cover - pre-3.8 importlib.metadata shim
    __version__ = "0.1.32"

__all__ = [
    "AdapterFactory",
    "BaseMiddleware",
    "BaseProviderAdapter",
    "BaseProviderAdapterProtocol",
    "BaseScheduler",
    "ChatCompletionDict",
    "ChatPayload",
    "CircuitBreakerProtocol",
    "ClientConfigDict",
    "ClientFactoryProtocol",
    "ClientNode",
    "CompositeRouter",
    "CompositeRouterProtocol",
    "EmbeddingPayload",
    "EmbeddingResponseDict",
    "ExponentialRetry",
    "LLMRouter",
    "LLMRouterSync",
    "LeastBusyScheduler",
    "MetricsCollector",
    "MetricsCollectorProtocol",
    "MiddlewareProtocol",
    "OperationPayload",
    "PriorityScheduler",
    "ProviderConfigDict",
    "ProviderInfoDict",
    "ProviderRouter",
    "ProviderRouterProtocol",
    "RandomScheduler",
    "RerankPayload",
    "RerankResultDict",
    "RetryPolicyProtocol",
    "RoundRobinScheduler",
    "RouterConfig",
    "RouterConfigDict",
    "SDKClient",
    "SchedulerProtocol",
    "StreamChunkDict",
    "StreamPayload",
    "UsageDict",
    "WeightedScheduler",
    "__version__",
]
