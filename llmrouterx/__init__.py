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

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("llmrouterx")
    except PackageNotFoundError:
        __version__ = "0.0.0"
except ImportError:  # pragma: no cover - pre-3.8 importlib.metadata shim
    __version__ = "0.0.0"

__all__ = [
    "AdapterFactory",
    "BaseMiddleware",
    "BaseProviderAdapter",
    "BaseScheduler",
    "ClientNode",
    "CompositeRouter",
    "ExponentialRetry",
    "LLMRouter",
    "LLMRouterSync",
    "LeastBusyScheduler",
    "MetricsCollector",
    "PriorityScheduler",
    "ProviderRouter",
    "RandomScheduler",
    "RoundRobinScheduler",
    "RouterConfig",
    "WeightedScheduler",
    "__version__",
]
