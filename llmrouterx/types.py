from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Protocol, TypedDict, TypeVar, runtime_checkable

# =============================================================================
# Core Response Types (TypedDict for structured, dict-compatible responses)
# =============================================================================


class UsageDict(TypedDict):
    """Token usage from provider responses."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int | None


class ChatCompletionDict(TypedDict):
    """Normalized chat completion response."""

    text: str
    model: str | None
    usage: UsageDict | None
    provider: str | None
    raw: Any  # Original provider response for advanced use


class EmbeddingResponseDict(TypedDict):
    """Normalized embedding response."""

    embedding: list[float]
    model: str | None
    usage: UsageDict | None
    provider: str | None


class StreamChunkDict(TypedDict):
    """Normalized stream chunk."""

    text: str
    index: int | None
    finish_reason: str | None
    usage: UsageDict | None


class RerankResultDict(TypedDict):
    """Normalized rerank result."""

    index: int
    relevance_score: float


class ProviderInfoDict(TypedDict):
    """Provider health/info."""

    name: str
    healthy: bool
    clients: int
    scheduler: str | None


# =============================================================================
# Payload Types (what flows through middleware/hooks)
# =============================================================================


class ChatPayload(TypedDict):
    prompt: str
    model: str | None


class EmbeddingPayload(TypedDict):
    text: str
    model: str | None


class RerankPayload(TypedDict):
    query: str
    documents: list[str]
    model: str | None
    top_n: int | None


class StreamPayload(TypedDict):
    prompt: str
    model: str | None


OperationPayload = ChatPayload | EmbeddingPayload | RerankPayload | StreamPayload


# =============================================================================
# Protocol Definitions (structural typing, no inheritance required)
# =============================================================================


# SDK clients are too diverse to model precisely - use Any for flexibility
# Adapters use getattr/hasattr to access SDK-specific methods
# Type alias for clarity
SDKClient = Any


@runtime_checkable
class BaseProviderAdapterProtocol(Protocol):
    """Protocol for all provider adapters."""

    @property
    def client(self) -> SDKClient: ...

    @property
    def default_model(self) -> str | None: ...

    @property
    def embedding_model(self) -> str | None: ...

    @property
    def provider_name(self) -> str: ...

    async def chat(
        self,
        prompt: str,
        *,
        model: str | None = None,
        context: Any | None = None,  # RequestContext
        **kwargs: Any,
    ) -> str: ...

    async def stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        context: Any | None = None,  # RequestContext
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]: ...

    async def embeddings(
        self,
        text: str,
        *,
        model: str | None = None,
        context: Any | None = None,  # RequestContext
        **kwargs: Any,
    ) -> list[float]: ...

    async def responses(
        self,
        *args: Any,
        model: str | None = None,
        context: Any | None = None,  # RequestContext
        **kwargs: Any,
    ) -> Any: ...

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str | None = None,
        context: Any | None = None,  # RequestContext
        top_n: int | None = None,
        **kwargs: Any,
    ) -> list[RerankResultDict]: ...

    async def health_check(self) -> bool: ...


# =============================================================================
# Provider Router Protocol
# =============================================================================


@runtime_checkable
class ProviderRouterProtocol(Protocol):
    """Protocol for provider routers (implemented by ProviderRouter)."""

    @property
    def name(self) -> str: ...

    @property
    def clients(self) -> list[Any]:  # list[ClientNode]
        ...

    @property
    def last_api_key(self) -> str | None: ...

    @property
    def scheduler(self) -> Any | None:  # BaseScheduler
        ...

    async def handle(
        self,
        op: str,
        payload: OperationPayload,
        *,
        context: Any | None = None,  # RequestContext
        **kwargs: Any,
    ) -> Any: ...

    async def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]: ...

    async def is_healthy(self, force: bool = False) -> bool: ...

    async def health(self, timeout: float = 5.0) -> dict[str, bool]: ...


# =============================================================================
# Composite Router Protocol
# =============================================================================


@runtime_checkable
class CompositeRouterProtocol(Protocol):
    """Protocol for composite router (implemented by CompositeRouter)."""

    @property
    def providers(self) -> list[ProviderRouterProtocol]: ...

    @property
    def last_provider(self) -> str | None: ...

    @property
    def last_api_key(self) -> str | None: ...

    @property
    def metrics(self) -> Any | None:  # MetricsCollector
        ...

    async def handle(
        self,
        op: str,
        payload: OperationPayload,
        *,
        context: Any | None = None,  # RequestContext
        **kwargs: Any,
    ) -> Any: ...

    async def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]: ...

    async def health(
        self,
        *,
        timeout: float | None = 5.0,
    ) -> dict[str, bool]: ...

    async def is_healthy(self) -> bool: ...


# =============================================================================
# Metrics Protocol
# =============================================================================


@runtime_checkable
class MetricsCollectorProtocol(Protocol):
    """Protocol for metrics collector."""

    def incr(
        self,
        key: str,
        amount: int = 1,
        labels: dict[str, str] | None = None,
    ) -> None: ...

    def timing(
        self,
        key: str,
        seconds: float,
        labels: dict[str, str] | None = None,
    ) -> None: ...

    def track_tokens(
        self,
        provider: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...

    def get(self) -> dict[str, Any]: ...

    def timing_stats(self, key: str) -> dict[str, float]: ...

    def reset(self) -> None: ...


# =============================================================================
# Retry Policy Protocol
# =============================================================================


@runtime_checkable
class RetryPolicyProtocol(Protocol):
    """Protocol for retry policies."""

    def should_retry(self, error: BaseException, attempt: int) -> bool: ...

    def get_backoff(self, error: BaseException, attempt: int) -> float: ...

    async def wait(self, error: BaseException, attempt: int) -> None: ...


# =============================================================================
# Circuit Breaker Protocol
# =============================================================================


@runtime_checkable
class CircuitBreakerProtocol(Protocol):
    """Protocol for circuit breaker."""

    @property
    def state(self) -> Any:  # CircuitState
        ...

    @property
    def failure_count(self) -> int: ...

    def allow_request(self) -> bool: ...

    def record_success(self) -> None: ...

    def record_failure(self) -> None: ...

    def reset(self) -> None: ...

    def cooldown_remaining(self) -> float | None: ...

    def reset_if_expired(self) -> bool: ...


# =============================================================================
# Middleware Protocol
# =============================================================================


@runtime_checkable
class MiddlewareProtocol(Protocol):
    """Protocol for middleware hooks."""

    async def before_request(
        self,
        operation: str,
        payload: OperationPayload,
        context: Any,  # RequestContext
    ) -> dict[str, Any] | None: ...

    async def after_response(
        self,
        operation: str,
        payload: OperationPayload,
        response: Any,
        context: Any,  # RequestContext
    ) -> dict[str, Any] | None: ...

    async def on_exception(
        self,
        operation: str,
        payload: OperationPayload,
        exception: BaseException,
        context: Any,  # RequestContext
    ) -> None: ...

    async def on_retry(
        self,
        operation: str,
        payload: OperationPayload,
        exception: BaseException,
        attempt: int,
        context: Any,  # RequestContext
    ) -> bool: ...


# =============================================================================
# Scheduler Protocol
# =============================================================================


@runtime_checkable
class SchedulerProtocol(Protocol):
    """Protocol for client schedulers."""

    async def select(self, provider_router: ProviderRouterProtocol) -> Any | None:  # ClientNode
        ...


# =============================================================================
# Factory Protocols
# =============================================================================


@runtime_checkable
class ClientFactoryProtocol(Protocol):
    """Protocol for custom client factories."""

    def __call__(self, provider: str, api_key: str) -> SDKClient: ...


# =============================================================================
# Config Types
# =============================================================================


class ClientConfigDict(TypedDict, total=False):
    """Client configuration from YAML/JSON."""

    client: str  # e.g., "openai"
    api_key: str | None
    api_key_env: str | None
    api_key_env_scan: bool | None
    api_key_env_regex: str | None
    api_key_file: str | None
    default_model: str | None
    embedding_model: str | None
    base_url: str | None
    weight: float | None
    priority: int | None
    failure_threshold: int | None
    cooldown_seconds: float | None
    circuit_breaker_enabled: bool | None
    count_transient_failures: bool | None


class ProviderConfigDict(TypedDict, total=False):
    """Provider configuration from YAML/JSON."""

    name: str
    clients: list[ClientConfigDict]
    scheduler: str | None


class RouterConfigDict(TypedDict, total=False):
    """Full router configuration from YAML/JSON."""

    providers: list[ProviderConfigDict]
    timeout: float
    max_retries: int
    max_concurrent_per_key: int
    max_concurrent_requests: int | None
    total_timeout: float | None
    enable_circuit_breaker: bool
    circuit_breaker_threshold: int
    circuit_breaker_reset_timeout: float


# =============================================================================
# Type Variables for Generics
# =============================================================================

T = TypeVar("T")
PayloadT = TypeVar("PayloadT", bound=OperationPayload)
ResponseT = TypeVar("ResponseT")


# =============================================================================
# Utility Types
# =============================================================================

# Type alias for labeled metrics keys
LabelDict = dict[str, str]
CounterKey = str
TimingKey = str

# Type alias for hook results
HookResult = dict[str, Any] | None

# Type alias for operation names
OperationName = str

# Type alias for model names
ModelName = str

# Type alias for API keys (masked in logs)
MaskedKey = str
