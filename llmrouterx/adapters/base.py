from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from functools import wraps
from importlib import import_module
from typing import TYPE_CHECKING, Any

from ..retry.exponential import HTTPError

if TYPE_CHECKING:
    from ..context.request_context import RequestContext

# ---------------------------------------------------------------------------
# Strict provider-SDK error classification
# ---------------------------------------------------------------------------
# Known provider-SDK exception types, referenced by dotted ``module.Class``
# path so that no SDK has to be installed for the router to work. Types are
# resolved lazily on first use: if the module is not installed (or the class
# no longer exists) that path is simply skipped. Matching is by actual type
# (``isinstance``) rather than by class-name string, so an unrelated exception
# whose name merely contains "timeout" or "connection" is never misclassified.
_SDK_ERROR_PATHS: dict[int, tuple[str, ...]] = {
    503: (
        # Connection failures -> retryable upstream unavailability.
        "openai.APIConnectionError",
        "anthropic.APIConnectionError",
        "groq.APIConnectionError",
        "mistralai.APIConnectionError",
        "together.APIConnectionError",
        "google.genai.errors.ClientError",
        "openai.APIRetryableError",
        "anthropic.APIRetryableError",
        "httpx.ConnectError",  # also matches httpx.ConnectTimeout (subclass)
    ),
    504: (
        # Timeouts -> retryable gateway timeout.
        "openai.APITimeoutError",
        "anthropic.APITimeoutError",
        "groq.APITimeoutError",
        "mistralai.APITimeoutError",
        "together.APITimeoutError",
        "httpx.TimeoutException",
        "httpx.ReadTimeout",
    ),
}

_RESOLVED_SDK_TYPES: dict[int, tuple[type, ...]] = {}


def _resolve_sdk_types(status: int) -> tuple[type, ...]:
    """Resolve (and cache) the SDK exception types registered for ``status``."""
    if status not in _RESOLVED_SDK_TYPES:
        resolved: list[type] = []
        for dotted in _SDK_ERROR_PATHS.get(status, ()):
            module_name, _, class_name = dotted.rpartition(".")
            try:
                module = import_module(module_name)
            except ImportError:  # SDK not installed
                continue
            exc_type = getattr(module, class_name, None)
            if isinstance(exc_type, type):
                resolved.append(exc_type)
        _RESOLVED_SDK_TYPES[status] = tuple(resolved)
    return _RESOLVED_SDK_TYPES[status]


def translate_sdk_error(exc: Exception) -> Exception:
    """
    Normalize provider-SDK exceptions into router-native types.

    SDKs (OpenAI, Anthropic, ...) raise their own error hierarchy. They
    almost always expose a ``status_code`` (and often ``headers``), which is
    everything the router needs to classify the error. Mapping those into
    :class:`HTTPError` makes rate limits and 5xx retryable via the shared
    retry policy and lets ``Retry-After`` be honored.

    Connection and timeout errors that are not builtin ``ConnectionError`` /
    ``asyncio.TimeoutError`` subclasses (for example OpenAI's
    ``APIConnectionError`` / ``APITimeoutError``) are mapped to a retryable
    status *only* when they are a known SDK exception type (see
    ``_SDK_ERROR_PATHS``). Anything else is returned unchanged rather than
    guessed at from its class name.
    """
    if isinstance(
        exc, (HTTPError, asyncio.CancelledError, ConnectionError, asyncio.TimeoutError)
    ):
        return exc

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        headers = getattr(exc, "headers", None) or {}
        return HTTPError(status_code, str(exc), headers=headers)

    for status in _SDK_ERROR_PATHS:
        if isinstance(exc, _resolve_sdk_types(status)):
            return HTTPError(status, str(exc))

    return exc


def translate_async_errors(func):
    """Translate SDK errors raised while awaiting an async adapter call."""

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise translate_sdk_error(exc) from exc

    return wrapper


def translate_stream_errors(func):
    """
    Translate SDK errors raised by an async generator adapter method.

    Covers both errors raised when the generator is created and errors raised
    while iterating it. The decorated function is itself an async generator,
    so ``async for token in adapter.stream(...)`` keeps working unchanged.
    """

    @wraps(func)
    async def wrapper(self, *args, **kwargs) -> AsyncGenerator[str, None]:
        try:
            agen = func(self, *args, **kwargs)
            async for token in agen:
                yield token
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise translate_sdk_error(exc) from exc

    return wrapper


class BaseProviderAdapter(ABC):
    """
    Base class for every provider adapter.

    Implementations:
        - OpenAIAdapter
        - NIMAdapter
        - AzureOpenAIAdapter
        - AnthropicAdapter
        - GeminiAdapter
        - MistralAdapter
    """

    def __init__(
        self,
        client: Any,
        *,
        default_model: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self._client = client
        self._default_model = default_model
        self._embedding_model = embedding_model

    @property
    def client(self) -> Any:
        return self._client

    @property
    def default_model(self) -> str | None:
        return self._default_model

    @property
    def embedding_model(self) -> str | None:
        return self._embedding_model

    @abstractmethod
    async def chat(
        self,
        prompt: str,
        *,
        model: str | None = None,
        context: RequestContext | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Return the complete response.
        """
        raise NotImplementedError

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        context: RequestContext | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Yield streamed response tokens.
        """
        raise NotImplementedError

    @abstractmethod
    async def embeddings(
        self,
        text: str,
        *,
        model: str | None = None,
        context: RequestContext | None = None,
        **kwargs: Any,
    ) -> list[float]:
        """
        Generate embeddings.
        """
        raise NotImplementedError

    async def responses(
        self,
        *args: Any,
        model: str | None = None,
        context: RequestContext | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Optional Responses API.

        Override if the provider supports it.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the Responses API."
        )

    async def health_check(self) -> bool:
        """
        Lightweight health check.

        Providers may override this to perform
        an API ping or model listing.
        """
        return True

    @staticmethod
    def _extract_text(response: Any) -> str:
        """
        Normalize a provider response to its text content.

        Tries, in order: ``output_text`` (OpenAI Responses-style), then
        ``choices[0].message.content`` (chat-style), then ``str`` as a
        last resort. Missing content becomes the empty string rather than
        ``None``.
        """
        text = getattr(response, "output_text", None)
        if text:
            return str(text)
        choices = getattr(response, "choices", None)
        if choices:
            content = getattr(getattr(choices[0], "message", None), "content", None)
            if content:
                return str(content)
        return str(response) if response is not None else ""

    @staticmethod
    def _record_usage(
        context: RequestContext | None,
        response: Any,
    ) -> None:
        """
        Extract token usage from a provider response and attach it to the
        request context for later metric recording.

        Providers expose usage differently (``response.usage`` on OpenAI-style
        clients, ``response.usage`` on Anthropic). Anything else is ignored.
        """
        if context is None:
            return

        usage = getattr(response, "usage", None)
        if usage is None:
            return

        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)

        if prompt_tokens is None and completion_tokens is None:
            return

        context.set(
            "usage",
            {
                "prompt_tokens": prompt_tokens or 0,
                "completion_tokens": completion_tokens or 0,
            },
        )

    @property
    def provider_name(self) -> str:
        """
        Human-readable provider name.
        """
        return self.__class__.__name__.replace("Adapter", "")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"provider={self.provider_name!r}, "
            f"default_model={self.default_model!r})"
        )
