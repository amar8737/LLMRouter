from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from functools import wraps
from typing import Any

from ..retry.exponential import HTTPError


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
    status so the retry policy treats them as transient.
    """
    if isinstance(exc, (HTTPError, asyncio.CancelledError, ConnectionError, asyncio.TimeoutError)):
        return exc

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        headers = getattr(exc, "headers", None) or {}
        return HTTPError(status_code, str(exc), headers=headers)

    name = type(exc).__name__.lower()
    if "connection" in name or "connect" in name:
        return HTTPError(503, str(exc))

    if "timeout" in name or "timedout" in name or "timed_out" in name:
        return HTTPError(504, str(exc))

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
