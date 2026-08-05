from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any


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