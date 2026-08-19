from __future__ import annotations

from typing import Any, ClassVar

from ..exceptions import ConfigurationError
from .anthropic import AnthropicAdapter
from .azure import AzureOpenAIAdapter
from .base import BaseProviderAdapter
from .cohere import CohereAdapter
from .gemini import GeminiAdapter
from .groq import GroqAdapter
from .mistral import MistralAdapter
from .nim import NIMAdapter
from .openai import OpenAIAdapter
from .openai_compatible import OpenAICompatibleAdapter
from .together import TogetherAdapter


class AdapterFactory:
    """
    Factory for creating provider adapters.

    This isolates provider-specific construction logic from
    RouterFactory and keeps adapter selection centralized.
    """

    _ADAPTERS: ClassVar[dict[str, type[BaseProviderAdapter]]] = {
        "openai": OpenAIAdapter,
        "cohere": CohereAdapter,
        "azure": AzureOpenAIAdapter,
        "azure_openai": AzureOpenAIAdapter,
        "nim": NIMAdapter,
        "mistral": MistralAdapter,
        "anthropic": AnthropicAdapter,
        "gemini": GeminiAdapter,
        "groq": GroqAdapter,
        "together": TogetherAdapter,
        # Escape hatch for any other OpenAI-compatible endpoint
        # (DeepInfra, OpenRouter, vLLM, Ollama, ...).
        "openai_compatible": OpenAICompatibleAdapter,
    }

    @classmethod
    def register(
        cls,
        provider: str,
        adapter: type[BaseProviderAdapter],
        *,
        overwrite: bool = True,
    ) -> None:
        """
        Register a custom adapter.

        Raises
        ------
        ConfigurationError
            If ``provider`` is blank, ``adapter`` is not a
            :class:`BaseProviderAdapter` subclass, or the name is already
            taken and ``overwrite`` is ``False``.
        """
        key = (provider or "").strip().lower()

        if not key:
            raise ConfigurationError("Provider name must be a non-empty string.")

        if not (isinstance(adapter, type) and issubclass(adapter, BaseProviderAdapter)):
            raise ConfigurationError(
                f"Adapter for '{provider}' must be a subclass of BaseProviderAdapter, "
                f"got {adapter!r}."
            )

        if key in cls._ADAPTERS and not overwrite:
            raise ConfigurationError(f"Adapter '{key}' is already registered.")

        cls._ADAPTERS[key] = adapter

    @classmethod
    def unregister(cls, provider: str) -> None:
        """
        Remove a registered adapter. No-op if it does not exist.
        """
        cls._ADAPTERS.pop((provider or "").strip().lower(), None)

    @classmethod
    def create(
        cls,
        *,
        provider: str,
        client: Any,
        default_model: str | None = None,
        embedding_model: str | None = None,
    ) -> BaseProviderAdapter:
        """
        Create an adapter instance for the requested provider.
        """

        if client is None:
            raise ConfigurationError(
                f"A provider SDK client instance is required for '{provider}'."
            )

        # Already an adapter: pass it straight through so that callers may
        # supply hand-built or third-party adapters. Defaults requested here
        # still apply when the adapter has none of its own.
        if isinstance(client, BaseProviderAdapter):
            if default_model is not None and client.default_model is None:
                client._default_model = default_model
            if embedding_model is not None and client.embedding_model is None:
                client._embedding_model = embedding_model
            return client

        try:
            adapter_cls = cls._ADAPTERS[(provider or "").strip().lower()]
        except KeyError as exc:
            supported = ", ".join(sorted(cls._ADAPTERS))
            raise ConfigurationError(
                f"Unsupported provider '{provider}'. Supported providers: {supported}"
            ) from exc

        return adapter_cls(
            client=client,
            default_model=default_model,
            embedding_model=embedding_model,
        )

    @classmethod
    def supported_providers(cls) -> tuple[str, ...]:
        """
        Return all registered provider names.
        """
        return tuple(sorted(cls._ADAPTERS))
