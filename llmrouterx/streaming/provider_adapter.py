from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any


class BaseProviderAdapter(ABC):
    """
    Common interface for all LLM providers.
    """

    @abstractmethod
    async def chat(
        self,
        prompt: str,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Return the full response.
        """

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Yield tokens from the provider.
        """

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


class GenericProviderAdapter(BaseProviderAdapter):
    """
    Adapter for OpenAI-compatible SDKs.

    Works with:

    - NVIDIA NIM
    - OpenAI
    - Together
    - Groq
    - Local OpenAI-compatible APIs
    """

    def __init__(
        self,
        client: Any,
        *,
        default_model: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.client = client
        self.default_model = default_model
        self.embedding_model = embedding_model

    async def chat(
        self,
        prompt: str,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:

        response = await self.client.chat.completions.create(
            model=model or self.default_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=False,
            **kwargs,
        )

        return response.choices[0].message.content

    async def stream(
        self,
        prompt: str,
        *,
        model: str |None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:

        stream = await self.client.chat.completions.create(
            model=model or self.default_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=True,
            **kwargs,
        )

        async for chunk in stream:

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta is None:
                continue

            token = getattr(delta, "content", None)

            if token:
                yield token

    async def embeddings(
        self,
        text: str,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> list[float]:

        result = await self.client.embeddings.create(
            model=model or self.embedding_model,
            input=text,
            **kwargs,
        )

        return result.data[0].embedding