from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from .base import BaseProviderAdapter


class OpenAICompatibleAdapter(BaseProviderAdapter):
    """
    Adapter for OpenAI-compatible SDKs.

    Supports:

    - OpenAI
    - NVIDIA NIM
    - Groq
    - Together
    - DeepInfra
    - OpenRouter
    - Mistral (OpenAI endpoint)
    """

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
        model: str | None = None,
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

    async def responses(
        self,
        *args: Any,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:

        if not hasattr(self.client, "responses"):
            return await super().responses(
                *args,
                model=model,
                **kwargs,
            )

        return await self.client.responses.create(
            model=model or self.default_model,
            *args,
            **kwargs,
        )

    async def health_check(self) -> bool:

        try:

            await self.client.models.list()

            return True

        except Exception:

            return False