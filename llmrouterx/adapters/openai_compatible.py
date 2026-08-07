from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from ..retry.exponential import HTTPError
from .base import (
    BaseProviderAdapter,
    translate_async_errors,
    translate_stream_errors,
)

if TYPE_CHECKING:
    from ..context.request_context import RequestContext


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

    @translate_async_errors
    async def chat(
        self,
        prompt: str,
        *,
        model: str | None = None,
        context: RequestContext | None = None,
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

        self._record_usage(context, response)

        content = response.choices[0].message.content

        return content or ""

    @translate_stream_errors
    async def stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        context: RequestContext | None = None,
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
            # A final chunk may carry usage (e.g. stream_options include_usage).
            usage = getattr(chunk, "usage", None)
            if usage is not None and context is not None:
                context.set(
                    "usage",
                    {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    },
                )

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta is None:
                continue

            token = getattr(delta, "content", None)

            if token:
                yield token

    @translate_async_errors
    async def embeddings(
        self,
        text: str,
        *,
        model: str | None = None,
        context: RequestContext | None = None,
        **kwargs: Any,
    ) -> list[float]:

        result = await self.client.embeddings.create(
            model=model or self.embedding_model,
            input=text,
            **kwargs,
        )

        self._record_usage(context, result)

        if not result.data:
            raise HTTPError(502, "Provider returned no embeddings.")

        return result.data[0].embedding

    @translate_async_errors
    async def responses(
        self,
        *args: Any,
        model: str | None = None,
        context: RequestContext | None = None,
        **kwargs: Any,
    ) -> Any:

        if not hasattr(self.client, "responses"):
            return await super().responses(
                *args,
                model=model,
                context=context,
                **kwargs,
            )

        params = {
            "model": model or self.default_model,
            **kwargs,
        }

        response = await self.client.responses.create(
            *args,
            **params,
        )

        self._record_usage(context, response)

        # Normalize to the same ``str`` contract as ``chat``.
        return self._extract_text(response)

    async def health_check(self) -> bool:

        try:
            await self.client.models.list()

            return True

        except Exception:
            return False
