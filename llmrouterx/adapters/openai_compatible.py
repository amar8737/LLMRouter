from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from ..retry.exponential import HTTPError
from ..types import RerankResultDict
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
            # Guard the whole extraction so a malformed/missing usage block from a
            # fallback provider (Groq, Mistral, ...) never terminates the stream.
            if context is not None:
                with suppress(Exception):
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        prompt_tokens, completion_tokens = self._extract_usage_tokens(usage)
                        context.set(
                            "usage",
                            {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
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

    @translate_async_errors
    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        model: str | None = None,
        context: RequestContext | None = None,
        top_n: int | None = None,
        **kwargs: Any,
    ) -> list[RerankResultDict]:

        client = self.client
        rerank_api = getattr(client, "rerank", None)
        if rerank_api is None:
            return await super().rerank(
                query,
                documents,
                model=model,
                context=context,
                top_n=top_n,
                **kwargs,
            )

        params: dict[str, Any] = {
            "model": model or self.default_model,
            "query": query,
            "documents": documents,
            **kwargs,
        }
        if top_n is not None:
            params["top_n"] = top_n

        response = await rerank_api.create(**params)

        self._record_usage(context, response)

        results = getattr(response, "results", None)
        if results is None:
            results = getattr(response, "data", None)
        if not results:
            raise HTTPError(502, "Provider returned no rerank results.")

        return self._normalize_rerank(results)

    async def health_check(self) -> bool:

        try:
            await self.client.models.list()

            return True

        except Exception:
            return False
