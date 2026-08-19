from __future__ import annotations

from collections.abc import AsyncGenerator
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


class CohereAdapter(BaseProviderAdapter):
    """
    Adapter for Cohere (chat, stream, embeddings, rerank).

    Uses the ``cohere`` Python SDK (``AsyncClient``). Install with
    ``pip install llmrouterx[cohere]``.
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

        response = await self.client.chat(
            message=prompt,
            model=model or self.default_model,
            **kwargs,
        )

        self._record_usage(context, response)

        text = getattr(response, "text", None)
        if not text:
            raise HTTPError(502, "Cohere returned no chat text.")
        return str(text)

    @translate_stream_errors
    async def stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        context: RequestContext | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:

        stream = self.client.chat_stream(
            message=prompt,
            model=model or self.default_model,
            **kwargs,
        )

        async for event in stream:
            if getattr(event, "event_type", None) == "text-generation":
                text = getattr(event, "text", None)
                if text:
                    yield str(text)

    @translate_async_errors
    async def embeddings(
        self,
        text: str,
        *,
        model: str | None = None,
        context: RequestContext | None = None,
        **kwargs: Any,
    ) -> list[float]:

        response = await self.client.embed(
            texts=[text],
            model=model or self.embedding_model,
            **kwargs,
        )

        self._record_usage(context, response)

        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            raise HTTPError(502, "Cohere returned no embeddings.")
        return list(embeddings[0])

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

        params: dict[str, Any] = {
            "model": model or self.default_model,
            "query": query,
            "documents": documents,
            **kwargs,
        }
        if top_n is not None:
            params["top_n"] = top_n

        response = await self.client.v2.rerank(**params)

        self._record_usage(context, response)

        results = getattr(response, "results", None)
        if not results:
            raise HTTPError(502, "Cohere returned no rerank results.")

        return self._normalize_rerank(results)

    async def health_check(self) -> bool:

        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
