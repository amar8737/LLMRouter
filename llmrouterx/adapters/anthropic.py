from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from ..exceptions import ConfigurationError
from .base import BaseProviderAdapter


class AnthropicAdapter(BaseProviderAdapter):
    """
    Adapter for the native Anthropic Messages API.

    Anthropic is not OpenAI-compatible, so this adapter talks to
    ``client.messages`` rather than ``client.chat.completions``.

    Notes
    -----
    - ``max_tokens`` is required by the Messages API. A default is applied
      when the caller does not supply one.
    - Anthropic does not expose an embeddings endpoint; :meth:`embeddings`
      raises :class:`NotImplementedError` so callers fail loudly instead of
      silently routing embeddings to a provider that cannot serve them.
    """

    DEFAULT_MAX_TOKENS = 1024

    def _resolve_model(self, model: str | None) -> str:
        resolved = model or self.default_model

        if not resolved:
            raise ConfigurationError(
                "No model specified for AnthropicAdapter. Pass `model=` per call "
                "or set `default_model` on the client configuration."
            )

        return resolved

    async def chat(
        self,
        prompt: str,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:

        kwargs.setdefault("max_tokens", self.DEFAULT_MAX_TOKENS)

        response = await self.client.messages.create(
            model=self._resolve_model(model),
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            **kwargs,
        )

        blocks = getattr(response, "content", None) or []

        return "".join(
            block.text
            for block in blocks
            if getattr(block, "type", None) == "text"
            and getattr(block, "text", None)
        )

    async def stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:

        kwargs.setdefault("max_tokens", self.DEFAULT_MAX_TOKENS)

        stream = await self.client.messages.create(
            model=self._resolve_model(model),
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            stream=True,
            **kwargs,
        )

        async for event in stream:
            if getattr(event, "type", None) != "content_block_delta":
                continue

            delta = getattr(event, "delta", None)

            token = getattr(delta, "text", None)

            if token:
                yield token

    async def embeddings(
        self,
        text: str,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> list[float]:

        raise NotImplementedError(
            "Anthropic does not provide an embeddings API. Route embeddings to "
            "a provider that supports them (for example OpenAI or Mistral)."
        )

    async def health_check(self) -> bool:

        try:
            await self.client.models.list()

            return True

        except Exception:
            return False
