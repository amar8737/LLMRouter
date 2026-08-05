from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from typing import Any

from .provider_adapter import BaseProviderAdapter


class AsyncStreamEngine:
    """
    Handles asynchronous streaming from a provider adapter.

    Responsibilities
    ----------------
    - Consume provider streams
    - Invoke callbacks
    - Support cancellation
    - Yield tokens
    """

    def __init__(
        self,
        adapter: BaseProviderAdapter,
    ) -> None:
        self._adapter = adapter

    async def stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
        stop_condition: Callable[[], bool] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from the provider.
        """

        async for token in self._adapter.stream(
            prompt,
            model=model,
            **kwargs,
        ):

            if stop_condition and stop_condition():
                break

            if on_chunk:
                on_chunk(token)

            yield token

    async def stream_to_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
        stop_condition: Callable[[], bool] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Collect streamed tokens into a single string.
        """

        parts: list[str] = []

        async for token in self.stream(
            prompt,
            model=model,
            on_chunk=on_chunk,
            stop_condition=stop_condition,
            **kwargs,
        ):
            parts.append(token)

        return "".join(parts)