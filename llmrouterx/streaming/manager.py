from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from typing import Any

from ..types import BaseProviderAdapterProtocol
from .async_stream import AsyncStreamEngine
from .sync import SyncStreamEngine
from .tokenizer import TokenizerManager, default_tokenizer


class StreamingManager:
    """
    High-level streaming API.

    This class coordinates:

        ProviderAdapter
            ↓
        AsyncStreamEngine
            ↓
        SyncStreamEngine
            ↓
        Tokenizer
    """

    def __init__(
        self,
        adapter: BaseProviderAdapterProtocol,
        *,
        tokenizer: TokenizerManager | None = None,
    ) -> None:
        self._adapter = adapter

        self._tokenizer = tokenizer or default_tokenizer

        self._async = AsyncStreamEngine(adapter)

        self._sync = SyncStreamEngine(self._async)

    # ----------------------------------------------------
    # Async
    # ----------------------------------------------------

    async def stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:

        async for token in self._async.stream(
            prompt,
            model=model,
            on_chunk=on_chunk,
            **kwargs,
        ):
            yield token

    async def stream_to_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> str:

        return await self._async.stream_to_text(
            prompt,
            model=model,
            on_chunk=on_chunk,
            **kwargs,
        )

    # ----------------------------------------------------
    # Sync
    # ----------------------------------------------------

    def stream_sync(
        self,
        prompt: str,
        *,
        model: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
        **kwargs: Any,
    ):

        yield from self._sync.stream(
            prompt,
            model=model,
            on_chunk=on_chunk,
            **kwargs,
        )

    def stream_sync_to_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> str:

        return self._sync.stream_to_text(
            prompt,
            model=model,
            on_chunk=on_chunk,
            **kwargs,
        )

    # ----------------------------------------------------
    # Tokenizer
    # ----------------------------------------------------

    def tokenize(
        self,
        text: str,
        *,
        tokenizer: str = "whitespace",
    ) -> list[str]:

        return self._tokenizer.tokenize(
            text,
            tokenizer=tokenizer,
        )

    def register_tokenizer(
        self,
        name: str,
        tokenizer,
    ) -> None:

        self._tokenizer.register(
            name,
            tokenizer,
        )
