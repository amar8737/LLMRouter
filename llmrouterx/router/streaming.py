import asyncio
from collections.abc import AsyncGenerator, Callable

from ..metrics.metrics import MetricsCollector
from ..providers.composite_router import CompositeRouter


def _default_tokenize(text: str):
    # simple word-based tokenizer that keeps spaces
    parts = text.split(" ")
    for i, p in enumerate(parts):
        if i + 1 < len(parts):
            yield p + " "
        else:
            yield p


class StreamingLLMRouter:
    """Streaming router with three modes:
    - async generator `stream()`
    - sync iterator `stream_sync()` (wraps async behavior)
    - callback via `on_chunk`

    Note: Real provider-level streaming requires clients that expose streaming APIs.
    This implementation will attempt to call a `stream` method on provider clients
    when available; otherwise it falls back to tokenizing the full response.
    """

    def __init__(
        self,
        composite: CompositeRouter,
        *,
        metrics: MetricsCollector | None = None,
        tokenizer: Callable[[str], list] | None = None,
    ):
        self._composite = composite
        self.metrics = metrics or MetricsCollector()
        self.tokenizer = tokenizer or _default_tokenize

    async def _stream_from_text(
        self, text: str, on_chunk: Callable[[str], None] | None = None
    ) -> AsyncGenerator[str, None]:
        for token in self.tokenizer(text):
            if on_chunk:
                try:
                    on_chunk(token)
                except Exception:
                    pass
            yield token

    async def stream(
        self, prompt: str, on_chunk: Callable[[str], None] | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Async generator that yields tokens (strings).

        If provider returns a streaming-capable response (not implemented by default),
        it should be used; otherwise the full response is tokenized and streamed.
        """
        payload = {"prompt": prompt, **kwargs}
        # get a single full response from composite (providers may support streaming in future)
        result = await self._composite.handle("chat", payload)
        text = None
        if isinstance(result, dict):
            text = result.get("response") or result.get("text") or str(result)
        else:
            text = str(result)

        async for token in self._stream_from_text(text, on_chunk=on_chunk):
            self.metrics.incr("stream.tokens")
            yield token

    async def stream_until_complete(
        self, prompt: str, on_chunk: Callable[[str], None] | None = None, **kwargs
    ) -> str:
        """Collect tokens from `stream` and return the full concatenated response."""
        out = []
        async for token in self.stream(prompt, on_chunk=on_chunk, **kwargs):
            out.append(token)
        return "".join(out)

    def stream_sync(self, prompt: str, on_chunk: Callable[[str], None] | None = None, **kwargs):
        """Synchronous iterator wrapper that runs the async stream to completion and yields tokens.

        This is a best-effort shim for blocking code (CLI). It will run the async
        path to completion and then yield tokens synchronously as they are produced
        from the final response. It does not provide true incremental IO when the
        provider does not support streaming.
        """
        text = asyncio.run(self.stream_until_complete(prompt, on_chunk=on_chunk, **kwargs))
        for token in self.tokenizer(text):
            if on_chunk:
                try:
                    on_chunk(token)
                except Exception:
                    pass
            yield token

    def stream_sync_until_complete(
        self, prompt: str, on_chunk: Callable[[str], None] | None = None, **kwargs
    ) -> str:
        """Synchronous convenience: return concatenated response."""
        return "".join(self.stream_sync(prompt, on_chunk=on_chunk, **kwargs))
