"""Synchronous facade over the async :class:`~llmrouterx.router.llmrouter.LLMRouter`.

Normal users who prefer blocking code can use :class:`LLMRouterSync` exactly
like the async router, with no ``async``/``await``:

.. code-block:: python

    from llmrouterx import LLMRouterSync

    router = LLMRouterSync.from_cascade(["openai:OPEN_AI_KEY"])
    print(router.chat("Hello!"))

The facade runs every async call on a single persistent background event loop
(see :mod:`llmrouterx.streaming.sync`), so loop-bound objects such as
``asyncio.Lock`` / ``asyncio.Semaphore`` inside the router stay valid across
calls. ``close()`` shuts the router down; the shared loop stays alive for other
routers and is released by :func:`llmrouterx.streaming.sync.shutdown_sync_engine`.
"""

from __future__ import annotations

import asyncio
import queue
from collections.abc import Generator
from typing import Any

from .config.config import RouterConfig
from .providers.composite_router import CompositeRouter
from .router.llmrouter import LLMRouter
from .streaming.sync import _get_shared_loop, shutdown_sync_engine


async def _run_sync_call(fn: Any) -> Any:
    """Run a synchronous callable inside the background event loop."""
    result = fn()
    if asyncio.iscoroutine(result):
        return await result
    return result


def _run_on_loop(fn: Any, *, timeout: float | None = None) -> Any:
    """Execute a synchronous callable on the shared background event loop."""
    loop = _get_shared_loop()
    future = asyncio.run_coroutine_threadsafe(_run_sync_call(fn), loop)
    return future.result(timeout=timeout)


def _extract_token_text(token: Any) -> str:
    """Extract text from a token which may be a string or dict."""
    if isinstance(token, str):
        return token
    if isinstance(token, dict):
        # Common keys used by different adapters
        return token.get("text") or token.get("response") or token.get("content") or ""
    return str(token)


class LLMRouterSync:
    """
    Blocking, synchronous wrapper around :class:`LLMRouter`.

    All ``chat`` / ``stream`` / ``embeddings`` / ``rerank`` / ``responses``
    calls run on a persistent background event loop, so the same router can be
    used from plain scripts, notebooks, and threads.
    """

    _router: LLMRouter

    def __init__(
        self,
        router: LLMRouter | CompositeRouter | LLMRouterSync | None = None,
    ) -> None:
        if isinstance(router, LLMRouterSync):
            router = router._router
        if isinstance(router, CompositeRouter):
            router = LLMRouter(router)
        if router is None:
            raise ValueError(
                "LLMRouterSync needs an LLMRouter, CompositeRouter, or another LLMRouterSync. "
                "Use LLMRouterSync.from_cascade(...), LLMRouterSync.from_providers(...), "
                "or LLMRouterSync.from_config(...)."
            )
        self._router = router

    @classmethod
    def from_config(cls, config: RouterConfig, **kwargs: Any) -> LLMRouterSync:
        """Build a sync router from a :class:`RouterConfig`."""
        return cls(LLMRouter.from_config(config), **kwargs)

    @classmethod
    def from_cascade(
        cls,
        cascade: list[str],
        **kwargs: Any,
    ) -> LLMRouterSync:
        """Build a sync router from a ``"provider:key_or_env_name"`` chain."""
        return cls(LLMRouter.from_cascade(cascade, **kwargs))

    @classmethod
    def from_providers(
        cls,
        providers: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMRouterSync:
        """Build a sync router with the ergonomic :meth:`LLMRouter.from_providers`."""
        return cls(LLMRouter.from_providers(providers, **kwargs))

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def chat(self, prompt: str, **kwargs: Any) -> Any:
        """Send a chat request and block for the response."""
        return _run_on_loop(lambda: _await(self._router.chat(prompt, **kwargs)))

    def embeddings(self, text: str, **kwargs: Any) -> list[float]:
        """Generate embeddings and block for the result."""
        return _run_on_loop(lambda: _await(self._router.embeddings(text, **kwargs)))

    def rerank(
        self,
        query: str,
        documents: list[str],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Re-rank documents against a query and block for the result."""
        return _run_on_loop(lambda: _await(self._router.rerank(query, documents, **kwargs)))

    def responses(self, *args: Any, **kwargs: Any) -> Any:
        """Call the provider's Responses API and block for the result."""
        return _run_on_loop(lambda: _await(self._router.responses(*args, **kwargs)))

    def stream(self, prompt: str, **kwargs: Any) -> str:
        """Stream a response and block until the full text is collected."""
        return "".join(self.stream_chunks(prompt, **kwargs))

    def stream_chunks(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        """Yield streamed tokens as they arrive, blocking between tokens."""
        q: queue.Queue[Any | None] = queue.Queue()
        error: list[BaseException | None] = [None]

        async def runner() -> None:
            try:
                async for token in self._router.stream(prompt, **kwargs):
                    q.put(token)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                error[0] = exc
            finally:
                q.put(None)

        loop = _get_shared_loop()
        future = asyncio.run_coroutine_threadsafe(runner(), loop)

        try:
            while True:
                token = q.get()
                if token is None:
                    break
                yield _extract_token_text(token)
        finally:
            if not future.done():
                future.cancel()

        if error[0]:
            raise error[0]

    # ------------------------------------------------------------------
    # Introspection / lifecycle
    # ------------------------------------------------------------------

    def health(self) -> dict[str, bool]:
        """Return a ``{provider_name: is_healthy}`` map."""
        return _run_on_loop(lambda: _await(self._router.health()))

    def get_metrics(self) -> dict[str, Any]:
        """Return a snapshot of counters and timings."""
        return self._router.get_metrics()

    def get_timing_stats(self, key: str) -> dict[str, float]:
        """Return latency statistics for a given timing key."""
        return self._router.get_timing_stats(key)

    @property
    def metrics(self) -> Any:
        """The underlying :class:`~llmrouterx.metrics.MetricsCollector`."""
        return self._router.metrics

    @property
    def providers(self) -> list[Any]:
        """The configured provider routers."""
        return self._router.providers

    @property
    def is_closed(self) -> bool:
        return self._router.is_closed

    @property
    def router(self) -> LLMRouter:
        """The wrapped async router."""
        return self._router

    def close(self, timeout: float = 10.0) -> None:
        """Gracefully shut down the router (waits for in-flight requests)."""
        _run_on_loop(lambda: _await(self._router.close(timeout=timeout)), timeout=timeout + 5.0)

    def __enter__(self) -> LLMRouterSync:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def _await(coro: Any) -> Any:
    """Await a coroutine; ``asyncio.run_coroutine_threadsafe`` needs this."""
    return asyncio.wait_for(coro, timeout=None)


__all__ = ["LLMRouterSync", "shutdown_sync_engine"]
