from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .async_stream import AsyncStreamEngine

_SHARED_EXECUTOR: ThreadPoolExecutor | None = None
_SHARED_LOOP: asyncio.AbstractEventLoop | None = None
_SHARED_LOOP_THREAD: threading.Thread | None = None
_SHARED_LOOP_STARTED = threading.Event()
_EXECUTOR_LOCK = threading.Lock()


def _run_background_loop() -> None:
    """Run a persistent event loop in a background thread."""
    global _SHARED_LOOP
    _SHARED_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(_SHARED_LOOP)
    _SHARED_LOOP_STARTED.set()
    _SHARED_LOOP.run_forever()


def _get_shared_loop() -> asyncio.AbstractEventLoop:
    """Get or create the shared background event loop."""
    global _SHARED_EXECUTOR, _SHARED_LOOP_THREAD

    with _EXECUTOR_LOCK:
        loop_alive = (
            _SHARED_LOOP is not None
            and _SHARED_LOOP_THREAD is not None
            and _SHARED_LOOP_THREAD.is_alive()
        )
        if loop_alive:
            _SHARED_LOOP_STARTED.wait(timeout=5.0)
            assert _SHARED_LOOP is not None
            return _SHARED_LOOP

        _SHARED_LOOP_STARTED.clear()

        _SHARED_LOOP_THREAD = threading.Thread(
            target=_run_background_loop,
            daemon=True,
            name="llmrouterx-sync-stream-loop",
        )
        _SHARED_LOOP_THREAD.start()

        _SHARED_LOOP_STARTED.wait(timeout=5.0)

        if _SHARED_LOOP is None:
            raise RuntimeError("Failed to start shared event loop for sync streaming")

        assert _SHARED_LOOP is not None
        return _SHARED_LOOP


def _get_shared_executor(max_workers: int = 4) -> ThreadPoolExecutor:
    """Get or create the shared thread pool executor."""
    global _SHARED_EXECUTOR

    with _EXECUTOR_LOCK:
        if _SHARED_EXECUTOR is None:
            _SHARED_EXECUTOR = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="llmrouterx-sync-stream",
            )
        return _SHARED_EXECUTOR


def shutdown_sync_engine() -> None:
    """Shutdown the shared sync streaming engine resources.

    Stops the background event loop and shuts down the thread pool executor.
    Safe to call multiple times.
    """
    global _SHARED_EXECUTOR, _SHARED_LOOP, _SHARED_LOOP_THREAD

    with _EXECUTOR_LOCK:
        if _SHARED_LOOP is not None:
            _SHARED_LOOP.call_soon_threadsafe(_SHARED_LOOP.stop)
            _SHARED_LOOP = None

        if _SHARED_LOOP_THREAD is not None:
            _SHARED_LOOP_THREAD.join(timeout=5.0)
            _SHARED_LOOP_THREAD = None

        if _SHARED_EXECUTOR is not None:
            _SHARED_EXECUTOR.shutdown(wait=True, cancel_futures=True)
            _SHARED_EXECUTOR = None

        _SHARED_LOOP_STARTED.clear()


class SyncStreamEngine:
    """
    Synchronous wrapper around AsyncStreamEngine.

    This class runs the async stream in a shared background thread pool with
    a persistent event loop, avoiding per-stream thread allocation overhead.

    .. note::
        Uses a shared thread pool (default 4 workers) and a single persistent
        event loop for all streams. This scales to high numbers of concurrent
        synchronous streams. For servers and batch jobs, prefer the async
        ``StreamingManager.stream`` API directly.
    """

    def __init__(
        self,
        engine: AsyncStreamEngine,
        *,
        max_workers: int = 4,
    ) -> None:
        self._engine = engine
        self._max_workers = max_workers
        _get_shared_executor(max_workers)

    def stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
        stop_event: threading.Event | None = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:

        q: queue.Queue[str | None] = queue.Queue()

        error: list[BaseException | None] = [None]

        async def runner() -> None:
            try:
                async for token in self._engine.stream(
                    prompt,
                    model=model,
                    on_chunk=on_chunk,
                    stop_condition=(stop_event.is_set if stop_event else None),
                    **kwargs,
                ):
                    q.put(token)

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

                yield token
        finally:
            if not future.done():
                future.cancel()
            future.result(timeout=0)

        if error[0]:
            raise error[0]

    def stream_to_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        on_chunk: Callable[[str], None] | None = None,
        stop_event: threading.Event | None = None,
        **kwargs: Any,
    ) -> str:

        return "".join(
            self.stream(
                prompt,
                model=model,
                on_chunk=on_chunk,
                stop_event=stop_event,
                **kwargs,
            )
        )
