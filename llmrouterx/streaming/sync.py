from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Callable, Generator
from typing import Any

from .async_stream import AsyncStreamEngine


class SyncStreamEngine:
    """
    Synchronous wrapper around AsyncStreamEngine.

    This class runs the async stream in a background thread and
    exposes a synchronous generator.

    .. note::
        Each active stream spawns one daemon thread running its own event
        loop. This is convenient for interactive use but does not scale to a
        high number of concurrent synchronous streams — prefer the async
        ``StreamingManager.stream`` API in servers and batch jobs.
    """

    def __init__(
        self,
        engine: AsyncStreamEngine,
    ) -> None:
        self._engine = engine

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

        def background() -> None:
            asyncio.run(runner())

        thread = threading.Thread(
            target=background,
            daemon=True,
        )

        thread.start()

        while True:
            token = q.get()

            if token is None:
                break

            yield token

        thread.join()

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
