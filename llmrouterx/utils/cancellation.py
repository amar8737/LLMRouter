import asyncio
from collections.abc import AsyncGenerator, Iterable
from contextlib import asynccontextmanager


async def cancel_and_wait(task: asyncio.Task, timeout: float = 5.0) -> None:
    """Cancel a task and wait up to `timeout` seconds for it to finish.

    Silently swallow `CancelledError` and `TimeoutError` so callers can
    continue cleanup.
    """
    if task.done():
        return
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=timeout)
    except asyncio.CancelledError:
        pass
    except TimeoutError:
        pass


async def cancel_tasks_and_wait(tasks: Iterable[asyncio.Task], timeout: float = 5.0) -> None:
    await asyncio.gather(*(cancel_and_wait(t, timeout=timeout) for t in tasks))


@asynccontextmanager
async def background_task(coro) -> AsyncGenerator[asyncio.Task, None]:
    """Run `coro` as a background task for the context lifetime and ensure
    it's cancelled on exit.
    """
    task = asyncio.create_task(coro)
    try:
        yield task
    finally:
        await cancel_and_wait(task)
