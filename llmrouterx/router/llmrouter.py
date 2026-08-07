from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import suppress
from typing import Any

from ..config.config import RouterConfig
from ..context import RequestContext
from ..exceptions import NoHealthyClientError, StreamError
from ..metrics.metrics import MetricsCollector
from ..middleware.base import BaseMiddleware, MiddlewareResult
from ..providers.composite_router import CompositeRouter
from ..retry.circuit_breaker import CircuitBreaker
from ..retry.exponential import ExponentialRetry, HTTPError
from ..utils.cancellation import cancel_tasks_and_wait

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    asyncio.TimeoutError,
    HTTPError,
)


class _StreamEnd:
    """Sentinel pushed to a stream queue when the worker finishes."""


class LLMRouter:
    """
    Public interface for all LLM operations.
    """

    def __init__(
        self,
        composite_router: CompositeRouter,
        *,
        retry: Any | None = None,
        metrics: MetricsCollector | None = None,
        middleware: list[BaseMiddleware] | None = None,
        max_retries: int = 3,
        circuit_breaker: CircuitBreaker | None = None,
        max_concurrent_requests: int | None = None,
        total_timeout: float | None = None,
    ) -> None:
        self._router = composite_router

        self.retry = retry or ExponentialRetry(
            max_retries=max_retries,
        )

        self.metrics = metrics or MetricsCollector()

        self.middleware = list(middleware or [])

        self._circuit_breaker = circuit_breaker

        self._active_tasks: set[asyncio.Task] = set()
        self._request_semaphore = (
            asyncio.Semaphore(max_concurrent_requests) if max_concurrent_requests else None
        )

        if total_timeout is not None and total_timeout <= 0:
            raise ValueError("total_timeout must be > 0, or None to disable.")

        self._total_timeout = total_timeout
        self._closed = False

    # --------------------------------------------------------
    # Middleware
    # --------------------------------------------------------

    async def _before(
        self,
        op: str,
        payload: dict[str, Any],
        context: RequestContext,
    ) -> tuple[dict[str, Any], Any | None]:
        """
        Run ``before_request`` hooks.

        Returns the (possibly rewritten) payload plus a short-circuit response
        when a middleware asked to stop.
        """

        for middleware in self.middleware:
            result = await middleware.before_request(op, payload, context)

            if result is None:
                continue

            if isinstance(result, MiddlewareResult):
                if result.payload is not None:
                    payload = result.payload

                if result.stop:
                    return payload, result.response

                continue

            if isinstance(result, dict):
                payload = result
                continue

            raise TypeError(
                f"{type(middleware).__name__}.before_request must return "
                f"None, a dict, or a MiddlewareResult; got {type(result).__name__}."
            )

        return payload, None

    async def _after(
        self,
        op: str,
        payload: dict[str, Any],
        response: Any,
        context: RequestContext,
    ) -> Any:
        """
        Run ``after_response`` hooks.
        """

        for middleware in self.middleware:
            result = await middleware.after_response(op, payload, response, context)

            if result is None:
                continue

            if isinstance(result, MiddlewareResult):
                if result.response is not None:
                    response = result.response

                if result.stop:
                    break

                continue

            response = result

        return response

    async def _on_exception(
        self,
        op: str,
        payload: dict[str, Any],
        exc: BaseException,
        context: RequestContext,
    ) -> None:
        """
        Notify middleware about a failed attempt.

        Observers must never mask the original failure, so errors raised by a
        hook here are logged rather than propagated.
        """

        for middleware in self.middleware:
            try:
                await middleware.on_exception(op, payload, exc, context)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Middleware %s.on_exception raised; ignoring.",
                    type(middleware).__name__,
                )

    async def _on_retry(
        self,
        op: str,
        payload: dict[str, Any],
        exc: BaseException,
        attempt: int,
        context: RequestContext,
    ) -> bool:
        """
        Consult middleware about whether to retry a failed attempt.

        Every hook returns a bool; if any returns ``False`` the retry is
        cancelled. Errors raised by a hook are logged and treated as ``True``
        so an observer can never accidentally disable retries.
        """
        for middleware in self.middleware:
            try:
                if not await middleware.on_retry(op, payload, exc, attempt, context):
                    return False
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "Middleware %s.on_retry raised; assuming retry.",
                    type(middleware).__name__,
                )
        return True

    async def _track(self, coro) -> Any:
        """
        Run a request coroutine as a tracked task so graceful shutdown
        can wait for or cancel in-flight requests.

        If the awaiting caller is cancelled, the underlying request task is
        cancelled too so work does not leak.
        """
        task = asyncio.create_task(coro)
        self._track_task(task)
        try:
            return await task
        except asyncio.CancelledError:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            raise
        finally:
            self._active_tasks.discard(task)

    # --------------------------------------------------------
    # Core routing
    # --------------------------------------------------------

    async def _execute(
        self,
        op: str,
        payload: dict[str, Any],
        context: RequestContext | None = None,
    ) -> Any:

        if self._closed:
            raise RuntimeError("Router has been shut down.")

        if self._circuit_breaker and not self._circuit_breaker.allow_request():
            raise NoHealthyClientError(
                "Circuit breaker is open — requests are temporarily rejected."
            )

        if self._request_semaphore:
            await self._request_semaphore.acquire()
        context = context or RequestContext(
            operation=op,
            prompt=payload.get("prompt"),
            model=payload.get("model"),
            max_retries=getattr(self.retry, "max_retries", 0),
        )

        payload, short_circuit = await self._before(op, payload, context)

        if short_circuit is not None:
            self.metrics.incr(f"middleware.short_circuit.{op}")
            context.finish()
            if self._request_semaphore:
                self._request_semaphore.release()
            return short_circuit

        started = time.perf_counter()
        deadline = started + self._total_timeout if self._total_timeout is not None else None

        attempt = 0

        try:
            while True:
                if deadline is not None and time.perf_counter() >= deadline:
                    self.metrics.incr(f"errors.{op}.total_timeout")
                    raise asyncio.TimeoutError(
                        f"'{op}' exceeded total_timeout={self._total_timeout}s"
                    )

                try:
                    if deadline is not None:
                        remaining = deadline - time.perf_counter()

                        if remaining <= 0:
                            raise asyncio.TimeoutError(
                                f"'{op}' exceeded total_timeout={self._total_timeout}s"
                            )

                        response = await asyncio.wait_for(
                            self._router.handle(
                                op,
                                payload,
                                context=context,
                            ),
                            timeout=remaining,
                        )
                    else:
                        response = await self._router.handle(
                            op,
                            payload,
                            context=context,
                        )

                    context.provider = context.provider or self._router.last_provider
                    context.api_key = context.api_key or self._router.last_api_key

                    if deadline is not None and time.perf_counter() >= deadline:
                        self.metrics.incr(f"errors.{op}.total_timeout")
                        raise asyncio.TimeoutError(
                            f"'{op}' exceeded total_timeout={self._total_timeout}s"
                        )

                    response = await self._after(
                        op,
                        payload,
                        response,
                        context,
                    )

                    elapsed = time.perf_counter() - started

                    labels = (
                        {"provider": context.provider} if context.provider is not None else None
                    )
                    self.metrics.incr(f"requests.{op}", labels=labels)
                    self.metrics.timing(f"latency.{op}", elapsed, labels=labels)

                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()

                    context.finish()

                    return response

                except asyncio.CancelledError:
                    self.metrics.incr(f"cancelled.{op}")
                    raise

                except NoHealthyClientError:
                    self.metrics.incr(f"errors.{op}.no_healthy_clients")
                    logger.error(
                        "No healthy clients for '%s' [req_id=%s]",
                        op,
                        context.request_id,
                    )
                    if self._circuit_breaker:
                        self._circuit_breaker.record_failure()
                    raise

                except RETRYABLE_EXCEPTIONS as exc:
                    attempt += 1
                    self.metrics.incr(f"errors.{op}")

                    # Never retry once the overall deadline has elapsed.
                    if deadline is not None and time.perf_counter() >= deadline:
                        self.metrics.incr(f"errors.{op}.total_timeout")
                        raise asyncio.TimeoutError(
                            f"'{op}' exceeded total_timeout={self._total_timeout}s"
                        ) from exc

                    await self._on_exception(op, payload, exc, context)

                    should_retry = self.retry.should_retry(exc, attempt)
                    if should_retry:
                        should_retry = await self._on_retry(op, payload, exc, attempt, context)

                    if not should_retry:
                        logger.error(
                            "Max retries exceeded for '%s' [req_id=%s]: %s",
                            op,
                            context.request_id,
                            str(exc),
                        )
                        self.metrics.incr(f"errors.{op}.retries_exhausted")
                        if self._circuit_breaker:
                            self._circuit_breaker.record_failure()
                        raise

                    delay = self.retry.get_backoff(exc, attempt)
                    context.increment_retry()
                    logger.warning(
                        "Retrying '%s' [req_id=%s] (attempt %d, delay=%.2fs): %s",
                        op,
                        context.request_id,
                        attempt,
                        delay,
                        str(exc),
                    )

                    with suppress(Exception):
                        self.metrics.timing(f"retry.backoff.{op}", delay)

                    await self.retry.wait(exc, attempt)

                except Exception as exc:
                    self.metrics.incr(f"errors.{op}.non_retryable")
                    await self._on_exception(op, payload, exc, context)
                    logger.exception(
                        "Non-retryable error in '%s' [req_id=%s]: %s",
                        op,
                        context.request_id,
                        exc.__class__.__name__,
                    )
                    if self._circuit_breaker:
                        self._circuit_breaker.record_failure()
                    raise

        finally:
            if self._request_semaphore:
                self._request_semaphore.release()

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    async def chat(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        """
        Send a chat request to the best available provider.
        """

        return await self._track(
            self._execute(
                "chat",
                {
                    "prompt": prompt,
                    **kwargs,
                },
            )
        )

    async def embeddings(
        self,
        text: str,
        **kwargs: Any,
    ) -> Any:
        """
        Generate embeddings using the best available provider.
        """

        return await self._track(
            self._execute(
                "embeddings",
                {
                    "text": text,
                    **kwargs,
                },
            )
        )

    async def responses(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Call the provider's Responses API, when supported.
        """

        return await self._track(
            self._execute(
                "responses",
                {
                    "args": args,
                    **kwargs,
                },
            )
        )

    async def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Stream response tokens from the best available provider.

        Failover happens only *before* the first token is emitted. Once any
        token has been delivered the stream is committed to that provider,
        because switching mid-stream would duplicate output. Retries are not
        applied to streams for the same reason.

        The stream runs in a tracked background task so that a router
        ``close()`` can cancel in-flight streams, and so that a consumer
        abandoning the generator (via ``aclose()``) promptly stops the
        underlying provider stream.
        """

        if self._closed:
            raise RuntimeError("Router has been shut down.")

        if self._request_semaphore:
            await self._request_semaphore.acquire()

        context = RequestContext(
            operation="stream",
            prompt=prompt,
            model=kwargs.get("model"),
            max_retries=getattr(self.retry, "max_retries", 0),
        )

        queue: asyncio.Queue[Any] = asyncio.Queue()
        end = _StreamEnd()

        async def worker() -> None:
            try:
                payload, short_circuit = await self._before(
                    "stream",
                    {"prompt": prompt, **kwargs},
                    context,
                )

                if short_circuit is not None:
                    self.metrics.incr("middleware.short_circuit.stream")

                    if isinstance(short_circuit, str):
                        await queue.put(short_circuit)
                    else:
                        for token in short_circuit:
                            await queue.put(token)

                    return

                payload.pop("prompt", None)

                started = time.perf_counter()
                tokens = 0

                try:
                    async for token in self._router.stream(prompt, **payload):
                        tokens += 1
                        self.metrics.incr("stream.tokens")
                        await queue.put(token)

                except asyncio.CancelledError:
                    raise

                except Exception as exc:
                    self.metrics.incr("errors.stream")
                    await self._on_exception("stream", payload, exc, context)
                    raise StreamError(f"Stream failed: {exc}") from exc

                else:
                    self.metrics.incr("requests.stream")
                    self.metrics.timing(
                        "latency.stream",
                        time.perf_counter() - started,
                    )

                finally:
                    context.set("tokens", tokens)

            finally:
                context.finish()
                await queue.put(end)

        task = asyncio.create_task(worker())
        self._track_task(task)

        closed_early = False

        try:
            while True:
                item = await queue.get()

                if isinstance(item, _StreamEnd):
                    break

                yield item
        except GeneratorExit:
            closed_early = True
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

            self._active_tasks.discard(task)

            if self._request_semaphore:
                self._request_semaphore.release()

        if closed_early:
            # Consumer abandoned the generator (aclose()); end it cleanly.
            return

        if task.cancelled():
            raise asyncio.CancelledError

        exc = task.exception()
        if exc is not None:
            raise exc

    async def stream_to_text(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """
        Collect a full streamed response into a single string.
        """

        parts: list[str] = []

        async for token in self.stream(prompt, **kwargs):
            parts.append(token)

        return "".join(parts)

    # --------------------------------------------------------
    # Introspection
    # --------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        """
        Return a snapshot of counters and timings.
        """
        return self.metrics.snapshot()

    async def health(self) -> dict[str, bool]:
        """
        Return a ``{provider_name: is_healthy}`` map.
        """
        return await self._router.health()

    @property
    def providers(self) -> list[Any]:
        return self._router.providers

    # --------------------------------------------------------
    # Factory
    # --------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: RouterConfig,
    ) -> LLMRouter:
        """
        Build a fully wired router from a :class:`RouterConfig`.

        Delegates to :class:`~llmrouterx.router.factory.RouterFactory` so that
        both entry points build identical object graphs.
        """

        from .factory import RouterFactory

        return RouterFactory.build(config)

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    async def close(self, timeout: float = 10.0) -> None:
        """
        Gracefully shut down the router.

        Waits for in-flight requests to complete (up to ``timeout`` seconds),
        then cancels any remaining tasks.
        """
        self._closed = True

        pending = {t for t in self._active_tasks if not t.done()}
        if pending:
            logger.info(
                "Waiting for %d in-flight request(s) to complete (timeout=%.1fs).",
                len(pending),
                timeout,
            )

            # Phase 1: give in-flight requests a chance to finish gracefully.
            done, pending = await asyncio.wait(
                pending,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED,
            )
            for task in done:
                exc = task.exception()
                if exc is not None and not isinstance(exc, (asyncio.CancelledError,)):
                    logger.warning("In-flight request finished with error: %s", exc)

            # Phase 2: cancel anything still running.
            if pending:
                logger.warning(
                    "Timeout reached; cancelling %d in-flight request(s).",
                    len(pending),
                )
                await cancel_tasks_and_wait(pending, timeout=timeout)

        self._active_tasks.clear()
        logger.info("Router shut down gracefully.")

    async def __aenter__(self) -> LLMRouter:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def _track_task(self, task: asyncio.Task) -> None:
        """Track an active request task for graceful shutdown."""
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def circuit_breaker(self) -> CircuitBreaker | None:
        return self._circuit_breaker

    def get_timing_stats(self, key: str) -> dict[str, float]:
        """Return latency statistics for a given timing key."""
        return self.metrics.timing_stats(key)
