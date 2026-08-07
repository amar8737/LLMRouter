from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from typing import Any

from ..adapters.base import BaseProviderAdapter
from ..exceptions import ConfigurationError
from ..retry.circuit_breaker import CircuitBreaker, CircuitState
from ..retry.exponential import ExponentialRetry, HTTPError
from ..streaming.manager import StreamingManager
from ..utils.masking import mask_api_key

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ClientNode:
    """
    Represents a single API key/client.

    Responsibilities:
    - Health tracking
    - Failure tracking
    - Cooldown management
    - Concurrency limiting
    - Sending requests
    """

    #: Class-level defaults. Override per instance with the
    #: ``failure_threshold`` / ``cooldown_seconds`` constructor arguments.
    FAILURE_THRESHOLD = 5
    COOLDOWN_SECONDS = 30.0

    SUPPORTED_OPERATIONS = ("chat", "embeddings", "responses")

    def __init__(
        self,
        api_key: str,
        client: BaseProviderAdapter,
        *,
        streaming: StreamingManager | None = None,
        timeout: float | None = 60.0,
        max_concurrent: int = 100,
        failure_threshold: int | None = None,
        cooldown_seconds: float | None = None,
        circuit_breaker_enabled: bool = True,
        weight: float = 1.0,
        priority: int = 100,
        count_transient_failures: bool = False,
    ) -> None:
        if max_concurrent < 1:
            raise ConfigurationError("max_concurrent must be >= 1.")

        if timeout is not None and timeout <= 0:
            raise ConfigurationError("timeout must be greater than zero, or None to disable.")

        self.streaming = streaming
        self.api_key = api_key
        self.client = client

        self.timeout = timeout
        self.max_concurrent = max_concurrent

        self.failure_threshold = (
            self.FAILURE_THRESHOLD if failure_threshold is None else failure_threshold
        )
        self.cooldown_seconds = (
            self.COOLDOWN_SECONDS if cooldown_seconds is None else cooldown_seconds
        )

        # When True, transient HTTP errors (429/5xx) also count toward the
        # failure threshold. Off by default so a busy-but-alive provider is
        # not circuit-broken purely from retryable load.
        self.count_transient_failures = count_transient_failures

        # Per-key circuit breaker. Health is evaluated per client so that one
        # broken provider/key does not take down the whole router.
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.failure_threshold,
            reset_timeout=self.cooldown_seconds,
        )
        self.circuit_breaker_enabled = circuit_breaker_enabled

        # Consumed by WeightedScheduler / PriorityScheduler.
        self.weight = weight
        self.priority = priority

        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)

        self.active_requests = 0

        self.last_success: datetime | None = None
        self.last_failure: datetime | None = None

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @property
    def failures(self) -> int:
        """
        Number of consecutive failures recorded on the circuit breaker.

        The per-key circuit breaker is the single source of truth for client
        health; this is a read-only view over it.
        """
        return self.circuit_breaker.failure_count

    @property
    def cooldown_until(self) -> datetime | None:
        """
        Wall-clock estimate of when the circuit breaker will recover, or None
        when it is not currently cooling down.
        """
        remaining = self.circuit_breaker.cooldown_remaining()
        if remaining is None:
            return None
        return _utcnow() + timedelta(seconds=remaining)

    async def is_healthy(self) -> bool:
        async with self._lock:
            if not self.circuit_breaker_enabled:
                return True

            # Cooldown expired -> full recovery.
            self.circuit_breaker.reset_if_expired()

            return self.circuit_breaker.state != CircuitState.OPEN

    async def check_health(self, force: bool = False) -> bool:
        """
        Proactive health check with automatic recovery.

        Only performs an actual provider ping if:
        1. force is True, or
        2. Client has no active requests (and is not in cooldown)

        Args:
            force: If True, always perform health check regardless of state.

        Returns:
            bool: True if client is healthy, False otherwise.
        """
        async with self._lock:
            if not self.circuit_breaker_enabled:
                should_probe = force
            else:
                self.circuit_breaker.reset_if_expired()

                if self.circuit_breaker.state == CircuitState.OPEN:
                    return False

                should_probe = force or self.active_requests == 0

        if not should_probe:
            return True

        try:
            result = await asyncio.wait_for(
                self.client.health_check(),
                timeout=self.timeout,
            )
        except Exception:
            key_suffix = self.api_key[-4:] if self.api_key else "????"
            logger.exception("Health check failed for key ...%s", key_suffix)
            return False

        if result:
            async with self._lock:
                if self.circuit_breaker_enabled:
                    self.circuit_breaker.record_success()

        return result

    @property
    def is_saturated(self) -> bool:
        """
        True when every concurrency slot for this key is already in use.
        """
        return self.active_requests >= self.max_concurrent

    async def reset(self) -> None:
        """
        Clear failure state and end any active cooldown.
        """
        async with self._lock:
            self.circuit_breaker.reset()

    # ------------------------------------------------------------------
    # Reservation API
    # ------------------------------------------------------------------

    async def acquire(self) -> None:
        """
        Reserve one concurrency slot.
        """
        await self._semaphore.acquire()

        # If we are cancelled between taking the semaphore and recording the
        # reservation, hand the slot back instead of leaking it.
        try:
            async with self._lock:
                self.active_requests += 1
        except BaseException:
            self._semaphore.release()
            raise

    async def release(self) -> None:
        """
        Release one concurrency slot.
        """
        try:
            async with self._lock:
                self.active_requests = max(0, self.active_requests - 1)
        finally:
            self._semaphore.release()

    async def execute(self, coroutine):
        """
        Execute a coroutine with proper resource tracking and error handling.

        Ensures that acquire/release pair is atomic - if an exception
        occurs, the slot is returned to the pool.

        Health is only penalized for availability failures:
        - Transient HTTP errors (429, 5xx) do not count by default because
          they are server-side load issues, not client faults.
        - Permanent 4xx request errors (auth, bad payload, not found) never
          count because they are configuration/request faults, not an
          indication the key is unavailable.
        - Non-retryable server errors (e.g. 501) still count as failures.
        """
        await self.acquire()

        try:
            result = await coroutine
        except asyncio.CancelledError:
            raise
        except HTTPError as exc:
            if exc.status_code in ExponentialRetry.RETRYABLE_HTTP_STATUS:
                logger.debug(
                    "Transient HTTP %d for key ...%s — not penalizing health.",
                    exc.status_code,
                    self.api_key[-4:] if self.api_key else "????",
                )
                if self.count_transient_failures:
                    await self._record_failure()
                raise
            if 400 <= exc.status_code < 500:
                logger.debug(
                    "Permanent HTTP %d for key ...%s — request error, not penalizing health.",
                    exc.status_code,
                    self.api_key[-4:] if self.api_key else "????",
                )
                raise
            await self._record_failure()
            raise
        except Exception:
            await self._record_failure()
            raise
        else:
            await self._record_success()
            return result
        finally:
            await self.release()

    async def _record_failure(self) -> None:
        """
        Record a failure against this key and open its circuit breaker once
        the threshold is crossed.
        """
        async with self._lock:
            self.last_failure = _utcnow()
            if self.circuit_breaker_enabled:
                self.circuit_breaker.record_failure()

    async def _record_success(self) -> None:
        """
        Record a success, resetting failure state and closing the breaker.
        """
        async with self._lock:
            self.last_success = _utcnow()
            if self.circuit_breaker_enabled:
                self.circuit_breaker.record_success()

    # ------------------------------------------------------------------
    # Request
    # ------------------------------------------------------------------

    @staticmethod
    def _merge(
        payload: dict[str, Any],
        exclude: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Merge payload fields with per-call kwargs.

        Explicit ``kwargs`` win over payload entries so an override never
        raises ``TypeError: got multiple values for keyword argument``.
        """
        merged = {k: v for k, v in payload.items() if k != exclude}
        merged.update(kwargs)
        return merged

    def _build_coroutine(
        self,
        op: str,
        payload: dict[str, Any],
        kwargs: dict[str, Any],
    ):
        context = kwargs.pop("context", None)

        if op == "chat":
            if "prompt" not in payload:
                raise ValueError("The 'chat' operation requires a 'prompt' field.")

            return self.client.chat(
                payload["prompt"],
                **self._merge(payload, "prompt", kwargs),
                context=context,
            )

        if op == "embeddings":
            if "text" not in payload:
                raise ValueError("The 'embeddings' operation requires a 'text' field.")

            return self.client.embeddings(
                payload["text"],
                **self._merge(payload, "text", kwargs),
                context=context,
            )

        if op == "responses":
            return self.client.responses(
                *payload.get("args", ()),
                **self._merge(payload, "args", kwargs),
                context=context,
            )

        raise ValueError(
            f"Unsupported operation '{op}'. "
            f"Supported operations: {', '.join(self.SUPPORTED_OPERATIONS)}."
        )

    async def send(
        self,
        op: str,
        payload: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """
        Dispatch a request to the underlying provider adapter.
        """
        coro = self._build_coroutine(op, payload, kwargs)

        if self.timeout is not None:
            coro = asyncio.wait_for(coro, timeout=self.timeout)

        return await self.execute(coro)

    async def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Stream tokens from the underlying provider adapter.

        The concurrency slot is held for the lifetime of the stream. Consumers
        that stop early should close the generator (``aclose()``, or
        ``contextlib.aclosing``) so the slot is returned promptly.

        A per-token timeout (``stream_timeout`` kwarg or the client's
        ``timeout``) guards against hung streams.
        """

        stream_timeout: float | None = kwargs.pop("stream_timeout", None)
        if stream_timeout is None:
            stream_timeout = self.timeout

        await self.acquire()

        source = None

        try:
            source = (
                self.streaming.stream(prompt, **kwargs)
                if self.streaming is not None
                else self.client.stream(prompt, **kwargs)
            )

            if stream_timeout is not None:
                gen = source.__aiter__()  # type: ignore[union-attr]
                while True:
                    try:
                        token = await asyncio.wait_for(
                            gen.__anext__(),
                            timeout=stream_timeout,
                        )
                        yield token
                    except StopAsyncIteration:
                        return
            else:
                async for token in source:  # type: ignore[union-attr]
                    yield token
        finally:
            if source is not None:
                with suppress(Exception):
                    await source.aclose()  # type: ignore[union-attr]
            await self.release()

    def __repr__(self) -> str:
        return (
            f"ClientNode(api_key={_mask(self.api_key)!r}, "
            f"active={self.active_requests}, failures={self.failures})"
        )


def _mask(api_key: str | None) -> str:
    """
    Render an API key safe for logs: keep the last 4 characters only.
    """
    return mask_api_key(api_key)
