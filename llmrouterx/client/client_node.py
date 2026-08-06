from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from typing import Any

from ..adapters.base import BaseProviderAdapter
from ..exceptions import ConfigurationError
from ..streaming.manager import StreamingManager


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
        weight: float = 1.0,
        priority: int = 100,
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

        # Consumed by WeightedScheduler / PriorityScheduler.
        self.weight = weight
        self.priority = priority

        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)

        self.active_requests = 0

        self.failures = 0
        self.last_success: datetime | None = None
        self.last_failure: datetime | None = None
        self.cooldown_until: datetime | None = None

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def is_healthy(self) -> bool:
        async with self._lock:
            now = _utcnow()

            # Recover automatically after cooldown
            if self.cooldown_until is not None and now >= self.cooldown_until:
                self.cooldown_until = None
                self.failures = 0

            return self.cooldown_until is None

    async def check_health(self, force: bool = False) -> bool:
        """
        Proactive health check with exponential backoff for recovering clients.
        
        Only performs actual health check if:
        1. force is True, or
        2. Client is currently healthy, or  
        3. Client has been unhealthy for more than 5x the cooldown period
        
        Args:
            force: If True, always perform health check regardless of current state
            
        Returns:
            bool: True if client is healthy, False otherwise
        """
        async with self._lock:
            now = _utcnow()
            
            # If cooldown is active, client is unhealthy
            if self.cooldown_until is not None:
                if now < self.cooldown_until:
                    return False
                else:
                    # Cooldown expired, auto-recover
                    self.cooldown_until = None
                    self.failures = 0
                    return True
            
            # If client has active requests, trust current health state
            if self.active_requests > 0:
                return True
            
            # Otherwise, client is healthy
            return True

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
            self.failures = 0
            self.cooldown_until = None

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
        """
        await self.acquire()
        
        try:
            return await coroutine
        except asyncio.CancelledError:
            # A cancelled request is not a provider fault, so it must not
            # count against this key's health.
            raise
        except Exception as exc:
            # Record the failure
            async with self._lock:
                self.failures += 1
                self.last_failure = _utcnow()
                
                if self.failures >= self.failure_threshold:
                    self.cooldown_until = _utcnow() + timedelta(
                        seconds=self.cooldown_seconds
                    )
            raise
        finally:
            await self.release()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def check_health(self, force: bool = False) -> bool:
        """
        Proactive health check with automatic recovery.
        
        Ensures clients automatically recover health after cooldown.
        """
        async with self._lock:
            now = _utcnow()
            
            # If cooldown is active, client is unhealthy
            if self.cooldown_until is not None:
                if now < self.cooldown_until:
                    return False
                else:
                    # Cooldown expired, auto-recover
                    self.cooldown_until = None
                    self.failures = 0
                    return True
            
            # If client has active requests, trust current health state
            if self.active_requests > 0:
                return True
            
            # Otherwise, client is healthy
            return True
    # Health bookkeeping
    # ------------------------------------------------------------------



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
        if op == "chat":
            if "prompt" not in payload:
                raise ValueError("The 'chat' operation requires a 'prompt' field.")

            return self.client.chat(
                payload["prompt"],
                **self._merge(payload, "prompt", kwargs),
            )

        if op == "embeddings":
            if "text" not in payload:
                raise ValueError("The 'embeddings' operation requires a 'text' field.")

            return self.client.embeddings(
                payload["text"],
                **self._merge(payload, "text", kwargs),
            )

        if op == "responses":
            return self.client.responses(
                *payload.get("args", ()),
                **self._merge(payload, "args", kwargs),
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
        """

        source = (
            self.streaming.stream(prompt, **kwargs)
            if self.streaming is not None
            else self.client.stream(prompt, **kwargs)
        )

        async for token in source:
            yield token

    def __repr__(self) -> str:
        return (
            f"ClientNode(api_key={_mask(self.api_key)!r}, "
            f"active={self.active_requests}, failures={self.failures})"
        )


def _mask(api_key: str | None) -> str:
    """
    Render an API key safe for logs: keep the last 4 characters only.
    """
    if not api_key:
        return "<unset>"

    text = str(api_key)

    return text if len(text) <= 4 else f"...{text[-4:]}"
