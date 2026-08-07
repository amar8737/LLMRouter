from __future__ import annotations

import logging
import time
from enum import Enum
from threading import Lock

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):  # noqa: UP042 — StrEnum requires 3.11+
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


def _monotonic() -> float:
    return time.monotonic()


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.

    States:
        CLOSED   — normal operation, requests pass through.
        OPEN     — failure threshold reached, requests are rejected immediately.
        HALF_OPEN — after reset_timeout, one trial request is allowed.

    The circuit breaker tracks failures per provider. When the number of
    consecutive failures exceeds ``failure_threshold``, the circuit opens.
    After ``reset_timeout`` seconds, it transitions to HALF_OPEN. A successful
    request in HALF_OPEN closes the circuit; a failure re-opens it.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._half_open_max_calls = half_open_max_calls
        self._lock = Lock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if (
                self._state == CircuitState.OPEN
                and time.monotonic() - self._last_failure_time >= self._reset_timeout
            ):
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")
            return self._state

    def allow_request(self) -> bool:
        current = self.state
        if current == CircuitState.CLOSED:
            return True
        if current == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self._half_open_max_calls:
                    self._half_open_calls += 1
                    return True
            return False
        return False

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED
            self._half_open_calls = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker re-opened after HALF_OPEN failure "
                    "(total failures: %d)",
                    self._failure_count,
                )
                return

            if (
                self._failure_count >= self._failure_threshold
                and self._state != CircuitState.OPEN
            ):
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker opened after %d consecutive failures.",
                    self._failure_count,
                )

    def reset(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(state={self.state.value}, "
            f"failures={self._failure_count})"
        )
