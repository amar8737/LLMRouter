from __future__ import annotations

import logging
import time
from enum import Enum
from threading import Lock

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.

    States:
        CLOSED   — normal operation, requests pass through.
        OPEN     — failure threshold reached, requests are rejected immediately.
        HALF_OPEN — after reset_timeout, trial requests are allowed.

    The circuit breaker tracks failures per provider. When the number of
    consecutive failures exceeds ``failure_threshold``, the circuit opens.
    After ``reset_timeout`` seconds, it transitions to HALF_OPEN. Successful
    requests in HALF_OPEN gradually decrement the failure count based on
    ``success_decay_factor``; when the count reaches zero, the circuit closes.
    A failure in HALF_OPEN immediately re-opens the circuit.

    The ``state`` property is a pure read. Time-based transitions are explicit:
    call :meth:`maybe_advance` (OPEN -> HALF_OPEN) or :meth:`reset_if_expired`
    (OPEN -> CLOSED, full recovery) at the appropriate point in the request
    flow.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        success_decay_factor: float = 1.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._half_open_max_calls = half_open_max_calls
        self._success_decay_factor = max(0.0, success_decay_factor)
        self._lock = Lock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._half_open_successes = 0

    @property
    def state(self) -> CircuitState:
        """Current breaker state. Pure read — no side effects."""
        with self._lock:
            return self._state

    @property
    def failure_count(self) -> int:
        """Number of consecutive failures currently recorded."""
        with self._lock:
            return self._failure_count

    def maybe_advance(self) -> None:
        """
        Transition OPEN -> HALF_OPEN once the reset timeout has elapsed.

        Intended for the request-gating path (``allow_request``): a breaker
        that has cooled down allows a limited number of trial requests.
        """
        with self._lock:
            if (
                self._state == CircuitState.OPEN
                and time.monotonic() - self._last_failure_time >= self._reset_timeout
            ):
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._half_open_successes = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")

    def reset_if_expired(self) -> bool:
        """
        Fully recover an OPEN breaker once its cooldown has elapsed.

        Returns ``True`` if the breaker was reset, ``False`` otherwise. Used
        by health checks that treat "cooldown expired" as a full recovery.
        """
        with self._lock:
            if (
                self._state == CircuitState.OPEN
                and time.monotonic() - self._last_failure_time >= self._reset_timeout
            ):
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
                self._half_open_successes = 0
                logger.info("Circuit breaker recovered to CLOSED after cooldown")
                return True
            return False

    def cooldown_remaining(self) -> float | None:
        """
        Seconds left before an OPEN breaker may retry, or None when not OPEN.
        """
        with self._lock:
            if self._state != CircuitState.OPEN:
                return None
            remaining = self._last_failure_time + self._reset_timeout - time.monotonic()
            return max(0.0, remaining)

    def allow_request(self) -> bool:
        # Perform the OPEN -> HALF_OPEN transition and the gated decision
        # atomically under a single lock so a failure recorded by a concurrent
        # waiter cannot reopen the breaker between the transition check and the
        # slot allocation (TOCTOU).
        with self._lock:
            if (
                self._state == CircuitState.OPEN
                and time.monotonic() - self._last_failure_time >= self._reset_timeout
            ):
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._half_open_successes = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self._half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    def record_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                # Default behavior (decay_factor >= 1.0): single success fully recovers
                # Gradual decay (decay_factor < 1.0): decrement failure count by decay_factor
                if self._success_decay_factor >= 1.0:
                    self._failure_count = 0
                    self._state = CircuitState.CLOSED
                    self._half_open_calls = 0
                    successes = self._half_open_successes
                    self._half_open_successes = 0
                    logger.info(
                        "Circuit breaker recovered to CLOSED after %d successful trial(s)",
                        successes,
                    )
                else:
                    self._failure_count = int(
                        max(0, self._failure_count - self._success_decay_factor)
                    )
                    if self._failure_count <= 0:
                        self._failure_count = 0
                        self._state = CircuitState.CLOSED
                        self._half_open_calls = 0
                        successes = self._half_open_successes
                        self._half_open_successes = 0
                        logger.info(
                            "Circuit breaker recovered to CLOSED after %d successful trial(s)",
                            successes,
                        )
                return

            self._failure_count = 0
            self._state = CircuitState.CLOSED
            self._half_open_calls = 0
            self._half_open_successes = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._half_open_calls = 0
                self._half_open_successes = 0
                logger.warning(
                    "Circuit breaker re-opened after HALF_OPEN failure (total failures: %d)",
                    self._failure_count,
                )
                return

            if self._failure_count >= self._failure_threshold and self._state != CircuitState.OPEN:
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
            self._half_open_successes = 0

    def __repr__(self) -> str:
        with self._lock:
            state = self._state
            failures = self._failure_count
        return f"CircuitBreaker(state={state.value}, failures={failures})"
