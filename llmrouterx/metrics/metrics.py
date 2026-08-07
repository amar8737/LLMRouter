from __future__ import annotations

import logging
from collections import defaultdict, deque
from statistics import mean, median
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Thread-safe in-memory metrics collector.

    Keeps a bounded history of timings and a bounded number of
    distinct counter keys to avoid unbounded memory growth.
    """

    def __init__(
        self,
        *,
        max_samples: int = 1000,
        max_counter_keys: int = 500,
    ) -> None:

        self._lock = Lock()

        self._max_samples = max_samples
        self._max_counter_keys = max_counter_keys

        self._counters: dict[str, int] = {}

        self._timings: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )

    # -------------------------------------------------------
    # Counters
    # -------------------------------------------------------

    def incr(
        self,
        key: str,
        amount: int = 1,
    ) -> None:

        with self._lock:
            if key in self._counters:
                self._counters[key] += amount
            elif len(self._counters) < self._max_counter_keys:
                self._counters[key] = amount
            else:
                logger.warning(
                    "Counter key '%s' dropped: max %d keys reached.",
                    key,
                    self._max_counter_keys,
                )

    # -------------------------------------------------------
    # Timings
    # -------------------------------------------------------

    def timing(
        self,
        key: str,
        seconds: float,
    ) -> None:

        with self._lock:
            self._timings[key].append(seconds)

    # -------------------------------------------------------
    # Snapshot
    # -------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:

        with self._lock:

            return {
                "counters": dict(self._counters),
                "timings": {
                    key: list(values)
                    for key, values in self._timings.items()
                },
            }

    # Backward compatibility
    get = snapshot

    # -------------------------------------------------------
    # Statistics
    # -------------------------------------------------------

    def timing_stats(
        self,
        key: str,
    ) -> dict[str, float]:

        with self._lock:
            values = list(
                self._timings.get(
                    key,
                    (),
                )
            )

        if not values:
            return {}

        values.sort()

        count = len(values)

        return {
            "count": count,
            "min": values[0],
            "max": values[-1],
            "mean": mean(values),
            "median": median(values),
            "p95": values[int(count * 0.95)],
            "p99": values[int(count * 0.99)],
        }

    # -------------------------------------------------------
    # Reset
    # -------------------------------------------------------

    def counter_keys(self) -> int:
        """Return the number of distinct counter keys currently tracked."""
        with self._lock:
            return len(self._counters)

    def reset(self) -> None:

        with self._lock:

            self._counters.clear()

            self._timings.clear()