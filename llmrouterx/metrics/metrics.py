from __future__ import annotations

from collections import defaultdict, deque
from statistics import mean, median
from threading import Lock
from typing import Any


class MetricsCollector:
    """
    Thread-safe in-memory metrics collector.

    Keeps a bounded history of timings to avoid
    unbounded memory growth.
    """

    def __init__(
        self,
        *,
        max_samples: int = 1000,
    ) -> None:

        self._lock = Lock()

        self._max_samples = max_samples

        self._counters: dict[str, int] = defaultdict(int)

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
            self._counters[key] += amount

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

    def reset(self) -> None:

        with self._lock:

            self._counters.clear()

            self._timings.clear()