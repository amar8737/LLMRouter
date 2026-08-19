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
        max_timing_keys: int = 500,
    ) -> None:

        self._lock = Lock()

        self._max_samples = max_samples
        self._max_counter_keys = max_counter_keys
        self._max_timing_keys = max_timing_keys

        self._counters: dict[str, int] = {}

        self._labeled_counters: dict[str, dict[str, int]] = defaultdict(dict)

        self._timings: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=max_samples))

        self._labeled_timings: dict[str, dict[str, deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=max_samples))
        )

    # -------------------------------------------------------
    # Counters
    # -------------------------------------------------------

    def incr(
        self,
        key: str,
        amount: int = 1,
        labels: dict[str, str] | None = None,
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

            if labels:
                label_key = _serialize_labels(labels)
                if label_key not in self._labeled_counters[key]:
                    if len(self._labeled_counters[key]) < self._max_counter_keys:
                        self._labeled_counters[key][label_key] = amount
                    else:
                        logger.warning(
                            "Labeled counter '%s' dropped: max %d label sets reached.",
                            key,
                            self._max_counter_keys,
                        )
                else:
                    self._labeled_counters[key][label_key] += amount

    # -------------------------------------------------------
    # Timings
    # -------------------------------------------------------

    def timing(
        self,
        key: str,
        seconds: float,
        labels: dict[str, str] | None = None,
    ) -> None:

        with self._lock:
            if key not in self._timings:
                if len(self._timings) >= self._max_timing_keys:
                    oldest = next(iter(self._timings))
                    self._timings.pop(oldest)
                    self._labeled_timings.pop(oldest, None)
                self._timings[key] = deque(maxlen=self._max_samples)
                self._labeled_timings[key] = defaultdict(
                    lambda: deque(maxlen=self._max_samples)
                )
            self._timings[key].append(seconds)

            if labels:
                label_key = _serialize_labels(labels)
                bucket = self._labeled_timings[key]
                # Mirror the bounding applied to ``_labeled_counters``: cap the
                # number of distinct label sets per timing key so that dynamic,
                # high-cardinality labels can never grow this dict unbounded
                # (which would be a memory leak in long-running processes).
                if label_key not in bucket:
                    if len(bucket) >= self._max_counter_keys:
                        logger.warning(
                            "Labeled timing '%s' dropped: max %d label sets reached.",
                            key,
                            self._max_counter_keys,
                        )
                        return
                    bucket[label_key] = deque(maxlen=self._max_samples)
                bucket[label_key].append(seconds)

    # -------------------------------------------------------
    # Token tracking
    # -------------------------------------------------------

    def track_tokens(
        self,
        provider: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """
        Record token usage for economic/cost tracking.

        Usage is separated from request counts: global counters track
        ``tokens.prompt.total`` / ``tokens.completion.total``, and a per-provider
        ``tokens.total`` labeled counter enables cost attribution.
        """
        with self._lock:
            self._counters["tokens.prompt.total"] = (
                self._counters.get("tokens.prompt.total", 0) + prompt_tokens
            )
            self._counters["tokens.completion.total"] = (
                self._counters.get("tokens.completion.total", 0) + completion_tokens
            )

            label_key = _serialize_labels({"provider": provider})
            bucket = self._labeled_counters["tokens.total"]
            bucket[label_key] = bucket.get(label_key, 0) + prompt_tokens + completion_tokens

    # -------------------------------------------------------
    # Snapshot
    # -------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:

        with self._lock:
            return {
                "counters": dict(self._counters),
                "labeled_counters": {
                    key: dict(values) for key, values in self._labeled_counters.items()
                },
                "timings": {key: list(values) for key, values in self._timings.items()},
                "labeled_timings": {
                    key: {label: list(values) for label, values in values_by_label.items()}
                    for key, values_by_label in self._labeled_timings.items()
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

            self._labeled_counters.clear()

            self._timings.clear()

            self._labeled_timings.clear()


def _serialize_labels(labels: dict[str, str]) -> str:
    """Stable, order-independent string form of a label set."""
    return ",".join(f"{key}={labels[key]}" for key in sorted(labels))
