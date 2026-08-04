
import threading
from collections import defaultdict
from statistics import mean, median
from typing import Dict, List


class MetricsCollector:
    """Thread-safe in-memory metrics collector for counters and timings."""

    def __init__(self):
        self._lock = threading.Lock()
        self.counters: Dict[str, int] = defaultdict(int)
        self.timings: Dict[str, List[float]] = defaultdict(list)

    def incr(self, key: str, amount: int = 1):
        with self._lock:
            self.counters[key] += amount

    def timing(self, key: str, seconds: float):
        with self._lock:
            self.timings[key].append(seconds)

    def get(self):
        with self._lock:
            # shallow copy to avoid exposing internal structures
            return {"counters": dict(self.counters), "timings": {k: list(v) for k, v in self.timings.items()}}

    def timing_stats(self, key: str):
        with self._lock:
            vals = list(self.timings.get(key, []))
        if not vals:
            return {}
        vals.sort()
        return {
            "count": len(vals),
            "min": vals[0],
            "max": vals[-1],
            "mean": mean(vals),
            "median": median(vals),
        }
