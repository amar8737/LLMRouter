import time


class MetricsCollector:
    def __init__(self):
        self.counters = {}
        self.timings = {}

    def incr(self, key: str, amount: int = 1):
        self.counters[key] = self.counters.get(key, 0) + amount

    def timing(self, key: str, seconds: float):
        vals = self.timings.setdefault(key, [])
        vals.append(seconds)

    def get(self):
        return {"counters": self.counters, "timings": self.timings}
