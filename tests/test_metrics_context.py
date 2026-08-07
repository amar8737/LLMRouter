from llmrouterx.context.request_context import RequestContext
from llmrouterx.metrics.metrics import MetricsCollector


def test_counter_increments():
    m = MetricsCollector()
    m.incr("requests.chat")
    m.incr("requests.chat", 4)
    assert m.snapshot()["counters"]["requests.chat"] == 5


def test_counter_keys_bounded():
    m = MetricsCollector(max_counter_keys=3)
    for i in range(3):
        m.incr(f"key{i}")
    m.incr("overflow")  # should be dropped with a warning
    snapshot = m.snapshot()["counters"]
    assert "overflow" not in snapshot
    assert len(snapshot) == 3


def test_counter_keys_reports_count():
    m = MetricsCollector(max_counter_keys=2)
    m.incr("a")
    m.incr("b")
    m.incr("c")
    assert m.counter_keys() == 2


def test_timing_records_values():
    m = MetricsCollector()
    m.timing("latency.chat", 0.1)
    m.timing("latency.chat", 0.2)
    assert m.snapshot()["timings"]["latency.chat"] == [0.1, 0.2]


def test_timing_stats_computes_percentiles():
    m = MetricsCollector()
    for i in range(1, 101):
        m.timing("latency.chat", float(i))
    stats = m.timing_stats("latency.chat")
    assert stats["count"] == 100
    assert stats["min"] == 1.0
    assert stats["max"] == 100.0
    assert stats["mean"] == 50.5
    assert stats["p95"] == 96.0
    assert stats["p99"] == 100.0


def test_timing_stats_empty():
    m = MetricsCollector()
    assert m.timing_stats("nothing.here") == {}


def test_timing_samples_bounded():
    m = MetricsCollector(max_samples=5)
    for i in range(20):
        m.timing("latency.chat", float(i))
    stats = m.timing_stats("latency.chat")
    assert stats["count"] == 5
    assert stats["max"] == 19.0


def test_get_alias_for_snapshot():
    m = MetricsCollector()
    m.incr("requests.chat")
    assert m.get() == m.snapshot()


def test_reset_clears_everything():
    m = MetricsCollector()
    m.incr("a")
    m.timing("latency.chat", 0.5)
    m.reset()
    assert m.snapshot() == {"counters": {}, "timings": {}}


def test_request_context_generates_unique_ids():
    a = RequestContext()
    b = RequestContext()
    assert a.request_id != b.request_id


def test_request_context_elapsed_while_running():
    import time

    ctx = RequestContext()
    ctx.started_at = time.perf_counter() - 0.25
    assert 0.2 <= ctx.elapsed() <= 0.3


def test_request_context_finish_records_time():
    ctx = RequestContext()
    ctx.finish()
    assert ctx.finished_at is not None
    assert ctx.elapsed() >= 0


def test_request_context_increment_retry():
    ctx = RequestContext()
    assert ctx.retry_count == 0
    ctx.increment_retry()
    ctx.increment_retry()
    assert ctx.retry_count == 2


def test_request_context_set_get():
    ctx = RequestContext()
    ctx.set("tokens", 42)
    assert ctx.get("tokens") == 42
    assert ctx.get("missing", "fallback") == "fallback"
    assert ctx.get("missing") is None


def test_request_context_copy_is_independent():
    ctx = RequestContext(operation="chat", retry_count=2)
    ctx.set("tokens", 5)

    clone = ctx.copy()
    clone.set("tokens", 99)
    clone.retry_count = 10

    assert ctx.get("tokens") == 5
    assert ctx.retry_count == 2
    assert clone.get("tokens") == 99
    assert clone.retry_count == 10
    assert ctx.operation == clone.operation == "chat"
