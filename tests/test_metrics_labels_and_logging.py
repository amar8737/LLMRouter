import asyncio
import io
import json
import logging

from llmrouterx.client.client_node import ClientNode
from llmrouterx.metrics.metrics import MetricsCollector
from llmrouterx.providers.composite_router import CompositeRouter
from llmrouterx.providers.provider_router import ProviderRouter
from llmrouterx.providers.stub_provider import StubClient
from llmrouterx.router.llmrouter import LLMRouter
from llmrouterx.utils.logging import JsonFormatter, setup_logging

# --- Fix 8: metrics labels -------------------------------------------


def test_incr_with_labels():
    m = MetricsCollector()
    m.incr("requests.chat", labels={"provider": "openai"})
    m.incr("requests.chat", labels={"provider": "openai"})
    m.incr("requests.chat", labels={"provider": "anthropic"})

    snapshot = m.snapshot()
    assert snapshot["counters"]["requests.chat"] == 3
    assert snapshot["labeled_counters"]["requests.chat"]["provider=openai"] == 2
    assert snapshot["labeled_counters"]["requests.chat"]["provider=anthropic"] == 1


def test_incr_without_labels_keeps_old_shape():
    m = MetricsCollector()
    m.incr("plain")
    snapshot = m.snapshot()
    assert snapshot["counters"] == {"plain": 1}
    assert snapshot["labeled_counters"] == {}


def test_timing_with_labels():
    m = MetricsCollector()
    m.timing("latency.chat", 0.1, labels={"provider": "openai"})
    m.timing("latency.chat", 0.2, labels={"provider": "openai"})

    snapshot = m.snapshot()
    assert snapshot["timings"]["latency.chat"] == [0.1, 0.2]
    assert snapshot["labeled_timings"]["latency.chat"]["provider=openai"] == [0.1, 0.2]


def test_label_order_is_irrelevant():
    m = MetricsCollector()
    m.incr("k", labels={"a": "1", "b": "2"})
    m.incr("k", labels={"b": "2", "a": "1"})
    assert m.snapshot()["labeled_counters"]["k"]["a=1,b=2"] == 2


def test_labeled_timing_stats_not_exposed():
    # timing_stats should only consider the unlabeled series.
    m = MetricsCollector()
    m.timing("latency.chat", 0.5, labels={"provider": "openai"})
    assert m.timing_stats("latency.chat")["count"] == 1


def test_reset_clears_labels():
    m = MetricsCollector()
    m.incr("k", labels={"a": "b"})
    m.reset()
    assert m.snapshot()["labeled_counters"] == {}


def test_router_metrics_include_provider_label():
    node = ClientNode("k1", StubClient("s1"))
    provider = ProviderRouter("myprov", [node])
    composite = CompositeRouter([provider])
    metrics = MetricsCollector()
    router = LLMRouter(composite, metrics=metrics)

    asyncio.run(router.chat("hi"))
    snapshot = metrics.snapshot()
    assert snapshot["labeled_counters"]["requests.chat"]["provider=myprov"] == 1


# --- Fix 9: structured logging ----------------------------------------


def test_json_formatter_emits_valid_json():
    logger = logging.getLogger("llmrouterx.utils.logging.test")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    logger.info("routed", extra={"_llmrouterx_extra": {"request_id": "abc", "provider": "openai"}})
    logger.handlers.remove(handler)

    record = json.loads(stream.getvalue().strip())
    assert record["message"] == "routed"
    assert record["level"] == "INFO"
    assert record["request_id"] == "abc"
    assert record["provider"] == "openai"
    assert "ts" in record


def test_json_formatter_includes_exc_info():
    logger = logging.getLogger("llmrouterx.utils.logging.test.exc")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    try:
        raise ValueError("boom")
    except ValueError:
        logger.error("failed", exc_info=True)
    logger.handlers.remove(handler)

    record = json.loads(stream.getvalue().strip())
    assert "ValueError: boom" in record["exc_info"]


def test_setup_logging_plain_and_json():
    stream = io.StringIO()
    setup_logging(level=logging.INFO, fmt="json", stream=stream)
    logger = logging.getLogger("llmrouterx")
    logger.info("hello-json")
    parsed = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert parsed["message"] == "hello-json"

    stream = io.StringIO()
    setup_logging(level=logging.INFO, fmt="plain", stream=stream)
    logger = logging.getLogger("llmrouterx")
    logger.info("hello-plain")
    assert "hello-plain" in stream.getvalue()
