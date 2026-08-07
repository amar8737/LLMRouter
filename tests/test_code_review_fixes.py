import asyncio

import pytest

from llmrouterx.adapters.base import BaseProviderAdapter
from llmrouterx.client.client_node import ClientNode
from llmrouterx.context.request_context import RequestContext
from llmrouterx.metrics.metrics import MetricsCollector
from llmrouterx.providers.composite_router import CompositeRouter
from llmrouterx.providers.provider_router import ProviderRouter
from llmrouterx.providers.stub_provider import StubClient
from llmrouterx.router.llmrouter import LLMRouter

# --- Issue 1: shared httpx.AsyncClient survives event-loop changes ---------


def test_shared_http_client_recreated_across_event_loops():
    from llmrouterx.router import llmrouter as _llmrouter_mod
    from llmrouterx.router.llmrouter import (
        _get_shared_http_client,
        shutdown_shared_http_client,
    )

    # Start clean so the test is independent of module import order.
    asyncio.run(shutdown_shared_http_client())

    async def first():
        return _get_shared_http_client()

    async def second():
        # A second, fresh event loop: the previously cached client was bound to
        # a loop that is now closed and must be rebuilt rather than reused.
        return _get_shared_http_client()

    first_client = asyncio.run(first())
    assert first_client is not None

    second_client = asyncio.run(second())
    assert second_client is not None
    assert second_client is not first_client

    # The cache now points at the new (still-closed-safe) loop.
    assert _llmrouter_mod._SHARED_HTTP_CLIENT is second_client

    asyncio.run(shutdown_shared_http_client())
    assert _llmrouter_mod._SHARED_HTTP_CLIENT is None


# --- Issue 2: bounded labeled timings ---------------------------------------


def test_labeled_timings_are_bounded():
    collector = MetricsCollector(max_counter_keys=3)

    for i in range(10):
        collector.timing(
            "latency.chat",
            0.1,
            labels={"provider": f"p{i}"},
        )

    snap = collector.snapshot()["labeled_timings"]
    assert "latency.chat" in snap
    # The distinct-label cap protects memory just like labeled counters.
    assert len(snap["latency.chat"]) == 3  # 3 distinct label sets retained


def test_labeled_timings_within_limit_are_kept():
    collector = MetricsCollector(max_counter_keys=10)
    for i in range(3):
        collector.timing("latency.chat", 0.1, labels={"provider": f"p{i}"})
    snap = collector.snapshot()["labeled_timings"]["latency.chat"]
    assert len(snap) == 3


# --- Issue 3: total_timeout maps to errors.<op>.total_timeout ---------------


class _SlowClient:
    async def chat(self, prompt, **kwargs):
        await asyncio.sleep(0.5)
        return "slow"

    async def embeddings(self, text, **kwargs):
        await asyncio.sleep(0.5)
        return []

    async def responses(self, *args, **kwargs):
        await asyncio.sleep(0.5)
        return {}

    async def stream(self, prompt, **kwargs):
        yield "x"


def _bounded_total_timeout_router() -> LLMRouter:
    node = ClientNode("k1", _SlowClient())
    provider = ProviderRouter("p1", [node])
    composite = CompositeRouter([provider])
    return LLMRouter(composite, total_timeout=0.1)


def test_total_timeout_records_total_timeout_metric_once():
    router = _bounded_total_timeout_router()

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(router.chat("hi"))

    counters = router.metrics.snapshot()["counters"]
    # A total deadline breach must be attributed to total_timeout, not the
    # generic retry error counter.
    assert counters.get("errors.chat.total_timeout", 0) >= 1
    # No generic per-attempt error increment should be left by a pure timeout.
    assert "errors.chat" not in counters


def test_total_timeout_elapsed_under_budget():
    import time

    router = _bounded_total_timeout_router()
    start = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(router.chat("hi"))
    assert time.monotonic() - start < 0.5


def test_total_timeout_passes_under_limit():
    node = ClientNode("k1", _SlowClient())
    provider = ProviderRouter("p1", [node])
    composite = CompositeRouter([provider])
    router = LLMRouter(composite, total_timeout=5.0)

    result = asyncio.run(router.chat("hi"))
    assert result == "slow"


# --- Issue 4: streaming usage extraction is robust --------------------------


class _Choice:
    def __init__(self, delta=None):
        self.delta = delta


class _Delta:
    def __init__(self, content=None):
        self.content = content


class _Chunk:
    def __init__(self, usage=None, delta_content=None):
        self.usage = usage
        if delta_content is None:
            self.choices = []
        else:
            self.choices = [_Choice(delta=_Delta(content=delta_content))]


class _MalformedUsage:
    # Lacks the expected fields and is not a mapping.
    pass


class _FakeCompletions:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def create(self, **kwargs):
        async def _gen():
            for chunk in self._chunks:
                yield chunk

        return _gen()


class _FakeClient:
    def __init__(self, chunks):
        self.chat = _FakeChat(_FakeCompletions(chunks))


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


def _make_openai_stream_adapter(chunks):
    from llmrouterx.adapters.openai_compatible import OpenAICompatibleAdapter

    return OpenAICompatibleAdapter(client=_FakeClient(chunks), default_model="gpt-4")


def test_stream_with_malformed_usage_does_not_break():
    # A usage chunk whose block is malformed must not terminate the stream.
    chunks = [_Chunk(usage=_MalformedUsage()), _Chunk(delta_content="hello")]
    adapter = _make_openai_stream_adapter(chunks)

    async def collect():
        return [t async for t in adapter.stream("hi")]

    result = asyncio.run(collect())
    assert result == ["hello"]


def test_stream_with_dict_usage_does_not_break():
    # Some providers ship usage as a plain dict; attribute access must not crash
    # and the values must still be captured.
    chunks = [
        _Chunk(usage={"prompt_tokens": 5, "completion_tokens": 2}),
        _Chunk(delta_content="world"),
    ]
    adapter = _make_openai_stream_adapter(chunks)
    ctx = RequestContext(operation="stream")

    async def collect():
        return [t async for t in adapter.stream("hi", context=ctx)]

    result = asyncio.run(collect())
    assert result == ["world"]
    assert ctx.get("usage") == {"prompt_tokens": 5, "completion_tokens": 2}


def test_stream_with_valid_usage_records_tokens():
    class GoodUsage:
        prompt_tokens = 7
        completion_tokens = 3

    chunks = [_Chunk(usage=GoodUsage()), _Chunk(delta_content="ok")]
    adapter = _make_openai_stream_adapter(chunks)

    captured = RequestContext(operation="stream")

    async def collect():
        return [t async for t in adapter.stream("hi", context=captured)]

    asyncio.run(collect())
    assert captured.get("usage") == {"prompt_tokens": 7, "completion_tokens": 3}


def test_extract_usage_tokens_handles_all_shapes():
    assert BaseProviderAdapter._extract_usage_tokens(None) == (0, 0)

    class Attr:
        prompt_tokens = None
        completion_tokens = None

    assert BaseProviderAdapter._extract_usage_tokens(Attr()) == (0, 0)

    assert BaseProviderAdapter._extract_usage_tokens({"prompt_tokens": 3}) == (3, 0)
    assert BaseProviderAdapter._extract_usage_tokens({"completion_tokens": 7}) == (0, 7)

    class Good:
        prompt_tokens = 10
        completion_tokens = 4

    assert BaseProviderAdapter._extract_usage_tokens(Good()) == (10, 4)

    # Non-integer values degrade to 0 rather than raising.
    class Bad:
        prompt_tokens = "nope"
        completion_tokens = None

    assert BaseProviderAdapter._extract_usage_tokens(Bad()) == (0, 0)


def test_record_usage_uses_extraction_helper():
    ctx = RequestContext(operation="chat")
    router = LLMRouter(
        CompositeRouter([ProviderRouter("openai", [ClientNode("k", StubClient("s"))])])
    )

    class Obj:
        class _U:
            prompt_tokens = 42
            completion_tokens = 8

        usage = _U()

    BaseProviderAdapter._record_usage(ctx, Obj())
    assert ctx.get("usage") == {"prompt_tokens": 42, "completion_tokens": 8}

    # And the router records it through its metrics pipeline.
    router.metrics.track_tokens("openai", prompt_tokens=42, completion_tokens=8)
    counters = router.metrics.snapshot()["counters"]
    assert counters["tokens.prompt.total"] == 42
    assert counters["tokens.completion.total"] == 8
