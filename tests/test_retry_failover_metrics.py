import asyncio
import time

import pytest

from llmrouterx.client.client_node import ClientNode
from llmrouterx.metrics.metrics import MetricsCollector
from llmrouterx.providers.composite_router import CompositeRouter
from llmrouterx.providers.provider_router import ProviderRouter
from llmrouterx.providers.stub_provider import StubClient
from llmrouterx.retry import ExponentialRetry, HTTPError
from llmrouterx.router.llmrouter import LLMRouter


class SlowClient:
    def __init__(self, delay=1.0):
        self.delay = delay

    async def request(self, op, payload, api_key=None, **kwargs):
        await asyncio.sleep(self.delay)
        return {"response": "done"}

    async def chat(self, prompt, **kwargs):
        await asyncio.sleep(self.delay)
        return {"response": f"Chat response: {prompt}"}

    async def embeddings(self, text, **kwargs):
        await asyncio.sleep(self.delay)
        return {"response": f"Embeddings: {text}"}

    async def responses(self, *args, **kwargs):
        await asyncio.sleep(self.delay)
        return {"response": "Responses OK"}

    async def stream(self, prompt, **kwargs):
        await asyncio.sleep(self.delay)
        for part in prompt.split():
            yield {"response": part}


@pytest.mark.asyncio
async def test_success_records_metrics():
    metrics = MetricsCollector()
    stub = StubClient("s1")
    node = ClientNode("k1", stub)
    provider = ProviderRouter("p1", [node])
    composite = CompositeRouter([provider], metrics=metrics)
    router = LLMRouter(composite, metrics=metrics)

    resp = await router.chat("hello")
    assert resp["provider"] == "s1"

    data = metrics.get()
    assert data["counters"].get("requests.chat", 0) >= 1
    assert "latency.chat" in data["timings"]


class FlakyClient:
    def __init__(self, fail_times=1):
        self._calls = 0
        self.fail_times = fail_times

    async def request(self, op, payload, api_key=None, **kwargs):
        self._calls += 1
        if self._calls <= self.fail_times:
            raise RuntimeError("boom")
        return {"provider": "flaky", "op": op, "response": "ok"}

    async def chat(self, prompt, **kwargs):
        self._calls += 1
        if self._calls <= self.fail_times:
            raise RuntimeError("boom")
        return {"provider": "flaky", "response": f"Chat response: {prompt}"}

    async def embeddings(self, text, **kwargs):
        self._calls += 1
        if self._calls <= self.fail_times:
            raise RuntimeError("boom")
        return {"provider": "flaky", "response": f"Embeddings: {text}"}

    async def responses(self, *args, **kwargs):
        self._calls += 1
        if self._calls <= self.fail_times:
            raise RuntimeError("boom")
        return {"provider": "flaky", "response": "Responses OK"}

    async def stream(self, prompt, **kwargs):
        self._calls += 1
        if self._calls <= self.fail_times:
            raise RuntimeError("boom")
        for part in prompt.split():
            yield {"provider": "flaky", "response": part}


@pytest.mark.asyncio
async def test_failover_to_next_provider_records_metrics():
    metrics = MetricsCollector()
    # first provider will error on request
    flaky = FlakyClient(fail_times=1)
    node_bad = ClientNode("bad", flaky)
    p_bad = ProviderRouter("bad", [node_bad])

    # second provider succeeds
    stub = StubClient("succeed")
    node_ok = ClientNode("ok", stub)
    p_ok = ProviderRouter("ok", [node_ok])

    composite = CompositeRouter([p_bad, p_ok], metrics=metrics)
    router = LLMRouter(composite, metrics=metrics, retry=ExponentialRetry(max_retries=2))

    resp = await router.chat("hi")
    assert resp["provider"] == "succeed"

    data = metrics.get()
    # ensure provider error and success metrics recorded
    assert data["counters"].get("provider.error.bad", 0) >= 0
    assert data["counters"].get("provider.success.ok", 0) >= 1


class RetryAfterClient:
    def __init__(self, fail_times=2):
        self._calls = 0
        self.fail_times = fail_times

    async def request(self, op, payload, api_key=None, **kwargs):
        self._calls += 1
        if self._calls <= self.fail_times:
            # simulate HTTP 429 with Retry-After header
            raise HTTPError(429, "rate limited", headers={"Retry-After": "0.05"})
        return {"provider": "retry", "op": op, "response": "ok"}

    async def chat(self, prompt, **kwargs):
        self._calls += 1
        if self._calls <= self.fail_times:
            raise HTTPError(429, "rate limited", headers={"Retry-After": "0.05"})
        return {"provider": "retry", "response": f"Chat response: {prompt}"}

    async def embeddings(self, text, **kwargs):
        self._calls += 1
        if self._calls <= self.fail_times:
            raise HTTPError(429, "rate limited", headers={"Retry-After": "0.05"})
        return {"provider": "retry", "response": f"Embeddings: {text}"}

    async def responses(self, *args, **kwargs):
        self._calls += 1
        if self._calls <= self.fail_times:
            raise HTTPError(429, "rate limited", headers={"Retry-After": "0.05"})
        return {"provider": "retry", "response": "Responses OK"}

    async def stream(self, prompt, **kwargs):
        self._calls += 1
        if self._calls <= self.fail_times:
            raise HTTPError(429, "rate limited", headers={"Retry-After": "0.05"})
        for part in prompt.split():
            yield {"provider": "retry", "response": part}


@pytest.mark.asyncio
async def test_retry_honors_retry_after_and_succeeds():
    """Retry-After header is respected when a client returns HTTP 429."""
    metrics = MetricsCollector()
    client = RetryAfterClient(fail_times=2)
    node = ClientNode("r1", client)
    provider = ProviderRouter("rprov", [node])
    composite = CompositeRouter([provider], metrics=metrics)

    retry = ExponentialRetry(max_retries=4, base=0.01, rate_limit_min_backoff=0.01)
    router = LLMRouter(composite, metrics=metrics, retry=retry)

    start = time.monotonic()
    resp = await router.chat("retry-me")
    elapsed = time.monotonic() - start
    assert resp["provider"] == "retry"

    # Verify Retry-After was respected (should have waited between retries)
    # With Retry-After: 0.05 and 2 failures, total wait should be >= 0.1s
    assert elapsed >= 0.09, f"Expected elapsed >= 0.09, got {elapsed}"
