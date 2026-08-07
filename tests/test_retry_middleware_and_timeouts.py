import asyncio
from contextlib import suppress

import pytest

from llmrouterx.client.client_node import ClientNode
from llmrouterx.exceptions import StreamError
from llmrouterx.middleware.base import BaseMiddleware
from llmrouterx.providers.composite_router import CompositeRouter
from llmrouterx.providers.provider_router import ProviderRouter
from llmrouterx.providers.stub_provider import StubClient
from llmrouterx.retry.exponential import ExponentialRetry, HTTPError
from llmrouterx.router.llmrouter import LLMRouter


def make_router(*middleware, **kwargs):
    node = ClientNode("k1", StubClient("s1"))
    provider = ProviderRouter("p1", [node])
    composite = CompositeRouter([provider])
    return LLMRouter(composite, middleware=list(middleware), **kwargs)


# --- Fix 1: on_retry hook ---------------------------------------------


class VetoRetry(BaseMiddleware):
    def __init__(self):
        self.calls = []

    async def on_retry(self, operation, payload, exception, attempt, context):
        self.calls.append((attempt, str(exception)))
        return False  # cancel all retries


class ObserveRetry(BaseMiddleware):
    def __init__(self):
        self.calls = []

    async def on_retry(self, operation, payload, exception, attempt, context):
        self.calls.append((attempt, str(exception)))
        return True


class FailAlways:
    async def chat(self, prompt, **kwargs):
        raise HTTPError(500, "boom")

    async def embeddings(self, text, **kwargs):
        raise HTTPError(500, "boom")

    async def responses(self, *args, **kwargs):
        raise HTTPError(500, "boom")

    async def stream(self, prompt, **kwargs):
        raise HTTPError(500, "boom")
        yield  # pragma: no cover


def test_on_retry_can_cancel_retries():
    mw = VetoRetry()
    node = ClientNode("k1", FailAlways())
    provider = ProviderRouter("p1", [node])
    composite = CompositeRouter([provider])
    router = LLMRouter(composite, middleware=[mw], retry=ExponentialRetry(max_retries=5))

    with pytest.raises(HTTPError):
        asyncio.run(router.chat("hi"))

    assert len(mw.calls) == 1
    assert mw.calls[0][0] == 1


def test_on_retry_observer_preserves_retries():
    mw = ObserveRetry()
    node = ClientNode("k1", FailAlways())
    provider = ProviderRouter("p1", [node])
    composite = CompositeRouter([provider])
    router = LLMRouter(composite, middleware=[mw], retry=ExponentialRetry(max_retries=3))

    with pytest.raises(HTTPError):
        asyncio.run(router.chat("hi"))

    # max_retries=3 allows 3 retries after the initial failure.
    assert len(mw.calls) == 3


def test_on_retry_hook_error_does_not_break_retries():
    class Exploding(BaseMiddleware):
        async def on_retry(self, operation, payload, exception, attempt, context):
            raise RuntimeError("observer boom")

    node = ClientNode("k1", FailAlways())
    provider = ProviderRouter("p1", [node])
    composite = CompositeRouter([provider])
    router = LLMRouter(composite, middleware=[Exploding()], retry=ExponentialRetry(max_retries=2))

    with pytest.raises(HTTPError):
        asyncio.run(router.chat("hi"))


class RecoversAfter(BaseMiddleware):
    def __init__(self, veto_attempt=2):
        self.veto_attempt = veto_attempt
        self.calls = []

    async def on_retry(self, operation, payload, exception, attempt, context):
        self.calls.append(attempt)
        return attempt < self.veto_attempt


def test_on_retry_can_allow_bounded_retries_then_fail():
    mw = RecoversAfter(veto_attempt=2)
    node = ClientNode("k1", FailAlways())
    provider = ProviderRouter("p1", [node])
    composite = CompositeRouter([provider])
    router = LLMRouter(composite, middleware=[mw], retry=ExponentialRetry(max_retries=5))

    with pytest.raises(HTTPError):
        asyncio.run(router.chat("hi"))
    assert mw.calls == [1, 2]


# --- Fix 2: StreamError -----------------------------------------------


def test_stream_raises_streamerror_on_failure():
    node = ClientNode("k1", FailAlways())
    provider = ProviderRouter("p1", [node])
    composite = CompositeRouter([provider])
    router = LLMRouter(composite)

    with pytest.raises(StreamError):
        asyncio.run(drain(router.stream("hi")))


async def drain(agen):
    return [t async for t in agen]


# --- Fix 4: total_timeout ---------------------------------------------


class SlowClient:
    async def chat(self, prompt, **kwargs):
        await asyncio.sleep(0.5)
        return "slow-done"

    async def embeddings(self, text, **kwargs):
        await asyncio.sleep(0.5)
        return []

    async def responses(self, *args, **kwargs):
        await asyncio.sleep(0.5)
        return {}

    async def stream(self, prompt, **kwargs):
        await asyncio.sleep(0.5)
        yield "x"


def test_total_timeout_rejects_non_positive():

    with pytest.raises(ValueError):
        LLMRouter(CompositeRouter([]), total_timeout=0)


def test_total_timeout_raises_when_exceeded():
    node = ClientNode("k1", SlowClient(), timeout=None)
    provider = ProviderRouter("p1", [node])
    composite = CompositeRouter([provider])
    router = LLMRouter(composite, total_timeout=0.1)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(router.chat("hi"))


def test_total_timeout_ok_when_under_limit():
    node = ClientNode("k1", SlowClient(), timeout=None)
    provider = ProviderRouter("p1", [node])
    composite = CompositeRouter([provider])
    router = LLMRouter(composite, total_timeout=5.0)

    result = asyncio.run(router.chat("hi"))
    assert result == "slow-done"


def test_total_timeout_cancels_inflight_request():
    import time

    node = ClientNode("k1", SlowClient(), timeout=None)
    provider = ProviderRouter("p1", [node])
    composite = CompositeRouter([provider])
    router = LLMRouter(composite, total_timeout=0.1)

    start = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(router.chat("hi"))
    elapsed = time.monotonic() - start

    # The provider call sleeps 0.5s; the global deadline must interrupt it
    # instead of waiting for it to finish (i.e. a real cancellation).
    assert elapsed < 0.4


# --- Fix 5: health check timeout --------------------------------------


class HungryHealth:
    name = "hungry"

    async def is_healthy(self):
        await asyncio.sleep(10)
        return True

    async def handle(self, op, payload, **kwargs):
        return {"ok": True}

    async def stream(self, prompt, **kwargs):
        yield "x"


def test_health_swallows_hung_provider():
    composite = CompositeRouter([HungryHealth()])
    result = asyncio.run(composite.health(timeout=0.05))
    assert result == {"hungry": False}


# --- Fix 6: transient failure counting --------------------------------


def test_transient_failures_not_counted_by_default():
    node = ClientNode("k1", StubClient("a"), failure_threshold=2, cooldown_seconds=60)

    async def fail_with_transient():
        with suppress(HTTPError):
            await node.execute(_raise_http_500())
        assert node.failures == 0

    asyncio.run(fail_with_transient())


async def _raise_http_500():
    raise HTTPError(500, "boom")


def test_permanent_4xx_not_counted_as_failure():
    node = ClientNode("k1", StubClient("a"), failure_threshold=1, cooldown_seconds=60)

    async def fail_with_401():
        with suppress(HTTPError):
            await node.execute(_raise_http_401())
        assert node.failures == 0
        assert await node.is_healthy() is True

    asyncio.run(fail_with_401())


async def _raise_http_401():
    raise HTTPError(401, "unauthorized")


def test_non_retryable_5xx_still_counts_as_failure():
    node = ClientNode("k1", StubClient("a"), failure_threshold=1, cooldown_seconds=60)

    async def fail_with_501():
        with suppress(HTTPError):
            await node.execute(_raise_http_501())
        assert node.failures == 1
        assert await node.is_healthy() is False

    asyncio.run(fail_with_501())


async def _raise_http_501():
    raise HTTPError(501, "not implemented")


def test_transient_failures_counted_when_opted_in():
    node = ClientNode(
        "k1",
        StubClient("a"),
        failure_threshold=2,
        cooldown_seconds=60,
        count_transient_failures=True,
    )

    async def fail_twice():
        for _ in range(2):
            with suppress(HTTPError):
                await node.execute(_raise_http_500())
        assert node.failures == 2
        assert await node.is_healthy() is False

    asyncio.run(fail_twice())


# --- Fix 7: provider / key attribution --------------------------------


def test_context_gets_provider_and_key():
    node = ClientNode("sekret-key-1", StubClient("s1"))
    provider = ProviderRouter("myprov", [node])
    composite = CompositeRouter([provider])

    captured = {}

    class Capture(BaseMiddleware):
        async def after_response(self, operation, payload, response, context):
            captured["provider"] = context.provider
            captured["api_key"] = context.api_key
            return response

    router = LLMRouter(composite, middleware=[Capture()])
    asyncio.run(router.chat("hi"))

    assert captured["provider"] == "myprov"
    assert captured["api_key"] == "sekret-key-1"


def test_context_attribution_after_failover():
    class Broken:
        async def chat(self, prompt, **kwargs):
            raise HTTPError(500, "boom")

        async def embeddings(self, text, **kwargs):
            raise HTTPError(500, "boom")

        async def responses(self, *args, **kwargs):
            raise HTTPError(500, "boom")

        async def stream(self, prompt, **kwargs):
            raise HTTPError(500, "boom")
            yield  # pragma: no cover

    bad = ProviderRouter("bad", [ClientNode("bkey", Broken())])
    good = ProviderRouter("good", [ClientNode("gkey", StubClient("s"))])
    composite = CompositeRouter([bad, good])
    router = LLMRouter(composite)

    captured = {}

    class Capture(BaseMiddleware):
        async def after_response(self, operation, payload, response, context):
            captured["provider"] = context.provider
            captured["api_key"] = context.api_key
            return response

    router = LLMRouter(composite, middleware=[Capture()])
    asyncio.run(router.chat("hi"))

    assert captured["provider"] == "good"
    assert captured["api_key"] == "gkey"
