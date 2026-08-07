import asyncio
from contextlib import aclosing, suppress

import pytest

from llmrouterx.adapters.base import translate_sdk_error
from llmrouterx.adapters.openai_compatible import OpenAICompatibleAdapter
from llmrouterx.client.client_node import ClientNode
from llmrouterx.providers.composite_router import CompositeRouter
from llmrouterx.providers.provider_router import ProviderRouter
from llmrouterx.retry.exponential import HTTPError
from llmrouterx.router.llmrouter import LLMRouter


class StringClient:
    async def stream(self, prompt, **kwargs):
        for part in prompt.split():
            await asyncio.sleep(0.01)
            yield part

    async def chat(self, prompt, **kwargs):
        return "ok"

    async def embeddings(self, text, **kwargs):
        return [0.0]

    async def responses(self, *args, **kwargs):
        return "ok"

    async def health_check(self):
        return True


class SDKError(Exception):
    def __init__(self, status_code, message, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}


class _Completions:
    def __init__(self, handler):
        self._handler = handler

    async def create(self, **kwargs):
        return await self._handler("chat", kwargs)


class _Responses:
    async def create(self, *args, **kwargs):
        return "ok"


class _Embeddings:
    async def create(self, **kwargs):
        return type("R", (), {"data": [type("D", (), {"embedding": [0.0]})()]})()


class _Models:
    async def list(self, **kwargs):
        return []


def _content(text: str):
    """Build a response object shaped like ``choices[0].message.content``."""
    return type(
        "Message",
        (),
        {"content": text},
    )()


def _chat_response(text: str):
    return type(
        "Choice",
        (),
        {"message": _content(text)},
    )()


def _chat_obj(text: str):
    return type(
        "Response",
        (),
        {"choices": [_chat_response(text)]},
    )()


class SDKClientBase:
    """Mimics the OpenAI SDK object shape the adapter expects.

    ``client.chat.completions.create`` etc.
    """

    def __init__(self):
        self._handler = None
        self.chat = type(
            "Chat",
            (),
            {"completions": _Completions(self._handle)},
        )()
        self.responses = _Responses()
        self.embeddings = _Embeddings()
        self.models = _Models()

    async def _handle(self, op, kwargs):
        if self._handler is None:
            raise NotImplementedError
        return await self._handler(op, kwargs)

    def set_handler(self, handler):
        self._handler = handler

    async def health_check(self):
        return True


class FailingClient(SDKClientBase):
    def __init__(self, fail_times=1, status=500):
        super().__init__()
        self._calls = 0
        self.fail_times = fail_times
        self.status = status
        self.set_handler(self._handle)

    async def _handle(self, op, kwargs):
        self._calls += 1
        if self._calls <= self.fail_times:
            raise SDKError(self.status, "boom")
        return _chat_obj("recovered")


class HungStreamClient(StringClient):
    def __init__(self, started):
        self.started = started

    async def stream(self, prompt, **kwargs):
        self.started.set()
        await asyncio.sleep(10)
        yield "never"


@pytest.mark.asyncio
async def test_stream_returns_strings():
    node = ClientNode("k1", StringClient())
    provider = ProviderRouter("p1", [node])
    router = LLMRouter(CompositeRouter([provider]))

    text = await router.stream_to_text("hello world foo")
    assert text == "helloworldfoo"


@pytest.mark.asyncio
async def test_stream_short_circuit_string():
    from llmrouterx.middleware.base import BaseMiddleware, MiddlewareResult

    class Cache(BaseMiddleware):
        async def before_request(self, operation, payload, context):
            if operation == "stream":
                return MiddlewareResult(response="cached-full-response", stop=True)

    node = ClientNode("k1", StringClient())
    provider = ProviderRouter("p1", [node])
    router = LLMRouter(CompositeRouter([provider]), middleware=[Cache()])

    out = await router.stream_to_text("whatever")
    assert out == "cached-full-response"


@pytest.mark.asyncio
async def test_stream_early_close_releases_task():
    node = ClientNode("k1", StringClient())
    provider = ProviderRouter("p1", [node])
    router = LLMRouter(CompositeRouter([provider]))

    agen = router.stream("a b c")
    async with aclosing(agen) as s:
        async for _token in s:
            break

    await asyncio.sleep(0.1)
    assert len(router._active_tasks) == 0


@pytest.mark.asyncio
async def test_stream_early_close_cancels_worker():
    started = asyncio.Event()
    node = ClientNode("k1", HungStreamClient(started))
    provider = ProviderRouter("p1", [node])
    router = LLMRouter(CompositeRouter([provider]))

    agen = router.stream("x")
    async with aclosing(agen) as s:
        async for _ in s:
            pass

    assert started.is_set()
    await asyncio.sleep(0.1)
    assert len(router._active_tasks) == 0


@pytest.mark.asyncio
async def test_stream_releases_global_semaphore():
    node = ClientNode("k1", StringClient())
    provider = ProviderRouter("p1", [node])
    router = LLMRouter(CompositeRouter([provider]), max_concurrent_requests=1)

    await router.stream_to_text("a b c")
    # Semaphore should be released: a second request must not block.
    await asyncio.wait_for(router.stream_to_text("d"), timeout=1.0)


@pytest.mark.asyncio
async def test_stream_respects_client_concurrency_slot():
    class SlowStream(StringClient):
        async def stream(self, prompt, **kwargs):
            yield "first"
            await asyncio.sleep(0.5)
            yield "second"

    node = ClientNode("k1", SlowStream(), max_concurrent=1)
    provider = ProviderRouter("p1", [node])
    router = LLMRouter(CompositeRouter([provider]))

    async with aclosing(router.stream("a b c")) as s:
        async for _ in s:
            # While streaming, the slot is held.
            assert node.active_requests == 1
            break

    assert node.active_requests == 0


def test_translate_sdk_error_maps_status_code():
    exc = SDKError(429, "rate limited", headers={"Retry-After": "5"})
    translated = translate_sdk_error(exc)
    assert isinstance(translated, HTTPError)
    assert translated.status_code == 429
    assert translated.headers == {"Retry-After": "5"}


def test_translate_sdk_error_maps_connection_by_name():
    class APIConnectionError(Exception):
        pass

    translated = translate_sdk_error(APIConnectionError("boom"))
    assert isinstance(translated, HTTPError)
    assert translated.status_code == 503


def test_translate_sdk_error_passes_through_native():
    exc = HTTPError(500, "native")
    assert translate_sdk_error(exc) is exc

    err = ValueError("plain")
    assert translate_sdk_error(err) is err


@pytest.mark.asyncio
async def test_adapter_translates_sdk_errors():
    adapter = OpenAICompatibleAdapter(client=FailingClient(), default_model="m")

    with pytest.raises(HTTPError) as excinfo:
        await adapter.chat("hi")

    assert excinfo.value.status_code == 500


@pytest.mark.asyncio
async def test_retry_recovers_after_sdk_translation():
    class SDKClient(SDKClientBase):
        def __init__(self):
            super().__init__()
            self._calls = 0
            self.set_handler(self._handle)

        async def _handle(self, op, kwargs):
            self._calls += 1
            if self._calls <= 2:
                raise SDKError(500, "boom")
            return _chat_obj("recovered")

    sdk_client = SDKClient()
    adapter = OpenAICompatibleAdapter(client=sdk_client, default_model="m")
    node = ClientNode("k1", adapter)
    provider = ProviderRouter("p1", [node])
    composite = CompositeRouter([provider])
    retry_cls = __import__(
        "llmrouterx.retry.exponential",
        fromlist=["ExponentialRetry"],
    ).ExponentialRetry
    router = LLMRouter(
        composite,
        retry=retry_cls(max_retries=3, base=0.01),
    )

    result = await router.chat("hi")
    assert result == "recovered"


@pytest.mark.asyncio
async def test_per_key_circuit_breaker_opens_after_threshold():
    class AlwaysFail(SDKClientBase):
        def __init__(self):
            super().__init__()
            self.set_handler(self._handle)

        async def _handle(self, op, kwargs):
            # 501 is a non-retryable server error: an availability failure.
            raise SDKError(501, "boom")

    adapter = OpenAICompatibleAdapter(client=AlwaysFail(), default_model="m")
    node = ClientNode("k1", adapter, failure_threshold=2, cooldown_seconds=30)
    provider = ProviderRouter("p1", [node])
    router = LLMRouter(CompositeRouter([provider]))

    with suppress(Exception):
        await router.chat("a")
    with suppress(Exception):
        await router.chat("b")

    assert node.circuit_breaker.state.name == "OPEN"
    assert await node.is_healthy() is False


@pytest.mark.asyncio
async def test_per_key_circuit_breaker_ignores_permanent_4xx():
    class Always400(SDKClientBase):
        def __init__(self):
            super().__init__()
            self.set_handler(self._handle)

        async def _handle(self, op, kwargs):
            raise SDKError(400, "bad request")

    adapter = OpenAICompatibleAdapter(client=Always400(), default_model="m")
    node = ClientNode("k1", adapter, failure_threshold=1, cooldown_seconds=30)
    provider = ProviderRouter("p1", [node])
    router = LLMRouter(CompositeRouter([provider]))

    with suppress(Exception):
        await router.chat("a")
    with suppress(Exception):
        await router.chat("b")

    assert node.circuit_breaker.state.name == "CLOSED"
    assert node.failures == 0
    assert await node.is_healthy() is True


@pytest.mark.asyncio
async def test_close_waits_for_tracked_requests():
    import time

    class Slow(StringClient):
        async def chat(self, prompt, **kwargs):
            await asyncio.sleep(0.2)
            return "slow-done"

    node = ClientNode("k1", Slow())
    provider = ProviderRouter("p1", [node])
    router = LLMRouter(CompositeRouter([provider]))

    task = asyncio.create_task(router.chat("hi"))
    await asyncio.sleep(0.05)
    assert len(router._active_tasks) == 1

    started = time.monotonic()
    await router.close()
    elapsed = time.monotonic() - started

    assert task.done()
    assert elapsed >= 0.15
    assert len(router._active_tasks) == 0
