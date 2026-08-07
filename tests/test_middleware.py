import asyncio

import pytest

from llmrouterx.client.client_node import ClientNode
from llmrouterx.middleware.base import BaseMiddleware, MiddlewareResult
from llmrouterx.providers.composite_router import CompositeRouter
from llmrouterx.providers.provider_router import ProviderRouter
from llmrouterx.providers.stub_provider import StubClient
from llmrouterx.router.llmrouter import LLMRouter


def make_router(*middleware):
    node = ClientNode("k1", StubClient("s1"))
    provider = ProviderRouter("p1", [node])
    composite = CompositeRouter([provider])
    return LLMRouter(composite, middleware=list(middleware))


def test_middleware_result_repr():
    result = MiddlewareResult(response="x", stop=True)
    assert "stop=True" in repr(result)
    assert "response=set" in repr(result)


def test_middleware_result_plain():
    result = MiddlewareResult()
    assert result.payload is None
    assert result.response is None
    assert result.stop is False


class UppercaseAfter(BaseMiddleware):
    async def after_response(self, operation, payload, response, context):
        response["response"] = response["response"].upper()
        return response


def test_after_response_rewrites_response():
    router = make_router(UppercaseAfter())
    result = asyncio.run(router.chat("hi"))
    assert result["response"] == "CHAT FROM S1: HI"


class CollectContext(BaseMiddleware):
    def __init__(self):
        self.context = None

    async def before_request(self, operation, payload, context):
        self.context = context
        return None


def test_middleware_receives_request_context():
    mw = CollectContext()
    router = make_router(mw)
    asyncio.run(router.chat("hi"))
    assert mw.context is not None
    assert mw.context.operation == "chat"
    assert mw.context.prompt == "hi"
    assert mw.context.request_id


class ReplacePayload(BaseMiddleware):
    async def before_request(self, operation, payload, context):
        payload["prompt"] = "rewritten"
        return payload


def test_before_request_can_rewrite_payload():
    router = make_router(ReplacePayload())
    result = asyncio.run(router.chat("original"))
    assert result["response"] == "Chat from s1: rewritten"


class ShortCircuitMiddleware(BaseMiddleware):
    async def before_request(self, operation, payload, context):
        return MiddlewareResult(response={"response": "short-circuited"}, stop=True)


def test_before_request_short_circuits():
    router = make_router(ShortCircuitMiddleware())
    result = asyncio.run(router.chat("hi"))
    assert result["response"] == "short-circuited"


class BadOnException(BaseMiddleware):
    def __init__(self):
        self.called = False

    async def on_exception(self, operation, payload, exception, context):
        self.called = True
        raise RuntimeError("observer must not mask original error")


def test_on_exception_observer_error_does_not_mask_original():
    mw = BadOnException()
    node = ClientNode("k1", _FailClient())
    provider = ProviderRouter("p1", [node])
    composite = CompositeRouter([provider])
    router = LLMRouter(composite, middleware=[mw])

    with pytest.raises(RuntimeError, match="original failure"):
        asyncio.run(router.chat("hi"))
    assert mw.called is True


class _FailClient:
    async def chat(self, prompt, **kwargs):
        raise RuntimeError("original failure")


class ObservingOnException(BaseMiddleware):
    def __init__(self):
        self.errors = []

    async def on_exception(self, operation, payload, exception, context):
        self.errors.append((operation, str(exception)))
        return None


def test_on_exception_is_notified():
    mw = ObservingOnException()
    node = ClientNode("k1", _FailClient())
    provider = ProviderRouter("p1", [node])
    composite = CompositeRouter([provider])
    router = LLMRouter(composite, middleware=[mw])

    with pytest.raises(RuntimeError):
        asyncio.run(router.chat("hi"))
    assert mw.errors == [("chat", "original failure")]


class MiddlewareResultPayload(BaseMiddleware):
    async def before_request(self, operation, payload, context):
        return MiddlewareResult(payload={**payload, "prompt": "via-result"})


def test_middleware_result_payload_rewrite():
    router = make_router(MiddlewareResultPayload())
    result = asyncio.run(router.chat("original"))
    assert result["response"] == "Chat from s1: via-result"
