import asyncio

import pytest
from fastapi.testclient import TestClient

from llmrouterx.adapters.base import BaseProviderAdapter
from llmrouterx.client import ClientNode
from llmrouterx.exceptions import NoHealthyClientError
from llmrouterx.metrics.metrics import MetricsCollector
from llmrouterx.providers.composite_router import CompositeRouter
from llmrouterx.providers.provider_router import ProviderRouter
from llmrouterx.retry.exponential import HTTPError
from llmrouterx.router.llmrouter import LLMRouter
from llmrouterx.server.app import create_app


class FailingClient:
    def __init__(self, exc):
        self.exc = exc

    async def chat(self, prompt, **kwargs):
        raise self.exc

    async def embeddings(self, text, **kwargs):
        raise self.exc

    async def responses(self, *args, **kwargs):
        raise self.exc

    async def stream(self, prompt, **kwargs):
        raise self.exc
        yield  # pragma: no cover


@pytest.fixture
def client():
    router = LLMRouter(
        CompositeRouter(
            [ProviderRouter("ok", [ClientNode("k", HealthyStub())])]
        )
    )
    app = create_app(router=router)
    with TestClient(app) as test_client:
        yield test_client


class EchoAdapter(BaseProviderAdapter):
    async def chat(self, prompt, **kwargs):
        return {"provider": "echo", "response": f"hi {prompt}"}

    async def embeddings(self, text, **kwargs):
        return [0.1]

    async def stream(self, prompt, **kwargs):
        for part in prompt.split():
            yield {"provider": "echo", "response": part}


class HealthyStub:
    async def chat(self, prompt, **kwargs):
        return {"provider": "ok", "response": f"hi {prompt}"}

    async def embeddings(self, text, **kwargs):
        return [0.1, 0.2]

    async def responses(self, *args, **kwargs):
        return {"response": "ok"}

    async def stream(self, prompt, **kwargs):
        for part in prompt.split():
            yield {"provider": "ok", "response": part}


def _router_with(clients_by_provider, metrics=None):
    providers = [
        ProviderRouter(name, [ClientNode(f"{name}-k", client)])
        for name, client in clients_by_provider.items()
    ]
    return LLMRouter(CompositeRouter(providers, metrics=metrics), metrics=metrics)


def test_no_healthy_error_formats_failure_sequence():
    exc = NoHealthyClientError(
        "All providers failed.",
        errors=[HTTPError(429, "rate limit"), ValueError("bad input")],
    )
    text = str(exc)
    assert "All providers failed" in text
    assert "HTTPError: 429: rate limit" in text
    assert "ValueError: bad input" in text


def test_composite_accumulates_all_provider_errors():
    composite = CompositeRouter(
        [
            ProviderRouter("a", [ClientNode("k1", FailingClient(ValueError("a")))]),
            ProviderRouter("b", [ClientNode("k2", FailingClient(RuntimeError("b")))]),
        ]
    )
    with pytest.raises(NoHealthyClientError) as excinfo:
        asyncio.run(composite.handle("chat", {"prompt": "hi"}))
    assert [type(e).__name__ for e in excinfo.value.errors] == ["ValueError", "RuntimeError"]


def test_non_retryable_error_propagates_with_sequence():
    router = _router_with({"a": FailingClient(RuntimeError("boom"))})
    with pytest.raises(RuntimeError, match="boom") as excinfo:
        asyncio.run(router.chat("hi"))
    assert [type(e).__name__ for e in excinfo.value.errors] == ["RuntimeError"]


def test_transient_error_still_retries_and_succeeds():
    class FlakyThenOk:
        def __init__(self):
            self.calls = 0

        async def chat(self, prompt, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise HTTPError(429, "rate limited", headers={"Retry-After": "0.01"})
            return {"provider": "f", "response": "finally"}

    router = LLMRouter(
        CompositeRouter([ProviderRouter("f", [ClientNode("k", FlakyThenOk())])])
    )
    result = asyncio.run(router.chat("hi"))
    assert result["response"] == "finally"


def test_track_tokens_global_and_labeled():
    metrics = MetricsCollector()
    metrics.track_tokens("openai", prompt_tokens=10, completion_tokens=5)
    metrics.track_tokens("openai", prompt_tokens=3, completion_tokens=2)
    metrics.track_tokens("groq", prompt_tokens=100)

    data = metrics.snapshot()
    assert data["counters"]["tokens.prompt.total"] == 113
    assert data["counters"]["tokens.completion.total"] == 7
    labeled = data["labeled_counters"]["tokens.total"]
    assert labeled["provider=openai"] == 20
    assert labeled["provider=groq"] == 100


def test_chat_records_usage_as_tokens():
    """Adapters extract response usage and the router records it."""
    from tests.conftest import OpenAICompatibleClient, _chat_response

    class _Usage:
        prompt_tokens = 42
        completion_tokens = 8

    client = OpenAICompatibleClient()

    async def handler(op, kwargs):
        resp = _chat_response("usage-reply")
        resp.usage = _Usage()
        return resp

    client.set_handler(handler)

    from llmrouterx.adapters.openai_compatible import OpenAICompatibleAdapter

    adapter = OpenAICompatibleAdapter(client=client, default_model="gpt-4")
    provider = ProviderRouter("openai", [ClientNode(api_key="k", client=adapter)])
    router = LLMRouter(CompositeRouter([provider]))

    result = asyncio.run(router.chat("hi"))

    assert result == "usage-reply"
    counters = router.metrics.snapshot()["counters"]
    assert counters["tokens.prompt.total"] == 42
    assert counters["tokens.completion.total"] == 8
    labeled = router.metrics.snapshot()["labeled_counters"]["tokens.total"]
    assert labeled["provider=openai"] == 50


def test_from_cascade_builds_fallback_chain(monkeypatch):
    built = []

    def client_factory(provider, api_key):
        built.append((provider, api_key))
        return EchoAdapter(client=object())  # BaseProviderAdapter passes through

    router = LLMRouter.from_cascade(
        ["openai:sk-a", "groq:gsk-b"],
        client_factory=client_factory,
    )
    assert built == [("openai", "sk-a"), ("groq", "gsk-b")]
    assert [p.name for p in router.providers] == ["openai", "groq"]
    result = asyncio.run(router.chat("hello"))
    assert result["response"] == "hi hello"


def test_from_cascade_rejects_bad_format():
    with pytest.raises(ValueError, match="provider:api_key"):
        LLMRouter.from_cascade(["not-a-valid-item"])


def test_dashboard_returns_html(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "LLMRouter Operations" in response.text
    assert "/health" in response.text
    assert "/metrics" in response.text
