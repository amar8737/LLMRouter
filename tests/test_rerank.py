"""Tests for the rerank endpoint across adapters, client nodes, and the router."""

import asyncio

import pytest

from llmrouterx import LLMRouter
from llmrouterx.adapters import AdapterFactory
from llmrouterx.adapters.base import BaseProviderAdapter
from llmrouterx.client import ClientNode
from llmrouterx.providers import CompositeRouter, ProviderRouter


def test_openai_compatible_rerank_normalizes_results(openai_client):
    adapter = AdapterFactory.create(provider="openai", client=openai_client)

    results = asyncio.run(adapter.rerank("query", ["doc1", "doc2"], top_n=1))

    assert results == [
        {"index": 1, "relevance_score": 0.91},
        {"index": 0, "relevance_score": 0.42},
    ]
    assert openai_client.calls[-1] == (
        "rerank",
        {"model": None, "query": "query", "documents": ["doc1", "doc2"], "top_n": 1},
    )


def test_openai_compatible_rerank_default_model(openai_client):
    adapter = AdapterFactory.create(
        provider="openai", client=openai_client, default_model="rerank-v3"
    )

    asyncio.run(adapter.rerank("q", ["d"]))

    assert openai_client.calls[-1][1]["model"] == "rerank-v3"


def test_rerank_unsupported_provider_raises(anthropic_client):
    adapter = AdapterFactory.create(provider="anthropic", client=anthropic_client)
    with pytest.raises(NotImplementedError, match="rerank"):
        asyncio.run(adapter.rerank("q", ["d"]))


def test_base_normalize_rerank_sorts_and_filters():
    results = [
        type("R", (), {"index": 1, "relevance_score": 0.5})(),
        type("R", (), {"index": 2, "relevance_score": 0.9})(),
        type("R", (), {"index": 3})(),
    ]
    normalized = BaseProviderAdapter._normalize_rerank(results)
    assert normalized == [
        {"index": 2, "relevance_score": 0.9},
        {"index": 1, "relevance_score": 0.5},
    ]


class _RerankStub(BaseProviderAdapter):
    async def chat(self, prompt, **kwargs):
        return "chat"

    async def stream(self, prompt, **kwargs):
        yield "tok"

    async def embeddings(self, text, **kwargs):
        return [0.0]

    async def rerank(self, query, documents, **kwargs):
        return [{"index": 1, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.2}]


def test_router_rerank_dispatches():
    adapter = _RerankStub(client=object())
    node = ClientNode("key", adapter)
    provider = ProviderRouter("test", [node])
    router = LLMRouter(CompositeRouter([provider]))

    results = asyncio.run(router.rerank("query", ["a", "b"]))

    assert results == [{"index": 1, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.2}]
    snapshot = router.metrics.get()
    assert snapshot["counters"]["requests.rerank"] == 1


def test_router_rerank_requires_fields():
    adapter = _RerankStub(client=object())
    node = ClientNode("key", adapter)
    router = LLMRouter(CompositeRouter([ProviderRouter("test", [node])]))

    with pytest.raises(ValueError, match="documents"):
        asyncio.run(router.rerank("query", []))


def test_router_rerank_failover_across_providers():
    class _Broken(BaseProviderAdapter):
        async def chat(self, prompt, **kwargs):
            return "chat"

        async def stream(self, prompt, **kwargs):
            yield "tok"

        async def embeddings(self, text, **kwargs):
            return [0.0]

        async def rerank(self, query, documents, **kwargs):
            raise ConnectionError("down")

    broken = ProviderRouter("broken", [ClientNode("k1", _Broken(client=object()))])
    good = ProviderRouter("good", [ClientNode("k2", _RerankStub(client=object()))])
    router = LLMRouter(CompositeRouter([broken, good]))

    results = asyncio.run(router.rerank("q", ["x"]))

    assert results == [{"index": 1, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.2}]
    assert router.providers[-1].last_api_key == "k2"


def test_client_node_rejects_unknown_op():
    node = ClientNode("key", _RerankStub(client=object()))
    with pytest.raises(ValueError, match="Unsupported operation"):
        asyncio.run(node.send("frobnicate", {"prompt": "x"}))
