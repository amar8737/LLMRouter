"""Tests for the synchronous ``LLMRouterSync`` facade."""

import pytest

from llmrouterx import LLMRouter, LLMRouterSync
from llmrouterx.adapters.base import BaseProviderAdapter
from llmrouterx.client import ClientNode
from llmrouterx.providers import CompositeRouter, ProviderRouter


class _TextAdapter(BaseProviderAdapter):
    """Minimal adapter that returns/yields strings, with a rerank method."""

    async def chat(self, prompt, **kwargs):
        return f"chat:{prompt}"

    async def stream(self, prompt, **kwargs):
        for part in prompt.split():
            yield part

    async def embeddings(self, text, **kwargs):
        return [0.1, 0.2, 0.3]

    async def rerank(self, query, documents, **kwargs):
        return [{"index": 0, "relevance_score": 0.9}]


def _build_sync_router():
    adapter = _TextAdapter(client=object())
    node = ClientNode("key", adapter)
    provider = ProviderRouter("test", [node])
    return LLMRouterSync(LLMRouter(CompositeRouter([provider])))


def test_chat_blocking():
    router = _build_sync_router()
    try:
        assert router.chat("Hello") == "chat:Hello"
    finally:
        router.close()


def test_embeddings_blocking():
    router = _build_sync_router()
    try:
        assert router.embeddings("text") == [0.1, 0.2, 0.3]
    finally:
        router.close()


def test_stream_returns_joined_text():
    router = _build_sync_router()
    try:
        assert router.stream("a b c") == "abc"
    finally:
        router.close()


def test_stream_chunks_yields():
    router = _build_sync_router()
    try:
        assert list(router.stream_chunks("a b c")) == ["a", "b", "c"]
    finally:
        router.close()


def test_rerank_blocking():
    router = _build_sync_router()
    try:
        results = router.rerank("query", ["doc1", "doc2"])
        assert results == [{"index": 0, "relevance_score": 0.9}]
    finally:
        router.close()


def test_health_blocking():
    router = _build_sync_router()
    try:
        assert router.health() == {"test": True}
    finally:
        router.close()


def test_metrics_visible():
    router = _build_sync_router()
    try:
        router.chat("hi")
        assert router.get_metrics()["counters"]["requests.chat"] == 1
    finally:
        router.close()


def test_close_is_idempotent():
    router = _build_sync_router()
    router.close()
    router.close()


def test_context_manager():
    with _build_sync_router() as router:
        assert router.chat("hi") == "chat:hi"


def test_from_cascade_with_env_names(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-a")
    monkeypatch.setenv("TEST_API_KEY_1", "sk-b")

    router = LLMRouterSync.from_cascade(
        ["openai:TEST_API_KEY"],
        client_factory=lambda provider, key: _TextAdapter(client=object()),
    )
    try:
        assert len(router.providers) == 1
        assert len(router.providers[0].clients) == 2
        assert router.chat("hi") == "chat:hi"
    finally:
        router.close()


def test_from_providers_builds_router(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-a")

    router = LLMRouterSync.from_providers(
        [{"provider": "openai", "key_env": "OPENAI_API_KEY", "model": "gpt-4o"}],
        client_factory=lambda provider, key: _TextAdapter(client=object()),
    )
    try:
        assert router.providers[0].name == "openai"
        assert router.providers[0].clients[0].client.default_model == "gpt-4o"
    finally:
        router.close()


def test_from_providers_expands_scanned_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-a")
    monkeypatch.setenv("GROQ_API_KEY_1", "gsk-b")
    monkeypatch.setenv("GROQ_API_KEY_2", "gsk-c")

    router = LLMRouterSync.from_providers(
        [{"provider": "groq", "key_env": "GROQ_API_KEY"}],
        client_factory=lambda provider, key: _TextAdapter(client=object()),
    )
    try:
        assert len(router.providers[0].clients) == 3
    finally:
        router.close()


def test_init_requires_router():
    with pytest.raises(ValueError, match="needs an LLMRouter"):
        LLMRouterSync()
