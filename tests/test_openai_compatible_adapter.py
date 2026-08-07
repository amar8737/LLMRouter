import asyncio

import pytest

from llmrouterx.adapters.openai_compatible import OpenAICompatibleAdapter
from llmrouterx.retry.exponential import HTTPError


class ConnFailure(Exception):
    """Module-level stand-in for a provider-SDK connection error."""


class TimeoutFailure(Exception):
    """Module-level stand-in for a provider-SDK timeout error."""


@pytest.fixture
def adapter(openai_client):
    return OpenAICompatibleAdapter(client=openai_client, default_model="gpt-4")


@pytest.mark.asyncio
async def test_chat_returns_content(adapter, openai_client):
    async def handler(op, kwargs):
        from tests.conftest import _chat_response

        return _chat_response("hello world")

    openai_client.set_handler(handler)
    result = await adapter.chat("hi")
    assert result == "hello world"


@pytest.mark.asyncio
async def test_chat_uses_default_model(adapter, openai_client):
    captured = {}

    async def handler(op, kwargs):
        captured.update(kwargs)
        from tests.conftest import _chat_response

        return _chat_response("ok")

    openai_client.set_handler(handler)
    await adapter.chat("hi")
    assert captured["model"] == "gpt-4"
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["stream"] is False


@pytest.mark.asyncio
async def test_chat_per_call_model_override(adapter, openai_client):
    captured = {}

    async def handler(op, kwargs):
        captured.update(kwargs)
        from tests.conftest import _chat_response

        return _chat_response("ok")

    openai_client.set_handler(handler)
    await adapter.chat("hi", model="gpt-4o")
    assert captured["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_embeddings_returns_vector(adapter, openai_client):
    result = await adapter.embeddings("text")
    assert result == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embeddings_uses_embedding_model(openai_client):
    captured = {}

    async def handler(op, kwargs):
        captured.update(kwargs)
        from tests.conftest import _embedding_response

        return _embedding_response([1.0])

    openai_client.set_handler(handler)
    adapter = OpenAICompatibleAdapter(
        client=openai_client,
        default_model="gpt-4",
        embedding_model="text-embedding-3-small",
    )
    await adapter.embeddings("text")
    assert captured["model"] == "text-embedding-3-small"
    assert captured["input"] == "text"


@pytest.mark.asyncio
async def test_stream_yields_content_chunks(adapter, openai_client):
    async def handler(op, kwargs):
        async def gen():
            for text in ("Hello", " ", "world"):
                await asyncio.sleep(0)
                yield type(
                    "Chunk",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {
                                    "delta": type("Delta", (), {"content": text})(),
                                },
                            )()
                        ]
                    },
                )()

        return gen()

    openai_client.set_handler(handler)
    tokens = [token async for token in adapter.stream("hi")]
    assert tokens == ["Hello", " ", "world"]


@pytest.mark.asyncio
async def test_stream_skips_empty_deltas(adapter, openai_client):
    async def handler(op, kwargs):
        async def gen():
            yield type(
                "Chunk",
                (),
                {"choices": [type("Choice", (), {"delta": type("Delta", (), {"content": ""})})()]},
            )()
            yield type("Chunk", (), {"choices": []})()
            yield type(
                "Chunk",
                (),
                {
                    "choices": [
                        type("Choice", (), {"delta": type("Delta", (), {"content": "x"})}),
                    ]
                },
            )()

        return gen()

    openai_client.set_handler(handler)
    tokens = [token async for token in adapter.stream("hi")]
    assert tokens == ["x"]


@pytest.mark.asyncio
async def test_responses_api_used_when_available(adapter, openai_client):
    captured = {}

    async def handler(op, kwargs):
        captured.update(kwargs)
        from tests.conftest import _chat_response

        return _chat_response("responses-ok")

    openai_client.set_handler(handler)
    result = await adapter.responses("arg1", model="gpt-5")
    assert captured["model"] == "gpt-5"
    assert result == "responses-ok"


@pytest.mark.asyncio
async def test_responses_raises_when_client_lacks_api():
    client = type("Client", (), {"chat": object()})()
    adapter = OpenAICompatibleAdapter(client=client, default_model="gpt-4")
    with pytest.raises(NotImplementedError, match="does not implement the Responses API"):
        await adapter.responses()


@pytest.mark.asyncio
async def test_health_check_true(adapter):
    assert await adapter.health_check() is True


@pytest.mark.asyncio
async def test_health_check_false_on_error(openai_client):
    async def handler(op, kwargs):
        raise RuntimeError("api down")

    openai_client.set_handler(handler)
    adapter = OpenAICompatibleAdapter(client=openai_client, default_model="gpt-4")
    assert await adapter.health_check() is False


@pytest.mark.asyncio
async def test_sdk_error_translated_on_chat(adapter, openai_client):
    async def handler(op, kwargs):
        from tests.conftest import AsyncSDKError

        raise AsyncSDKError(500, "boom")

    openai_client.set_handler(handler)
    with pytest.raises(HTTPError) as excinfo:
        await adapter.chat("hi")
    assert excinfo.value.status_code == 500


@pytest.mark.asyncio
async def test_connection_error_mapped_by_registered_type(adapter, openai_client, monkeypatch):
    path = f"{ConnFailure.__module__}.ConnFailure"
    monkeypatch.setattr("llmrouterx.adapters.base._SDK_ERROR_PATHS", {503: (path,)})
    monkeypatch.setattr("llmrouterx.adapters.base._RESOLVED_SDK_TYPES", {})

    async def handler(op, kwargs):
        raise ConnFailure("connection failed")

    openai_client.set_handler(handler)
    with pytest.raises(HTTPError) as excinfo:
        await adapter.chat("hi")
    assert excinfo.value.status_code == 503


@pytest.mark.asyncio
async def test_timeout_mapped_by_registered_type(adapter, openai_client, monkeypatch):
    path = f"{TimeoutFailure.__module__}.TimeoutFailure"
    monkeypatch.setattr("llmrouterx.adapters.base._SDK_ERROR_PATHS", {504: (path,)})
    monkeypatch.setattr("llmrouterx.adapters.base._RESOLVED_SDK_TYPES", {})

    async def handler(op, kwargs):
        raise TimeoutFailure("timed out")

    openai_client.set_handler(handler)
    with pytest.raises(HTTPError) as excinfo:
        await adapter.chat("hi")
    assert excinfo.value.status_code == 504


@pytest.mark.asyncio
async def test_unregistered_sdk_error_passes_through(adapter, openai_client, monkeypatch):
    # A raised exception that is NOT a registered SDK type must not be
    # translated to HTTPError, even if its name looks connection-like.
    monkeypatch.setattr("llmrouterx.adapters.base._SDK_ERROR_PATHS", {})

    class APIConnectionError(Exception):
        pass

    async def handler(op, kwargs):
        raise APIConnectionError("not a known SDK type")

    openai_client.set_handler(handler)
    with pytest.raises(APIConnectionError):
        await adapter.chat("hi")


def test_provider_name_property():
    from llmrouterx.adapters.openai import OpenAIAdapter

    adapter = OpenAIAdapter(client=object())
    assert adapter.provider_name == "OpenAI"


def test_repr_contains_model():
    adapter = OpenAICompatibleAdapter(client=object(), default_model="gpt-4")
    assert "gpt-4" in repr(adapter)
    assert "OpenAICompatibleAdapter" in repr(adapter)
