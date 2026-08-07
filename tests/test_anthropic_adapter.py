import asyncio

import pytest

from llmrouterx.adapters.anthropic import AnthropicAdapter
from llmrouterx.exceptions import ConfigurationError
from llmrouterx.retry.exponential import HTTPError


def _text_block(text):
    return type("Block", (), {"type": "text", "text": text})()


@pytest.fixture
def anthropic_adapter(anthropic_client):
    return AnthropicAdapter(client=anthropic_client, default_model="claude-3")


@pytest.mark.asyncio
async def test_anthropic_chat_returns_text(anthropic_adapter):
    result = await anthropic_adapter.chat("hello")
    assert result == "hello anthropic"


@pytest.mark.asyncio
async def test_anthropic_chat_joins_multiple_text_blocks(anthropic_client):
    async def handler(**kwargs):
        return type(
            "Response",
            (),
            {
                "content": [
                    _text_block("part one "),
                    _text_block("part two"),
                    type("Block", (), {"type": "image", "text": "ignore me"})(),
                ]
            },
        )()

    anthropic_client.set_handler(handler)
    adapter = AnthropicAdapter(client=anthropic_client, default_model="claude-3")
    result = await adapter.chat("hello")
    assert result == "part one part two"


@pytest.mark.asyncio
async def test_anthropic_chat_applies_default_max_tokens(anthropic_client):
    captured = {}

    async def handler(**kwargs):
        captured.update(kwargs)
        return type("Response", (), {"content": [_text_block("ok")]})()

    anthropic_client.set_handler(handler)
    adapter = AnthropicAdapter(client=anthropic_client, default_model="claude-3")
    await adapter.chat("hello")
    assert captured["max_tokens"] == AnthropicAdapter.DEFAULT_MAX_TOKENS
    assert captured["model"] == "claude-3"
    assert captured["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_anthropic_chat_requires_model():
    adapter = AnthropicAdapter(client=anthropic_client_factory(), default_model=None)
    with pytest.raises(ConfigurationError, match="No model specified"):
        await adapter.chat("hello")


def anthropic_client_factory():
    from tests.conftest import AnthropicClient

    return AnthropicClient()


@pytest.mark.asyncio
async def test_anthropic_stream_yields_deltas(anthropic_client):
    async def handler(**kwargs):
        async def gen():
            for text in ("hello", " ", "world"):
                await asyncio.sleep(0)
                yield type(
                    "Event",
                    (),
                    {
                        "type": "content_block_delta",
                        "delta": type("Delta", (), {"text": text})(),
                    },
                )()
            yield type("Event", (), {"type": "message_stop", "delta": None})()

        return gen()

    anthropic_client.set_handler(handler)
    adapter = AnthropicAdapter(client=anthropic_client, default_model="claude-3")
    tokens = [token async for token in adapter.stream("hi")]
    assert tokens == ["hello", " ", "world"]


@pytest.mark.asyncio
async def test_anthropic_stream_skips_non_text_events(anthropic_client):
    async def handler(**kwargs):
        async def gen():
            yield type("Event", (), {"type": "message_start", "delta": None})()
            yield type(
                "Event",
                (),
                {
                    "type": "content_block_delta",
                    "delta": type("Delta", (), {"text": "token"})(),
                },
            )()
            yield type("Event", (), {"type": "content_block_stop", "delta": None})()

        return gen()

    anthropic_client.set_handler(handler)
    adapter = AnthropicAdapter(client=anthropic_client, default_model="claude-3")
    tokens = [token async for token in adapter.stream("hi")]
    assert tokens == ["token"]


@pytest.mark.asyncio
async def test_anthropic_embeddings_not_supported(anthropic_adapter):
    with pytest.raises(NotImplementedError, match="does not provide an embeddings API"):
        await anthropic_adapter.embeddings("text")


@pytest.mark.asyncio
async def test_anthropic_health_check_ok(anthropic_adapter):
    assert await anthropic_adapter.health_check() is True


@pytest.mark.asyncio
async def test_anthropic_health_check_fails_gracefully(anthropic_client):
    async def handler(**kwargs):
        raise RuntimeError("models endpoint down")

    anthropic_client.set_handler(handler)
    adapter = AnthropicAdapter(client=anthropic_client, default_model="claude-3")
    assert await adapter.health_check() is False


@pytest.mark.asyncio
async def test_anthropic_sdk_errors_translated(anthropic_client):
    async def handler(**kwargs):
        from tests.conftest import AsyncSDKError

        raise AsyncSDKError(429, "rate limited", headers={"Retry-After": "2"})

    anthropic_client.set_handler(handler)
    adapter = AnthropicAdapter(client=anthropic_client, default_model="claude-3")
    with pytest.raises(HTTPError) as excinfo:
        await adapter.chat("hello")
    assert excinfo.value.status_code == 429
    assert excinfo.value.headers == {"Retry-After": "2"}


@pytest.mark.asyncio
async def test_anthropic_stream_error_translated(anthropic_client):
    async def handler(**kwargs):
        from tests.conftest import AsyncSDKError

        raise AsyncSDKError(503, "down")

    anthropic_client.set_handler(handler)
    adapter = AnthropicAdapter(client=anthropic_client, default_model="claude-3")
    with pytest.raises(HTTPError) as excinfo:
        async for _ in adapter.stream("hi"):
            pass
    assert excinfo.value.status_code == 503
