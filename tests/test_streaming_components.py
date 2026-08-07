import asyncio

import pytest

from llmrouterx.streaming.tokenizer import (
    TokenizerManager,
    character_tokenizer,
    whitespace_tokenizer,
)


def test_whitespace_tokenizer_splits_on_space():
    assert whitespace_tokenizer("Hello world!") == ["Hello", "world!"]


def test_character_tokenizer_splits_characters():
    assert character_tokenizer("ab c") == ["a", "b", " ", "c"]


def test_tokenizer_manager_defaults():
    tm = TokenizerManager()
    assert tm.tokenize("a b c") == ["a", "b", "c"]
    assert tm.tokenize("abc", tokenizer="character") == ["a", "b", "c"]


def test_tokenizer_manager_register():
    tm = TokenizerManager()
    tm.register("comma", lambda text: text.split(","))
    assert tm.tokenize("x,y,z", tokenizer="comma") == ["x", "y", "z"]


def test_tokenizer_manager_register_duplicate_raises():
    tm = TokenizerManager()
    with pytest.raises(ValueError, match="already exists"):
        tm.register("whitespace", lambda t: t)


def test_tokenizer_manager_unregister():
    tm = TokenizerManager()
    tm.register("custom", lambda t: [t])
    tm.unregister("custom")
    with pytest.raises(ValueError, match="Unknown tokenizer"):
        tm.get("custom")


def test_tokenizer_manager_unknown_raises():
    tm = TokenizerManager()
    with pytest.raises(ValueError, match="Unknown tokenizer"):
        tm.get("missing")


class ChunkingAdapter:
    """Adapter whose stream yields a fixed sequence of tokens."""

    def __init__(self, tokens):
        self.tokens = tokens

    async def stream(self, prompt, **kwargs):
        for token in self.tokens:
            await asyncio.sleep(0)
            yield token


@pytest.mark.asyncio
async def test_async_stream_engine_yields_all_tokens():
    from llmrouterx.streaming.async_stream import AsyncStreamEngine

    engine = AsyncStreamEngine(ChunkingAdapter(["a", "b", "c"]))
    collected = [token async for token in engine.stream("hi")]
    assert collected == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_async_stream_engine_invokes_on_chunk():
    from llmrouterx.streaming.async_stream import AsyncStreamEngine

    seen = []
    engine = AsyncStreamEngine(ChunkingAdapter(["a", "b", "c"]))
    async for _ in engine.stream("hi", on_chunk=seen.append):
        pass
    assert seen == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_async_stream_engine_stop_condition_breaks_early():
    from llmrouterx.streaming.async_stream import AsyncStreamEngine

    seen = []
    condition = {"stop": False}

    def stop():
        if len(seen) >= 2:
            condition["stop"] = True
        return condition["stop"]

    engine = AsyncStreamEngine(ChunkingAdapter(["a", "b", "c", "d", "e"]))
    async for token in engine.stream("hi", stop_condition=stop):
        seen.append(token)
    assert seen == ["a", "b"]


@pytest.mark.asyncio
async def test_async_stream_engine_stream_to_text():
    from llmrouterx.streaming.async_stream import AsyncStreamEngine

    engine = AsyncStreamEngine(ChunkingAdapter(["Hello", " ", "world"]))
    assert await engine.stream_to_text("hi") == "Hello world"


@pytest.mark.asyncio
async def test_streaming_manager_async_and_sync():
    from llmrouterx.streaming.manager import StreamingManager

    manager = StreamingManager(ChunkingAdapter(["x", "y", "z"]))
    assert await manager.stream_to_text("hi") == "xyz"

    sync_tokens = list(manager.stream_sync("hi"))
    assert sync_tokens == ["x", "y", "z"]
    assert manager.stream_sync_to_text("hi") == "xyz"


@pytest.mark.asyncio
async def test_streaming_manager_tokenize():
    from llmrouterx.streaming.manager import StreamingManager

    manager = StreamingManager(ChunkingAdapter(["a"]))
    assert manager.tokenize("one two three") == ["one", "two", "three"]
    assert manager.tokenize("ab", tokenizer="character") == ["a", "b"]


@pytest.mark.asyncio
async def test_streaming_manager_register_tokenizer():
    from llmrouterx.streaming.manager import StreamingManager

    manager = StreamingManager(ChunkingAdapter(["a"]))
    manager.register_tokenizer("piped", lambda text: text.split("|"))
    assert manager.tokenize("p|q|r", tokenizer="piped") == ["p", "q", "r"]
