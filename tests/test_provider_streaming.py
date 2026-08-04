import asyncio

from llmrouter.providers.stub_provider import StubStreamingClient
from llmrouter.router.extended_streaming import ExtendedStreamingLLMRouter


def test_stub_streaming_async_and_sync():
    client = StubStreamingClient(name="stest", chunks=["A", "B", "C"], delay=0.001)
    router = ExtendedStreamingLLMRouter({"stest": {"client": client}})

    # async stream
    tokens = []

    async def run_async():
        async for t in router.chat_stream("hi"):
            tokens.append(t)

    asyncio.run(run_async())
    assert len(tokens) == 3
    assert "".join(tokens) == "ABC"

    # sync stream complete
    full = router.chat_stream_sync_complete("hi")
    assert full == "ABC"
