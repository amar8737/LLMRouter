from llmrouterx.client.client_node import ClientNode
from llmrouterx.providers.composite_router import CompositeRouter
from llmrouterx.providers.provider_router import ProviderRouter
from llmrouterx.providers.stub_provider import StubClient
from llmrouterx.router.streaming import StreamingLLMRouter


def test_async_streaming_and_callback():
    stub = StubClient("s1")
    node = ClientNode("k1", stub)
    provider = ProviderRouter("p1", [node])
    composite = CompositeRouter([provider])
    router = StreamingLLMRouter(composite)

    received = []

    async def run():
        async for token in router.stream("hello world"):
            received.append(token)

    import asyncio

    asyncio.run(run())

    assert len(received) > 0


def test_streaming_callback_on_chunk_and_sync():
    stub = StubClient("s2")
    node = ClientNode("k2", stub)
    provider = ProviderRouter("p2", [node])
    composite = CompositeRouter([provider])
    router = StreamingLLMRouter(composite)

    cb_received = []

    def on_chunk(t):
        cb_received.append(t)

    # sync streaming (runs async path under the hood)
    out = ""
    for tok in router.stream_sync("stream me", on_chunk=on_chunk):
        out += tok

    assert len(cb_received) > 0
    assert len(out) > 0
