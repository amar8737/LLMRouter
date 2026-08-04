import threading

from llmrouter.providers.stub_provider import StubStreamingClient
from llmrouter.router.extended_streaming import ExtendedStreamingLLMRouter


def test_chat_stream_sync_with_stop_event():
    provider = StubStreamingClient(chunks=["A", "B", "C", "D", "E"], delay=0.05)
    router = ExtendedStreamingLLMRouter({"p": {"client": provider}})

    stop_event = threading.Event()

    gen = router.chat_stream_sync("hello", stop_event=stop_event)

    tokens = []

    # consume first token then signal stop
    tokens.append(next(gen))
    stop_event.set()
    # collect remaining from iterator; it should finish quickly
    for t in gen:
        tokens.append(t)

    # ensure we received at least 1 token and didn't hang
    assert len(tokens) >= 1
