from llmrouterx.providers.stub_provider import StubClient
from llmrouterx.router.extended_streaming import ExtendedStreamingLLMRouter


def test_chat_sync_and_stream_sync_complete():
    stub = StubClient("stub")
    router = ExtendedStreamingLLMRouter({"stub": {"client": stub}})

    # sync full response
    resp = router.chat_sync("hello sync")
    assert "Echo from stub" in resp

    # sync stream complete
    out = router.chat_stream_sync_complete("stream me", print_output=False)
    assert len(out) > 0
