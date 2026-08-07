import asyncio
from contextlib import suppress

from llmrouterx.client.client_node import ClientNode
from llmrouterx.providers.composite_router import CompositeRouter
from llmrouterx.providers.provider_router import ProviderRouter
from llmrouterx.router.llmrouter import LLMRouter


class SlowClient:
    def __init__(self, delay=1.0):
        self.delay = delay

    async def request(self, op, payload, api_key=None, **kwargs):
        # simulate long-running provider call
        await asyncio.sleep(self.delay)
        return {"response": "done"}

    async def chat(self, prompt, **kwargs):
        await asyncio.sleep(self.delay)
        return {"response": f"Chat response: {prompt}"}

    async def embeddings(self, text, **kwargs):
        await asyncio.sleep(self.delay)
        return {"response": f"Embeddings: {text}"}

    async def responses(self, *args, **kwargs):
        await asyncio.sleep(self.delay)
        return {"response": "Responses OK"}

    async def stream(self, prompt, **kwargs):
        await asyncio.sleep(self.delay)
        for part in prompt.split():
            yield {"response": part}


def test_cancellation_propagates_to_client():
    client = SlowClient(delay=2.0)
    node = ClientNode("k", client, max_concurrent=1)
    provider = ProviderRouter("p", [node])
    composite = CompositeRouter([provider])
    router = LLMRouter(composite)

    async def run_and_cancel():
        task = asyncio.create_task(router.chat("hello"))
        # let it start and enter client.send
        await asyncio.sleep(0.05)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

        # ensure that active_requests is decremented and failures not incremented
        assert node.active_requests == 0
        assert node.failures == 0

    asyncio.run(run_and_cancel())
