import asyncio

from llmrouterx.client.client_node import ClientNode
from llmrouterx.providers.provider_router import ProviderRouter
from llmrouterx.providers.composite_router import CompositeRouter
from llmrouterx.router.llmrouter import LLMRouter


class SlowClient:
    def __init__(self, delay=1.0):
        self.delay = delay

    async def request(self, op, payload, api_key=None, **kwargs):
        # simulate long-running provider call
        await asyncio.sleep(self.delay)
        return {"response": "done"}


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
        try:
            await task
        except asyncio.CancelledError:
            pass

        # ensure that active_requests is decremented and failures not incremented
        assert node.active_requests == 0
        assert node.failures == 0

    asyncio.run(run_and_cancel())
