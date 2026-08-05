import asyncio

from llmrouterx.client.client_node import ClientNode
from llmrouterx.metrics.metrics import MetricsCollector
from llmrouterx.providers.composite_router import CompositeRouter
from llmrouterx.providers.provider_router import ProviderRouter
from llmrouterx.providers.stub_provider import StubClient
from llmrouterx.retry.exponential import ExponentialRetry
from llmrouterx.router.llmrouter import LLMRouter
from llmrouterx.scheduler.round_robin import RoundRobinScheduler


async def main():
    client = StubClient("stub1")
    node = ClientNode("key1", client)
    scheduler = RoundRobinScheduler()
    provider = ProviderRouter("stub_provider", [node], scheduler=scheduler)
    composite = CompositeRouter([provider])
    retry = ExponentialRetry(max_retries=2)
    metrics = MetricsCollector()
    router = LLMRouter(composite, retry=retry, metrics=metrics)

    resp = await router.chat("hello world")
    print("Response:", resp)
    print("Metrics:", metrics.get())


if __name__ == "__main__":
    asyncio.run(main())
