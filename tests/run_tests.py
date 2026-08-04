import asyncio

from llmrouter.client.client_node import ClientNode
from llmrouter.metrics.metrics import MetricsCollector
from llmrouter.providers.composite_router import CompositeRouter
from llmrouter.providers.provider_router import ProviderRouter
from llmrouter.providers.stub_provider import StubClient
from llmrouter.retry.exponential import ExponentialRetry
from llmrouter.router.llmrouter import LLMRouter
from llmrouter.scheduler.round_robin import RoundRobinScheduler


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
