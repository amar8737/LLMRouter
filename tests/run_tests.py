import asyncio

from llmrouter.router.llmrouter import LLMRouter
from llmrouter.providers.composite_router import CompositeRouter
from llmrouter.providers.provider_router import ProviderRouter
from llmrouter.client.client_node import ClientNode
from llmrouter.providers.stub_provider import StubClient
from llmrouter.scheduler.round_robin import RoundRobinScheduler
from llmrouter.retry.exponential import ExponentialRetry
from llmrouter.metrics.metrics import MetricsCollector



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
