import asyncio
import random

import pytest

from llmrouter.providers.provider_router import ProviderRouter
from llmrouter.client.client_node import ClientNode
from llmrouter.providers.stub_provider import StubClient
from llmrouter.scheduler.round_robin import RoundRobinScheduler
from llmrouter.scheduler.least_busy import LeastBusyScheduler
from llmrouter.scheduler.random import RandomScheduler
from llmrouter.scheduler.weighted import WeightedScheduler
from llmrouter.scheduler.priority import PriorityScheduler


class SimpleNode(ClientNode):
    def __init__(self, api_key, client, healthy=True, active=0, weight=1, priority=100):
        super().__init__(api_key, client)
        self._healthy = healthy
        self.active_requests = active
        self.weight = weight
        self.priority = priority

    async def is_healthy(self):
        return self._healthy


def test_round_robin_rotates():
    client = StubClient("a")
    nodes = [SimpleNode(f"k{i}", client) for i in range(3)]
    provider = ProviderRouter("prov", nodes, scheduler=None)
    rr = RoundRobinScheduler()

    async def run():
        picks = []
        for _ in range(6):
            n = await rr.select(provider)
            picks.append(n.api_key)
        return picks

    picks = asyncio.run(run())
    assert picks == ["k0", "k1", "k2", "k0", "k1", "k2"]


def test_least_busy_selects_least_active():
    client = StubClient("b")
    nodes = [SimpleNode("k0", client, active=5), SimpleNode("k1", client, active=1), SimpleNode("k2", client, active=3)]
    provider = ProviderRouter("prov2", nodes)
    ls = LeastBusyScheduler()

    chosen = asyncio.run(ls.select(provider))
    assert chosen.api_key == "k1"


def test_random_selects_healthy():
    client = StubClient("c")
    nodes = [SimpleNode("k0", client, healthy=False), SimpleNode("k1", client, healthy=True), SimpleNode("k2", client, healthy=True)]
    provider = ProviderRouter("prov3", nodes)
    rs = RandomScheduler()
    picks = set()
    for _ in range(20):
        n = asyncio.run(rs.select(provider))
        picks.add(n.api_key)
    assert picks <= {"k1", "k2"}
    assert len(picks) >= 1


def test_weighted_scheduler_distribution():
    client = StubClient("d")
    nodes = [SimpleNode("k0", client, weight=1), SimpleNode("k1", client, weight=9)]
    provider = ProviderRouter("prov4", nodes)
    ws = WeightedScheduler()

    counts = {"k0": 0, "k1": 0}
    # sample many times to get approximate distribution
    for _ in range(500):
        n = asyncio.run(ws.select(provider))
        counts[n.api_key] += 1

    # k1 should be chosen much more often than k0
    assert counts["k1"] > counts["k0"] * 3


def test_priority_scheduler_prefers_high_priority():
    client = StubClient("e")
    nodes = [SimpleNode("low", client, priority=50), SimpleNode("high", client, priority=1)]
    provider = ProviderRouter("prov5", nodes)
    ps = PriorityScheduler()
    chosen = asyncio.run(ps.select(provider))
    assert chosen.api_key == "high"
