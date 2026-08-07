import asyncio
import time

import pytest

from llmrouterx.client.client_node import ClientNode, _utcnow
from llmrouterx.exceptions import NoHealthyClientError
from llmrouterx.metrics.metrics import MetricsCollector
from llmrouterx.providers.composite_router import CompositeRouter
from llmrouterx.providers.provider_router import ProviderRouter
from llmrouterx.providers.stub_provider import StubClient
from llmrouterx.scheduler.base import BaseScheduler
from llmrouterx.scheduler.round_robin import RoundRobinScheduler


def healthy_node(key, name="stub"):
    """Build a ClientNode backed by a StubClient that is healthy."""
    return ClientNode(key, StubClient(name))


def unhealthy_node(key, name="stub"):
    """Build a ClientNode whose circuit breaker is open (unhealthy)."""
    node = ClientNode(key, StubClient(name), failure_threshold=1, cooldown_seconds=60)
    node.circuit_breaker.record_failure()
    return node


@pytest.fixture
def metrics():
    return MetricsCollector()


def test_composite_health_maps_provider_names(metrics):
    p1 = ProviderRouter("alpha", [healthy_node("k1")])
    p2 = ProviderRouter("beta", [unhealthy_node("k2")])
    composite = CompositeRouter([p1, p2], metrics=metrics)
    assert asyncio.run(composite.health()) == {"alpha": True, "beta": False}


def test_composite_health_swallows_check_errors(metrics):
    class Broken:
        name = "broken"

        async def is_healthy(self):
            raise RuntimeError("health boom")

    composite = CompositeRouter([Broken()], metrics=metrics)
    assert asyncio.run(composite.health()) == {"broken": False}


def test_composite_handle_fails_over_to_healthy_provider(metrics):
    bad = ProviderRouter("bad", [unhealthy_node("k1", "b")])
    good = ProviderRouter("good", [healthy_node("k2", "g")])
    composite = CompositeRouter([bad, good], metrics=metrics)

    result = asyncio.run(composite.handle("chat", {"prompt": "hi"}))
    assert result["provider"] == "g"


def test_composite_handle_raises_when_all_providers_unhealthy(metrics):
    bad = ProviderRouter("bad", [unhealthy_node("k1", "b")])
    composite = CompositeRouter([bad], metrics=metrics)
    with pytest.raises(NoHealthyClientError):
        asyncio.run(composite.handle("chat", {"prompt": "hi"}))


def test_composite_handle_records_metrics(metrics):
    good = ProviderRouter("good", [healthy_node("k1", "g")])
    composite = CompositeRouter([good], metrics=metrics)
    asyncio.run(composite.handle("chat", {"prompt": "hi"}))
    assert metrics.snapshot()["counters"].get("provider.success.good", 0) >= 1


def test_composite_handle_records_no_healthy_metric(metrics):
    bad = ProviderRouter("bad", [unhealthy_node("k1", "b")])
    composite = CompositeRouter([bad], metrics=metrics)
    with pytest.raises(NoHealthyClientError):
        asyncio.run(composite.handle("chat", {"prompt": "hi"}))
    assert metrics.snapshot()["counters"].get("provider.no_healthy.bad", 0) >= 1


def test_composite_stream_fails_over_before_first_token(metrics):
    class FailThenStream:
        name = "flaky"

        def __init__(self):
            self.streamed = False

        async def stream(self, prompt, **kwargs):
            self.streamed = True
            raise RuntimeError("stream start failed")
            yield  # pragma: no cover

    class FixedChunks:
        name = "good"

        async def stream(self, prompt, **kwargs):
            for part in ("Hello", " ", "world"):
                await asyncio.sleep(0)
                yield {"provider": self.name, "response": part}

    flaky = FailThenStream()
    p_flaky = ProviderRouter("flaky", [ClientNode("k1", flaky)])
    p_good = ProviderRouter("good", [ClientNode("k2", FixedChunks())])
    composite = CompositeRouter([p_flaky, p_good], metrics=metrics)

    tokens = asyncio.run(collect_stream(composite.stream("hi")))
    assert [t["response"] for t in tokens] == ["Hello", " ", "world"]
    assert flaky.streamed is True


async def collect_stream(agen):
    return [token async for token in agen]


@pytest.mark.asyncio
async def test_composite_stream_does_not_fail_over_after_first_token(metrics):
    class HalfStream:
        name = "half"

        async def stream(self, prompt, **kwargs):
            yield {"provider": "half", "response": "part1"}
            raise RuntimeError("mid-stream boom")
            yield  # pragma: no cover

    class GoodStream:
        name = "good"

        async def stream(self, prompt, **kwargs):
            yield {"provider": "good", "response": "should-not-appear"}

    p_half = ProviderRouter("half", [ClientNode("k1", HalfStream())])
    p_good = ProviderRouter("good", [ClientNode("k2", GoodStream())])
    composite = CompositeRouter([p_half, p_good], metrics=metrics)

    collected = []
    with pytest.raises(RuntimeError, match="mid-stream boom"):
        async for token in composite.stream("hi"):
            collected.append(token)

    # Only the committed provider's tokens were delivered; no silent
    # re-routing to the next provider mid-stream.
    assert [t["response"] for t in collected] == ["part1"]


def test_composite_stream_raises_when_all_fail(metrics):
    class Broken:
        name = "broken"

        async def stream(self, prompt, **kwargs):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    composite = CompositeRouter([Broken()], metrics=metrics)
    with pytest.raises(NoHealthyClientError):
        asyncio.run(collect_stream(composite.stream("hi")))


def test_provider_router_is_healthy_with_any_healthy_client():
    p = ProviderRouter(
        "p",
        [
            unhealthy_node("k1", "a"),
            healthy_node("k2", "b"),
        ],
    )
    assert asyncio.run(p.is_healthy()) is True


def test_provider_router_is_healthy_false_when_none():
    p = ProviderRouter(
        "p",
        [
            unhealthy_node("k1", "a"),
            unhealthy_node("k2", "b"),
        ],
    )
    assert asyncio.run(p.is_healthy()) is False


def test_provider_router_linear_fallback_to_next_healthy_client():
    class Failing:
        async def is_healthy(self):
            return True

        async def send(self, op, payload, **kwargs):
            raise RuntimeError("boom")

    class Working:
        async def is_healthy(self):
            return True

        async def send(self, op, payload, **kwargs):
            return {"provider": "working", "ok": True}

    p = ProviderRouter("p", [Failing(), Working()])
    result = asyncio.run(p.handle("chat", {"prompt": "hi"}))
    assert result["provider"] == "working"


def test_provider_router_scheduler_exhaustion_falls_back_to_linear_scan():
    calls = {"n": 0}

    class Flaky:
        async def is_healthy(self):
            return True

        async def send(self, op, payload, **kwargs):
            calls["n"] += 1
            raise RuntimeError("transient")

    class Backup:
        async def is_healthy(self):
            return True

        async def send(self, op, payload, **kwargs):
            return {"provider": "backup"}

    class AlwaysFirst(BaseScheduler):
        async def select(self, provider_router):
            return provider_router.clients[0]

    p = ProviderRouter("p", [Flaky(), Backup()], scheduler=AlwaysFirst())
    result = asyncio.run(p.handle("chat", {"prompt": "hi"}))
    assert result["provider"] == "backup"
    # Scheduler tried the flaky client 3 times, then the linear scan hit backup.
    assert calls["n"] == 3


def test_provider_router_preserves_last_error_when_all_clients_fail():
    class Failing:
        async def is_healthy(self):
            return True

        async def send(self, op, payload, **kwargs):
            raise ValueError("specific error")

    p = ProviderRouter("p", [Failing()])
    with pytest.raises(ValueError, match="specific error"):
        asyncio.run(p.handle("chat", {"prompt": "hi"}))


def test_provider_router_scheduler_retries_transient_errors():
    calls = {"n": 0}

    class Flaky:
        def __init__(self, name):
            self.name = name

        async def is_healthy(self):
            return True

        async def send(self, op, payload, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")
            return {"provider": self.name, "ok": True}

    nodes = [Flaky("k1")]
    p = ProviderRouter("p", nodes, scheduler=RoundRobinScheduler())
    result = asyncio.run(p.handle("chat", {"prompt": "hi"}))
    assert result["ok"] is True
    assert calls["n"] == 3


def test_provider_router_scheduler_skips_saturated_clients():
    class Busy:
        name = "busy"

        @property
        def is_saturated(self):
            return True

        async def is_healthy(self):
            return True

        async def send(self, op, payload, **kwargs):
            return {"provider": "busy"}

    class Free:
        name = "free"

        @property
        def is_saturated(self):
            return False

        async def is_healthy(self):
            return True

        async def send(self, op, payload, **kwargs):
            return {"provider": "free"}

    p = ProviderRouter("p", [Busy(), Free()], scheduler=RoundRobinScheduler())
    result = asyncio.run(p.handle("chat", {"prompt": "hi"}))
    assert result["provider"] == "free"


def test_client_node_healthy_after_no_failures():
    node = healthy_node("k1")
    assert asyncio.run(node.is_healthy()) is True


def test_client_node_unhealthy_during_cooldown():
    node = ClientNode("k1", StubClient("a"), failure_threshold=1, cooldown_seconds=60)
    node.circuit_breaker.record_failure()
    assert asyncio.run(node.is_healthy()) is False


def test_client_node_recovers_after_cooldown_expires(monkeypatch):
    node = ClientNode("k1", StubClient("a"), failure_threshold=1, cooldown_seconds=60)
    node.circuit_breaker.record_failure()
    assert asyncio.run(node.is_healthy()) is False

    now = time.monotonic()
    monkeypatch.setattr("llmrouterx.retry.circuit_breaker.time.monotonic", lambda: now + 61.0)

    assert asyncio.run(node.is_healthy()) is True
    assert node.failures == 0
    assert node.cooldown_until is None


def test_client_node_healthy_when_circuit_breaker_disabled():
    node = ClientNode(
        "k1",
        StubClient("a"),
        failure_threshold=1,
        cooldown_seconds=60,
        circuit_breaker_enabled=False,
    )
    node.last_failure = _utcnow()
    assert asyncio.run(node.is_healthy()) is True


def test_client_node_check_health_during_cooldown_false():
    node = ClientNode("k1", StubClient("a"), failure_threshold=1, cooldown_seconds=60)
    node.circuit_breaker.record_failure()
    assert asyncio.run(node.check_health(force=False)) is False


def test_client_node_check_health_force_pings_when_healthy():
    class WithHealth(StubClient):
        async def health_check(self):
            return True

    node = ClientNode("k1", WithHealth("a"))
    assert asyncio.run(node.check_health(force=True)) is True


def test_client_node_is_saturated():
    node = ClientNode("k1", StubClient("a"), max_concurrent=2)
    node.active_requests = 2
    assert node.is_saturated is True


def test_client_node_not_saturated_below_limit():
    node = ClientNode("k1", StubClient("a"), max_concurrent=5)
    node.active_requests = 3
    assert node.is_saturated is False


def test_client_node_reset_clears_failure_state():
    node = ClientNode("k1", StubClient("a"), failure_threshold=1, cooldown_seconds=60)
    node.circuit_breaker.record_failure()
    node.circuit_breaker.record_failure()
    node.circuit_breaker.record_failure()
    assert node.failures == 3

    asyncio.run(node.reset())
    assert node.failures == 0
    assert node.cooldown_until is None
    assert node.circuit_breaker.state.name == "CLOSED"


def test_client_node_rejects_bad_max_concurrent():
    from llmrouterx.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        ClientNode("k1", StubClient("a"), max_concurrent=0)


def test_client_node_rejects_bad_timeout():
    from llmrouterx.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        ClientNode("k1", StubClient("a"), timeout=0)


def test_client_node_repr_masks_api_key():
    node = ClientNode("super-secret-key-1234", StubClient("a"))
    assert "...1234" in repr(node)
    assert "super-secret-key" not in repr(node)


def test_mask_short_key_unchanged():
    from llmrouterx.client.client_node import _mask

    assert _mask("key") == "key"
    assert _mask(None) == "<unset>"
