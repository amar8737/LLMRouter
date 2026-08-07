import pytest

from llmrouterx.adapters.base import BaseProviderAdapter
from llmrouterx.adapters.factory import AdapterFactory
from llmrouterx.config.config import RouterConfig
from llmrouterx.exceptions import ConfigurationError
from llmrouterx.registry.provider_registry import ProviderRegistry


@pytest.fixture
def valid_config():
    return RouterConfig(
        providers=[
            {
                "name": "openai",
                "clients": [
                    {"api_key": "sk-1", "client": object()},
                ],
            }
        ],
        timeout=30.0,
        max_retries=2,
    )


def test_validate_accepts_valid_config(valid_config):
    valid_config.validate()


def test_validate_requires_providers():
    with pytest.raises(ValueError, match="At least one provider"):
        RouterConfig().validate()


def test_validate_rejects_non_positive_timeout(valid_config):
    with pytest.raises(ValueError, match="timeout"):
        RouterConfig(
            providers=valid_config.providers,
            timeout=0,
        ).validate()


def test_validate_rejects_negative_max_retries(valid_config):
    with pytest.raises(ValueError, match="max_retries"):
        RouterConfig(
            providers=valid_config.providers,
            max_retries=-1,
        ).validate()


def test_validate_rejects_non_positive_max_concurrent(valid_config):
    with pytest.raises(ValueError, match="max_concurrent_per_key"):
        RouterConfig(
            providers=valid_config.providers,
            max_concurrent_per_key=0,
        ).validate()


def test_validate_rejects_zero_circuit_threshold(valid_config):
    with pytest.raises(ValueError, match="circuit_breaker_threshold"):
        RouterConfig(
            providers=valid_config.providers,
            circuit_breaker_threshold=0,
        ).validate()


def test_validate_rejects_provider_missing_name():
    with pytest.raises(ValueError, match="'name' key"):
        RouterConfig(providers=[{"clients": [{"api_key": "x", "client": object()}]}]).validate()


def test_validate_rejects_provider_without_clients():
    with pytest.raises(ValueError, match="at least one client"):
        RouterConfig(providers=[{"name": "openai", "clients": []}]).validate()


def test_validate_rejects_non_dict_provider():
    with pytest.raises(ValueError, match="must be a dict"):
        RouterConfig(providers=["openai"]).validate()


def test_from_env_parses_defaults(monkeypatch):
    monkeypatch.delenv("LLMROUTER_TIMEOUT", raising=False)
    monkeypatch.delenv("LLMROUTER_MAX_RETRIES", raising=False)
    monkeypatch.delenv("LLMROUTER_MAX_CONCURRENT", raising=False)
    monkeypatch.delenv("LLMROUTER_MAX_CONCURRENT_REQUESTS", raising=False)
    monkeypatch.delenv("LLMROUTER_CIRCUIT_BREAKER", raising=False)
    monkeypatch.delenv("LLMROUTER_CB_THRESHOLD", raising=False)
    monkeypatch.delenv("LLMROUTER_CB_RESET_TIMEOUT", raising=False)

    cfg = RouterConfig.from_env()
    assert cfg.timeout == 60.0
    assert cfg.max_retries == 3
    assert cfg.max_concurrent_per_key == 100
    assert cfg.max_concurrent_requests is None
    assert cfg.enable_circuit_breaker is True


def test_from_env_parses_overrides(monkeypatch):
    monkeypatch.setenv("LLMROUTER_TIMEOUT", "10")
    monkeypatch.setenv("LLMROUTER_MAX_RETRIES", "5")
    monkeypatch.setenv("LLMROUTER_MAX_CONCURRENT", "7")
    monkeypatch.setenv("LLMROUTER_MAX_CONCURRENT_REQUESTS", "3")
    monkeypatch.setenv("LLMROUTER_CIRCUIT_BREAKER", "false")
    monkeypatch.setenv("LLMROUTER_CB_THRESHOLD", "8")
    monkeypatch.setenv("LLMROUTER_CB_RESET_TIMEOUT", "45")

    cfg = RouterConfig.from_env()
    assert cfg.timeout == 10.0
    assert cfg.max_retries == 5
    assert cfg.max_concurrent_per_key == 7
    assert cfg.max_concurrent_requests == 3
    assert cfg.enable_circuit_breaker is False
    assert cfg.circuit_breaker_threshold == 8
    assert cfg.circuit_breaker_reset_timeout == 45.0


def test_copy_with_updates(valid_config):
    copied = valid_config.copy(timeout=99.0, max_retries=1)
    assert copied.timeout == 99.0
    assert copied.max_retries == 1
    assert copied.providers == valid_config.providers


def test_copy_mutating_update_does_not_leak(valid_config):
    copied = valid_config.copy()
    copied.providers = []
    assert valid_config.providers


class _DummyAdapter(BaseProviderAdapter):
    async def chat(self, prompt, **kwargs):
        return "dummy"

    async def stream(self, prompt, **kwargs):
        yield "dummy"

    async def embeddings(self, text, **kwargs):
        return []


@pytest.fixture
def dummy_adapter_cls():
    return _DummyAdapter


def test_adapter_factory_passes_through_existing_adapter(dummy_adapter_cls):
    adapter = dummy_adapter_cls(client=object())
    created = AdapterFactory.create(provider="openai", client=adapter)
    assert created is adapter


def test_adapter_factory_create_known_provider(openai_client):
    adapter = AdapterFactory.create(
        provider="openai",
        client=openai_client,
        default_model="gpt-4",
    )
    assert isinstance(adapter, BaseProviderAdapter)
    assert adapter.default_model == "gpt-4"
    assert adapter.client is openai_client


def test_adapter_factory_create_unknown_provider():
    with pytest.raises(ConfigurationError, match="Unsupported provider"):
        AdapterFactory.create(provider="not-a-provider", client=object())


def test_adapter_factory_create_requires_client():
    with pytest.raises(ConfigurationError, match="SDK client instance is required"):
        AdapterFactory.create(provider="openai", client=None)


def test_adapter_factory_register_new(dummy_adapter_cls):
    AdapterFactory.register("my_provider", dummy_adapter_cls)
    assert "my_provider" in AdapterFactory.supported_providers()
    AdapterFactory.unregister("my_provider")


def test_adapter_factory_register_rejects_blank():
    with pytest.raises(ConfigurationError, match="non-empty"):
        AdapterFactory.register("", _DummyAdapter)


def test_adapter_factory_register_rejects_non_subclass():
    with pytest.raises(ConfigurationError, match="subclass"):
        AdapterFactory.register("bad_provider", int)


def test_adapter_factory_register_duplicate_without_overwrite(dummy_adapter_cls):
    with pytest.raises(ConfigurationError, match="already registered"):
        AdapterFactory.register("openai", dummy_adapter_cls, overwrite=False)


def test_adapter_factory_supported_providers_contains_known():
    supported = AdapterFactory.supported_providers()
    for expected in ("openai", "anthropic", "azure", "gemini", "groq", "together", "nim"):
        assert expected in supported


class FakeProvider:
    def __init__(self, name, healthy=True):
        self.name = name
        self._healthy = healthy

    async def is_healthy(self):
        return self._healthy


def test_registry_register_and_get():
    registry = ProviderRegistry()
    provider = FakeProvider("p1")
    registry.register(provider)
    assert registry.get("p1") is provider
    assert "p1" in registry
    assert len(registry) == 1


def test_registry_register_duplicate_raises():
    registry = ProviderRegistry()
    registry.register(FakeProvider("p1"))
    with pytest.raises(ValueError, match="already exists"):
        registry.register(FakeProvider("p1"))


def test_registry_register_overwrite():
    registry = ProviderRegistry()
    first = FakeProvider("p1")
    second = FakeProvider("p1")
    registry.register(first)
    registry.register(second, overwrite=True)
    assert registry.get("p1") is second


def test_registry_unregister():
    registry = ProviderRegistry()
    registry.register(FakeProvider("p1"))
    assert registry.unregister("p1").name == "p1"
    assert "p1" not in registry


def test_registry_unregister_unknown_raises():
    registry = ProviderRegistry()
    with pytest.raises(KeyError, match="Unknown provider"):
        registry.unregister("nope")


def test_registry_get_unknown_raises():
    registry = ProviderRegistry()
    with pytest.raises(KeyError, match="Unknown provider"):
        registry.get("nope")


def test_registry_exists_and_all():
    registry = ProviderRegistry()
    registry.register(FakeProvider("a"))
    registry.register(FakeProvider("b"))
    assert registry.exists("a")
    assert not registry.exists("z")
    assert sorted(p.name for p in registry.all()) == ["a", "b"]


@pytest.mark.asyncio
async def test_registry_healthy_filters():
    registry = ProviderRegistry()
    registry.register(FakeProvider("good", healthy=True))
    registry.register(FakeProvider("bad", healthy=False))
    healthy = await registry.healthy()
    assert [p.name for p in healthy] == ["good"]


def test_registry_iter_and_contains():
    registry = ProviderRegistry()
    registry.register(FakeProvider("a"))
    assert [p.name for p in registry] == ["a"]


def test_registry_clear_and_len():
    registry = ProviderRegistry()
    registry.register(FakeProvider("a"))
    registry.register(FakeProvider("b"))
    registry.clear()
    assert len(registry) == 0


def test_registry_repr():
    registry = ProviderRegistry()
    registry.register(FakeProvider("a"))
    assert "a" in repr(registry)
