import json

import pytest

from llmrouterx.cli import main as cli_main
from llmrouterx.client.client_node import ClientNode
from llmrouterx.config.config import RouterConfig
from llmrouterx.config.secrets import KeyResolutionError, resolve_key
from llmrouterx.providers.composite_router import CompositeRouter
from llmrouterx.providers.provider_router import ProviderRouter
from llmrouterx.providers.stub_provider import StubClient
from llmrouterx.retry.exponential import HTTPError
from llmrouterx.router.llmrouter import LLMRouter

try:
    from fastapi.testclient import TestClient

    from llmrouterx.server.app import create_app
except ImportError:
    pytest.skip(
        "fastapi / server extras not installed; run: pip install 'llmrouterx[server]'",
        allow_module_level=True,
    )


@pytest.fixture
def test_router():
    client = StubClient("test_stub")
    node = ClientNode("key1", client)
    provider = ProviderRouter("stub_provider", [node])
    composite = CompositeRouter([provider])
    return LLMRouter(composite)


@pytest.fixture
def client(test_router):
    app = create_app(router=test_router, api_keys=["test-api-key"])
    with TestClient(app) as test_client:
        # Add default auth header for all requests
        test_client.headers = {"Authorization": "Bearer test-api-key"}
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["providers"]["stub_provider"] is True
    assert data["healthy_count"] == 1
    assert data["total_providers"] == 1
    assert data["uptime_seconds"] >= 0


def test_metrics_endpoint(client):
    client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "snapshot" in data
    assert "derived" in data
    assert "counters" in data["snapshot"]
    assert "requests.chat" in data["snapshot"]["counters"]
    assert data["uptime_seconds"] >= 0


def test_list_models(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert "stub_provider" in ids


def test_chat_completions_non_streaming(client):
    payload = {
        "model": "stub-model",
        "messages": [{"role": "user", "content": "Hello World"}],
        "stream": False,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "stub-model"
    assert "Hello World" in data["choices"][0]["message"]["content"]


def test_chat_completions_streaming(client):
    payload = {
        "model": "stub-model",
        "messages": [{"role": "user", "content": "Hello World"}],
        "stream": True,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "data:" in response.text
    assert "[DONE]" in response.text


def test_chat_completions_empty_messages_rejected(client):
    response = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["message"] == "messages must contain at least one message"
    assert body["error"]["type"] == "invalid_request_error"


def test_validation_error_uses_openai_envelope(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": 123, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"


def test_chat_completions_returns_503_when_no_healthy_provider():
    class Broken:
        async def chat(self, prompt, **kwargs):
            raise HTTPError(503, "down")

        async def embeddings(self, text, **kwargs):
            raise HTTPError(503, "down")

        async def responses(self, *args, **kwargs):
            raise HTTPError(503, "down")

        async def stream(self, prompt, **kwargs):
            raise HTTPError(503, "down")
            yield  # pragma: no cover

    node = ClientNode("k1", Broken(), failure_threshold=1, cooldown_seconds=60)
    # Open the breaker so the router raises NoHealthyClientError -> HTTP 503.
    node.circuit_breaker.record_failure()
    provider = ProviderRouter("broken_provider", [node])
    router = LLMRouter(CompositeRouter([provider]))

    app = create_app(router=router, api_keys=["test-api-key"])
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer test-api-key"},
        )
    assert response.status_code == 503


def test_api_key_auth_blocks_unauthenticated_chat(test_router):
    app = create_app(router=test_router, api_keys=["sk-test"])
    with TestClient(app) as test_client:
        assert (
            test_client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            ).status_code
            == 401
        )
        assert (
            test_client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer wrong"},
            ).status_code
            == 401
        )
        assert (
            test_client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer sk-test"},
            ).status_code
            == 200
        )


def test_admin_token_protects_dashboard_and_metrics(test_router):
    app = create_app(router=test_router, admin_token="admin-secret")
    with TestClient(app) as test_client:
        assert test_client.get("/metrics").status_code == 401
        assert test_client.get("/dashboard").status_code == 401
        assert (
            test_client.get(
                "/metrics", headers={"Authorization": "Bearer admin-secret"}
            ).status_code
            == 200
        )
        assert (
            test_client.get(
                "/dashboard", headers={"Authorization": "Bearer admin-secret"}
            ).status_code
            == 200
        )


def test_docs_disabled_returns_404(test_router):
    app = create_app(router=test_router, docs_enabled=False)
    with TestClient(app) as test_client:
        assert test_client.get("/docs").status_code == 404


def test_streaming_error_emits_503_and_done():
    class Broken:
        async def chat(self, prompt, **kwargs):
            raise HTTPError(503, "down")

        async def embeddings(self, text, **kwargs):
            raise HTTPError(503, "down")

        async def responses(self, *args, **kwargs):
            raise HTTPError(503, "down")

        async def stream(self, prompt, **kwargs):
            raise HTTPError(503, "down")
            yield  # pragma: no cover

    node = ClientNode("k1", Broken(), failure_threshold=1, cooldown_seconds=60)
    node.circuit_breaker.record_failure()
    provider = ProviderRouter("broken_provider", [node])
    router = LLMRouter(CompositeRouter([provider]))

    app = create_app(router=router, api_keys=["test-api-key"])
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            headers={"Authorization": "Bearer test-api-key"},
        )
        assert response.status_code == 200
        text = response.text
        assert "[DONE]" in text
        assert "data:" in text


def test_cli_help(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["llmrouterx", "--help"])
    with pytest.raises(SystemExit):
        cli_main()
    captured = capsys.readouterr()
    assert "LLMRouter CLI" in captured.out


def test_cli_serve_has_config_and_workers_flags(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["llmrouterx", "serve", "--help"])
    with pytest.raises(SystemExit):
        cli_main()
    captured = capsys.readouterr()
    assert "--config" in captured.out
    assert "--workers" in captured.out
    assert "--log-level" in captured.out
    assert "--admin-token" in captured.out
    assert "--api-key" in captured.out
    assert "--no-docs" in captured.out


def test_config_roundtrip_from_dict():
    cfg = RouterConfig(
        providers=[{"name": "p", "clients": [{"client": "x", "api_key": "k"}]}],
        timeout=10,
        max_retries=5,
        total_timeout=30.0,
    )
    loaded = RouterConfig.from_dict(cfg.to_dict())
    assert loaded.timeout == 10
    assert loaded.max_retries == 5
    assert loaded.total_timeout == 30.0
    assert loaded.providers == cfg.providers


def test_config_from_file(tmp_path):
    path = tmp_path / "router.json"
    path.write_text(
        json.dumps(
            {
                "providers": [{"name": "p", "clients": [{"client": "x", "api_key": "k"}]}],
                "timeout": 20,
            }
        )
    )
    cfg = RouterConfig.from_file(path)
    assert cfg.timeout == 20
    assert cfg.providers[0]["name"] == "p"


def test_config_from_file_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"bogus_key": 1}))
    with pytest.raises(ValueError, match="Unknown config keys"):
        RouterConfig.from_file(path)


def test_config_validate_detects_missing_client_field():
    cfg = RouterConfig(providers=[{"name": "p", "clients": [{"api_key": "k"}]}])
    with pytest.raises(ValueError, match="without a 'client' field"):
        cfg.validate()


def test_config_bool_parsing_string_false():
    cfg = RouterConfig.from_dict(
        {
            "providers": [{"name": "p", "clients": [{"client": "x", "api_key": "k"}]}],
            "enable_circuit_breaker": "false",
        }
    )
    assert cfg.enable_circuit_breaker is False
    cfg_true = RouterConfig.from_dict(
        {
            "providers": [{"name": "p", "clients": [{"client": "x", "api_key": "k"}]}],
            "enable_circuit_breaker": "true",
        }
    )
    assert cfg_true.enable_circuit_breaker is True


def test_resolve_key_prefers_literal():
    assert resolve_key({"api_key": "literal"}) == "literal"


def test_resolve_key_from_env(monkeypatch):
    monkeypatch.setenv("LLMRouter_TEST_KEY", "env-secret")
    assert resolve_key({"api_key_env": "LLMRouter_TEST_KEY"}) == "env-secret"


def test_resolve_key_from_file(tmp_path):
    key_file = tmp_path / "key.txt"
    key_file.write_text("file-secret\n")
    assert resolve_key({"api_key_file": str(key_file)}) == "file-secret"


def test_resolve_key_missing_env(monkeypatch):
    monkeypatch.delenv("LLMRouter_MISSING_KEY", raising=False)
    with pytest.raises(KeyResolutionError, match="not set"):
        resolve_key({"api_key_env": "LLMRouter_MISSING_KEY"})


def test_resolve_key_no_source():
    with pytest.raises(KeyResolutionError, match="missing an API key"):
        resolve_key({"client": "x"})


def test_resolve_keys_materialises_literal(tmp_path):
    cfg = RouterConfig.from_dict(
        {"providers": [{"name": "p", "clients": [{"client": "x", "api_key": "k"}]}]}
    )
    assert cfg.providers[0]["clients"][0]["api_key"] == "k"


def test_resolve_keys_from_env_in_config(monkeypatch):
    monkeypatch.setenv("LLMRouter_CFG_KEY", "resolved-secret")
    cfg = RouterConfig.from_dict(
        {
            "providers": [
                {"name": "p", "clients": [{"client": "x", "api_key_env": "LLMRouter_CFG_KEY"}]}
            ]
        }
    )
    assert cfg.providers[0]["clients"][0]["api_key"] == "resolved-secret"
