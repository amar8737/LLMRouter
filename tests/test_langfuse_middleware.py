import pytest

from llmrouterx.context import RequestContext
from llmrouterx.middleware.langfuse_trace import LangfuseMiddleware, is_configured


class FakeObservation:
    def __init__(self):
        self.updated = None
        self.ended = False

    def update(self, **kwargs):
        self.updated = kwargs

    def end(self):
        self.ended = True


class FakeCM:
    def __init__(self, obs):
        self.obs = obs
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self.obs

    def __exit__(self, *args):
        self.exited = True
        return False


class FakeClient:
    def __init__(self):
        self.starts = []
        self.observations = []
        self.flushed = False

    def start_as_current_observation(self, **kwargs):
        self.starts.append(kwargs)
        obs = FakeObservation()
        self.observations.append(obs)
        return FakeCM(obs)

    def flush(self):
        self.flushed = True


@pytest.fixture
def fake():
    return FakeClient()


@pytest.fixture
def middleware(fake):
    return LangfuseMiddleware(client=fake)


def _context(**overrides):
    ctx = RequestContext(
        operation="chat",
        prompt="hello",
        model="stub-model",
        **overrides,
    )
    ctx.started_at = 0.0
    return ctx


async def test_records_generation_on_success(fake, middleware):
    ctx = _context()
    await middleware.before_request("chat", {"prompt": "hello", "model": "stub-model"}, ctx)
    await middleware.after_response("chat", {}, "Hello response", ctx)

    assert len(fake.starts) == 1
    assert fake.starts[0]["as_type"] == "generation"
    assert fake.starts[0]["input"] == "hello"
    assert fake.starts[0]["model"] == "stub-model"

    obs = fake.observations[0]
    assert obs.ended is True
    assert obs.updated["output"] == "Hello response"
    assert obs.updated["metadata"]["request_id"] == ctx.request_id
    assert obs.updated["metadata"]["operation"] == "chat"


async def test_records_error_on_exception(fake, middleware):
    ctx = _context()
    await middleware.before_request("chat", {"prompt": "hello"}, ctx)
    await middleware.on_exception("chat", {}, ValueError("boom"), ctx)

    obs = fake.observations[0]
    assert obs.ended is True
    assert obs.updated["level"] == "ERROR"
    assert "boom" in obs.updated["status_message"]


async def test_coerces_dict_output(fake, middleware):
    ctx = _context()
    await middleware.before_request("chat", {"prompt": "hello"}, ctx)
    await middleware.after_response("chat", {}, {"response": "nested text"}, ctx)
    assert fake.observations[0].updated["output"] == "nested text"


async def test_flush_delegates_to_client(fake, middleware):
    middleware.flush()
    assert fake.flushed is True


def test_is_configured_requires_host(monkeypatch):
    for name in (
        "LANGFUSE_HOST",
        "LANGFUSE_BASE_URL",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    assert is_configured() is False

    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-x")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-x")
    assert is_configured() is True
