# LLMRouter 🚀

**Intelligent routing, load balancing, and failover for LLM providers**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apache 2.0 License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/package-llmrouterx-blue.svg)](https://pypi.org/project/llmrouterx/)

LLMRouter is a Python package that acts as a routing layer between your application and LLM providers (OpenAI, Groq, Together AI, etc.). It handles provider selection, API key rotation, automatic failover, retries, and metrics—so you don't have to.

```python
# Instead of managing providers manually:
client = OpenAI(api_key="sk-...")
response = client.chat.completions.create(model="gpt-4", messages=[...])

# Use LLMRouter for intelligent routing:
router = LLMRouter(providers=[openai_provider, groq_provider])
response = await router.chat(prompt="Hello!")
# Router automatically picks the best provider, rotates keys, retries on failure, tracks metrics
```

---

## ✨ Features

### Core Capabilities (v0.1+)
- 🔄 **Automatic Provider Failover** — Switch to backup provider if primary fails
- 🔑 **API Key Pooling** — Rotate across multiple keys to avoid rate limits
- ⏱️ **Intelligent Scheduling** — Pick the best client (least busy, round-robin, random, weighted, priority)
- 🔁 **Automatic Retries** — Exponential backoff with jitter for transient failures
- 📊 **Metrics Collection** — Track requests, errors, latency, throughput, success rates
- 🏷️ **Labeled Metrics** — Per-provider counters and timings out of the box
- 🔌 **Extensible Design** — Custom schedulers, retry policies, middleware
- 🔍 **Health Checks** — Monitor provider availability in real-time, with timeouts
- 📡 **Streaming Support** — Stream responses from any provider
- ⏳ **Total Request Timeout** — Hard deadline covering retries and failover
- 🚦 **Per-Key Circuit Breakers** — Isolate a failing key without taking down the router
- 🔁 **Retry Middleware Hook** — `on_retry` lets middleware veto individual retries
- 📝 **Structured Logging** — Built-in JSON formatter for machine-readable logs
- 🛡️ **Error Handling** — Graceful degradation with meaningful exceptions
- 🔐 **Gateway Auth** — optional bearer-token tiers (admin for `/dashboard`+`/metrics`, API key for `/v1/*`) via `create_app(..., admin_token=, api_keys=)` or env/CLI
- 🖥️ **HTTP Gateway** — OpenAI-compatible server (`/v1/chat/completions`, streaming, health, metrics, dashboard)
- 📄 **Declarative Config** — JSON config files with env/file-based key loading (`api_key_env` / `api_key_file`)
- 🔭 **Langfuse Tracing** — Zero-code observability via middleware
- 📊 **Dashboard & Token Tracking** — zero-config UI plus global/per-provider token counters
- 🛠️ **Actionable Errors** — `NoHealthyClientError` reports the full `Failure sequence` across providers

### Planned Features (v0.2+)
- 💰 Cost-aware routing (pick cheapest provider)
- ⚡ Latency-aware routing (pick fastest provider)
- 🌍 Region-aware routing
- 💾 Response caching
- 🔐 Authentication gateway
- 📉 Prometheus metrics export

---

## 📦 Installation

### From PyPI
```bash
pip install llmrouterx
```

### From Source (Development)
```bash
git clone https://github.com/amar8737/LLMRouter.git
cd LLMRouter
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### Requirements
- **Python 3.10+** (tested on 3.10, 3.11, 3.12)
- No external LLM SDKs required (bring your own: OpenAI, Groq, Together AI, etc.)

---

## 🚀 Quick Start (5 Minutes)

### 1. Install the package
```bash
pip install llmrouterx
```

### 2. Basic example with OpenAI
```python
import asyncio
from openai import AsyncOpenAI
from llmrouterx import LLMRouter
from llmrouterx.adapters import AdapterFactory
from llmrouterx.client import ClientNode
from llmrouterx.providers import ProviderRouter, CompositeRouter


async def main():
    # Create an OpenAI client and wrap it in the OpenAI adapter
    sdk_client = AsyncOpenAI(api_key="sk-your-key-here")
    adapter = AdapterFactory.create(provider="openai", client=sdk_client, default_model="gpt-4")

    # Wrap it in a ClientNode (tracks identity and health)
    node = ClientNode("sk-your-key-here", adapter)

    # Create a provider (represents a service like OpenAI)
    provider = ProviderRouter("openai", [node])

    # Create a composite router (aggregates providers)
    composite = CompositeRouter([provider])

    # Create the router
    router = LLMRouter(composite)

    # Make a request
    response = await router.chat(prompt="Hello, what's 2+2?")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Run tests to verify installation
```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## 💡 Usage Examples

### Example 1: Single Provider with One Key

```python
import asyncio
from openai import AsyncOpenAI
from llmrouterx import LLMRouter
from llmrouterx.adapters import AdapterFactory
from llmrouterx.client import ClientNode
from llmrouterx.providers import ProviderRouter, CompositeRouter


async def main():
    # Wrap the SDK client in adapter + ClientNode
    client = AsyncOpenAI(api_key="sk-...")
    adapter = AdapterFactory.create(provider="openai", client=client, default_model="gpt-4")
    node = ClientNode("sk-...", adapter)

    provider = ProviderRouter("openai", [node])
    composite = CompositeRouter([provider])
    router = LLMRouter(composite)

    response = await router.chat(prompt="Hello!")
    print(response)


asyncio.run(main())
```

---

### Example 2: Multiple API Keys (Rate Limit Protection)

```python
import asyncio
from openai import AsyncOpenAI
from llmrouterx import LLMRouter
from llmrouterx.adapters import AdapterFactory
from llmrouterx.client import ClientNode
from llmrouterx.providers import ProviderRouter, CompositeRouter
from llmrouterx.scheduler import LeastBusyScheduler


async def main():
    keys = ["sk-key1", "sk-key2", "sk-key3"]
    nodes = [
        ClientNode(
            key,
            AdapterFactory.create(
                provider="openai", client=AsyncOpenAI(api_key=key), default_model="gpt-4"
            ),
        )
        for key in keys
    ]

    # LeastBusyScheduler rotates across keys, avoiding rate limits
    provider = ProviderRouter("openai", nodes, scheduler=LeastBusyScheduler())
    composite = CompositeRouter([provider])
    router = LLMRouter(composite)

    response = await router.chat(prompt="Hello!")
    print(response)


asyncio.run(main())
```

---

### Example 3: Multiple Providers with Automatic Failover

```python
import asyncio
from openai import AsyncOpenAI
from groq import AsyncGroq
from llmrouterx import LLMRouter
from llmrouterx.adapters import AdapterFactory
from llmrouterx.client import ClientNode
from llmrouterx.providers import ProviderRouter, CompositeRouter
from llmrouterx.scheduler import LeastBusyScheduler


async def main():
    def node(provider, sdk, key):
        adapter = AdapterFactory.create(provider=provider, client=sdk, default_model="gpt-4")
        return ClientNode(key, adapter)

    # Primary: OpenAI
    openai_provider = ProviderRouter(
        "openai",
        [node("openai", AsyncOpenAI(api_key="sk-..."), "sk-...")],
        scheduler=LeastBusyScheduler(),
    )

    # Fallback: Groq
    groq_provider = ProviderRouter(
        "groq",
        [node("groq", AsyncGroq(api_key="gsk-..."), "gsk-...")],
        scheduler=LeastBusyScheduler(),
    )

    # CompositeRouter tries providers in order; if first fails, tries next
    composite = CompositeRouter([openai_provider, groq_provider])
    router = LLMRouter(composite)

    # If OpenAI fails, automatically falls back to Groq
    response = await router.chat(prompt="Hello!")
    print(response)


asyncio.run(main())
```

---

### Example 4: Streaming Responses

```python
import asyncio
from llmrouterx import LLMRouter


async def main():
    router = LLMRouter(composite)  # from previous examples

    # Stream response chunks
    async for chunk in router.stream(prompt="Tell me a story"):
        print(chunk, end="", flush=True)
    print()


asyncio.run(main())
```

---

### Example 5: Concurrent Requests

```python
import asyncio
from llmrouterx import LLMRouter


async def main():
    router = LLMRouter(composite)

    # Make 10 concurrent requests
    prompts = [f"Request {i}: tell me a fact" for i in range(10)]
    responses = await asyncio.gather(*[router.chat(prompt=p) for p in prompts])

    print(f"Got {len(responses)} responses")
    for i, resp in enumerate(responses):
        print(f"{i}: {resp[:100]}...")


asyncio.run(main())
```

---

### Example 6: Custom Retry Policy

```python
from llmrouterx.retry import ExponentialRetry
from llmrouterx import LLMRouter

# Retry with exponential backoff
retry = ExponentialRetry(
    max_retries=5,  # Try up to 5 times
    base=0.5,  # Start with 0.5 second wait
    factor=2.0,  # Double wait time each retry
    max_backoff=30.0,  # Cap wait at 30 seconds
)

router = LLMRouter(composite, retry=retry)
```

---

### Example 7: Monitor Metrics

```python
import asyncio
from llmrouterx import LLMRouter


async def main():
    router = LLMRouter(composite)

    # Make some requests
    for i in range(10):
        await router.chat(prompt=f"Request {i}")

    # View metrics
    metrics = router.metrics.get()
    print("Total chat requests:", metrics["counters"].get("requests.chat", 0))
    print("Total errors:", metrics["counters"].get("errors.chat.non_retryable", 0))
    print(
        "Average chat latency:",
        sum(metrics["timings"].get("latency.chat", []))
        / len(metrics["timings"].get("latency.chat", []))
        if metrics["timings"].get("latency.chat", [])
        else 0,
    )


asyncio.run(main())
```

---

### Example 8: Custom Middleware for Logging

```python
import logging
from llmrouterx.middleware import BaseMiddleware
from llmrouterx import LLMRouter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseMiddleware):
    async def before_request(self, operation, payload, context):
        logger.info("→ %s [%s]: %s", operation, context.request_id, payload)
        return payload

    async def after_response(self, operation, payload, response, context):
        logger.info("← %s [%s]: response received", operation, context.request_id)
        return response


# Use it
router = LLMRouter(composite, middleware=[LoggingMiddleware()])
```

---

### Example 9: All Available Schedulers

```python
from llmrouterx.scheduler import (
    LeastBusyScheduler,  # Pick client with fewest active requests
    RoundRobinScheduler,  # Rotate through clients sequentially
    RandomScheduler,  # Pick random client
    WeightedScheduler,  # Pick by node.weight property
    PriorityScheduler,  # Pick by node.priority property
)

# Use any scheduler
provider = ProviderRouter(
    "name",
    nodes,
    scheduler=LeastBusyScheduler(),  # or any scheduler above
)
```

---

### Example 10: Check Provider Health

```python
async def check_health():
    composite = CompositeRouter([provider1, provider2])
    
    for provider in composite.providers:
        is_healthy = await provider.is_healthy()
        status = "✓ Up" if is_healthy else "✗ Down"
        print(f"{provider.name}: {status}")
```

---

### Example 11: Error Handling

```python
import asyncio
from llmrouterx import LLMRouter
from llmrouterx.exceptions import NoHealthyClientError


async def main():
    router = LLMRouter(composite)

    try:
        response = await router.chat(prompt="Hello!")
    except NoHealthyClientError as e:
        print(f"All providers are down: {e}")
    except Exception as e:
        print(f"Request failed: {e}")


asyncio.run(main())
```

---

### Example 12: Total Request Timeout

A hard deadline that covers the whole operation, including retries and
failover. Raise `asyncio.TimeoutError` when the budget is exhausted instead of
letting a slow provider consume all retries:

```python
import asyncio
from llmrouterx import LLMRouter

router = LLMRouter(composite, total_timeout=30.0)

try:
    response = await router.chat(prompt="Hello!")
except asyncio.TimeoutError:
    print("Operation took longer than 30s and was cancelled.")
```

---

### Example 13: Retry Middleware Hook

`on_retry` runs before every retry. Return `False` to veto the retry (e.g. for
idempotency or cost control); return `True` to allow it. Errors raised inside
the hook are logged and treated as `True` so observers can never break retry
handling:

```python
from llmrouterx.middleware import BaseMiddleware


class RateLimitAwareMiddleware(BaseMiddleware):
    async def on_retry(self, operation, payload, exception, attempt, context):
        # Never burn retries on a dead API key.
        if getattr(exception, "status_code", None) == 401:
            return False
        return True


router = LLMRouter(composite, middleware=[RateLimitAwareMiddleware()])
```

---

### Example 14: Structured (JSON) Logging

```python
import logging
from llmrouterx.utils import setup_logging

setup_logging(level=logging.INFO, fmt="json")  # one-line JSON records on stderr

logger = logging.getLogger("llmrouterx")
logger.info(
    "routed",
    extra={"_llmrouterx_extra": {"request_id": ctx.request_id, "provider": "openai"}},
)
```

---

## 🔑 Core Concepts

### ClientNode
Represents a single LLM client (e.g., one OpenAI API key). Wraps a provider
**adapter** (not a raw SDK client) and tracks health, active requests, and
metadata.

```python
adapter = AdapterFactory.create(provider="openai", client=client_instance, default_model="gpt-4")
node = ClientNode("identifier", adapter)  # the adapter, not the raw SDK client
node.weight = 2  # Optional: used by WeightedScheduler
node.priority = 10  # Optional: used by PriorityScheduler
```

### ProviderRouter
Aggregates multiple ClientNodes for a single provider (e.g., OpenAI with 3 keys). Selects the best node using a scheduler.

```python
provider = ProviderRouter(
    "openai",  # Provider name
    [node1, node2, node3],  # List of nodes
    scheduler=LeastBusyScheduler(),  # How to select among nodes
)
```

### CompositeRouter
Aggregates multiple ProviderRouters. Tries providers in order; if primary fails, tries next.

```python
composite = CompositeRouter([openai_provider, groq_provider, fallback_provider])
```

### LLMRouter
Main router. Orchestrates composite router, retry logic, middleware, and metrics.

```python
router = LLMRouter(
    composite_router,
    retry=ExponentialRetry(),
    middleware=[LoggingMiddleware()],
)
```

---

## 📊 Metrics

Track performance across your LLM infrastructure:

```python
metrics = router.metrics.get()  # alias for .snapshot()

# Counters
print(metrics["counters"]["requests.chat"])
print(metrics["counters"]["errors.chat.retries_exhausted"])

# Timings (bounded list of request latencies in seconds)
print(metrics["timings"]["latency.chat"])  # [0.45, 0.52, 0.38, ...]

# Labeled metrics (per-provider breakdown)
print(metrics["labeled_counters"]["requests.chat"])
# {"provider=openai": 12, "provider=groq": 4}
```

The router automatically records labeled `requests.<op>` / `latency.<op>`
metrics keyed by the provider that served the request. Compute summary
statistics for a timing series with `timing_stats`:

```python
stats = router.metrics.timing_stats("latency.chat")
print(stats)  # {"count": 10, "min": ..., "mean": ..., "p95": ..., "p99": ...}
```

You can also attach your own labels when recording directly:

```python
router.metrics.incr("billing.tokens", amount=512, labels={"tenant": "acme"})
router.metrics.timing("kv.get", 0.004, labels={"cache": "hit"})
```

### Token usage

Provider adapters automatically extract token usage from successful responses
and the router records it as global and per-provider counters:

```python
m = router.metrics.get()
print(m["counters"]["tokens.prompt.total"])  # global
print(m["labeled_counters"]["tokens.total"]["provider=openai"])  # per-provider
print(m["counters"]["tokens.completion.total"])
```

To record usage manually (e.g. from a custom adapter):

```python
router.metrics.track_tokens("openai", prompt_tokens=120, completion_tokens=45)
```

---

## 🔌 Extensibility

### Custom Scheduler

```python
from llmrouterx.scheduler import BaseScheduler


class MyScheduler(BaseScheduler):
    async def select(self, provider_router):
        candidates = [c for c in provider_router.clients if await c.is_healthy()]
        if not candidates:
            return None
        # Your logic here
        return candidates[0]


provider = ProviderRouter("name", nodes, scheduler=MyScheduler())
```

### Custom Middleware

```python
from llmrouterx.middleware import BaseMiddleware


class MyMiddleware(BaseMiddleware):
    async def before_request(self, operation, payload, context):
        # Modify request before sending
        return payload

    async def after_response(self, operation, payload, response, context):
        # Transform response after receiving
        return response

    async def on_retry(self, operation, payload, exception, attempt, context):
        # Veto a retry by returning False
        return True


router = LLMRouter(composite, middleware=[MyMiddleware()])
```

### Custom Retry Policy

```python
from llmrouterx.retry import BaseRetry


class MyRetry(BaseRetry):
    async def should_retry(self, error, attempt):
        # Your logic to decide if we should retry
        return attempt < 3


router = LLMRouter(composite, retry=MyRetry())
```

---

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test
```bash
pytest tests/test_retry_middleware_and_timeouts.py -v
```

### Coverage Report
```bash
pytest tests/ --cov=llmrouterx --cov-report=term-missing
```

### Static Checks
```bash
ruff check . && ruff format --check .   # lint + format
mypy                                   # type check the llmrouterx package
```

### Run Smoke Test
```bash
python tests/run_tests.py
```

### Write Your Own Test
```python
import pytest
from llmrouterx import LLMRouter
from llmrouterx.providers import StubClient, ProviderRouter, CompositeRouter
from llmrouterx.client import ClientNode


@pytest.mark.asyncio
async def test_basic_chat():
    stub = StubClient("test")
    node = ClientNode("test", stub)
    provider = ProviderRouter("test", [node])
    composite = CompositeRouter([provider])
    router = LLMRouter(composite)

    response = await router.chat(prompt="Hello")
    assert response is not None
```

---

## 🔧 Configuration

### Minimal Setup
```python
from llmrouterx import LLMRouter
from llmrouterx.providers import CompositeRouter, ProviderRouter
from llmrouterx.client import ClientNode

node = ClientNode("key", client)
provider = ProviderRouter("openai", [node])
composite = CompositeRouter([provider])
router = LLMRouter(composite)
```

### Full Setup with All Options
```python
from llmrouterx import LLMRouter
from llmrouterx.retry import ExponentialRetry
from llmrouterx.metrics import MetricsCollector
from llmrouterx.middleware import BaseMiddleware


class LogMiddleware(BaseMiddleware):
    async def before_request(self, operation, payload, context):
        print(f"Sending: {operation}")
        return payload


retry = ExponentialRetry(max_retries=5, base=1.0, factor=2.0)

router = LLMRouter(
    composite,
    retry=retry,
    metrics=MetricsCollector(max_samples=5000),
    middleware=[LogMiddleware()],
    max_retries=5,
    max_concurrent_requests=50,
    total_timeout=30.0,
)
```

The same options are available declaratively through `RouterConfig`, built from
a dict or environment variables, and assembled with `RouterFactory`:

```python
import os
from llmrouterx.config import RouterConfig
from llmrouterx.router.factory import RouterFactory

config = RouterConfig.from_env()  # reads LLMROUTER_* environment variables
# or
config = RouterConfig(providers=[...], max_retries=5, total_timeout=30.0)

router = RouterFactory.build(config)
```

Supported environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLMROUTER_TIMEOUT` | `60` | Per-request client timeout (seconds) |
| `LLMROUTER_MAX_RETRIES` | `3` | Number of retries per operation (after the initial call) |
| `LLMROUTER_MAX_CONCURRENT` | `100` | Concurrent requests per API key |
| `LLMROUTER_MAX_CONCURRENT_REQUESTS` | unset | Global concurrency cap |
| `LLMROUTER_TOTAL_TIMEOUT` | unset | Hard deadline for the whole operation |
| `LLMROUTER_CIRCUIT_BREAKER` | `true` | Enable per-key circuit breakers |
| `LLMROUTER_CB_THRESHOLD` | `5` | Failures before a key opens |
| `LLMROUTER_CB_RESET_TIMEOUT` | `30` | Cooldown before a key is retried |

### Gateway

Run the OpenAI-compatible HTTP server with auth and observability. Install the
`server` extra, then configure:

```bash
pip install llmrouterx[server]

export LLMROUTER_ADMIN_TOKEN=admin-secret       # protects /dashboard, /metrics
export LLMROUTER_API_KEYS=sk-openai-1,gsk-groq-1  # protects /v1/*
export LLMROUTER_DOCS=0                         # hide /docs, /redoc

llmrouterx serve --config router.json --no-docs --port 8000
```

`LLMROUTER_API_KEYS` is comma-separated; each value is accepted as a bearer
token on `/v1/models` and `/v1/chat/completions`. Omitting these vars leaves
the gateway open (useful for local dev only). See [docs/SERVER.md](docs/SERVER.md).

### Loading Config from a JSON File

Configs can be written as JSON and loaded with `RouterConfig.from_file` /
`from_dict`. Keys are resolved automatically (see below).

```python
from llmrouterx.config import RouterConfig
from llmrouterx.router.factory import RouterFactory

config = RouterConfig.from_file("router.json")  # resolves keys automatically
router = RouterFactory.build(config)
```

```json
{
  "providers": [
    {
      "name": "openai",
      "clients": [
        { "client": "openai", "api_key_env": "OPENAI_API_KEY", "default_model": "gpt-4o" }
      ]
    }
  ],
  "max_retries": 3,
  "total_timeout": 30.0
}
```

### Loading API Keys

A client's key can come from a literal value, an environment variable, or a
file (secrets). Precedence is `api_key` > `api_key_env` > `api_key_file`:

```json
{ "client": "openai", "api_key_env": "OPENAI_API_KEY" }
{ "client": "openai", "api_key_file": "/run/secrets/openai-key" }
{ "client": "openai", "api_key": "sk-plaintext-..." }
```

`from_file`/`from_dict` resolve these to the literal key immediately. If no
source is present or the env var/file is missing, a `KeyResolutionError` is
raised. This keeps secrets out of config files and works with container
secrets and CI/CD environments.

---

## 🌐 HTTP Gateway (Server)

The `server` extra provides an OpenAI-compatible HTTP gateway and a CLI to run
it, plus declarative config and key loading:

```bash
pip install llmrouterx[server]
llmrouterx serve --config router.json --port 8000
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`
(streaming and non-streaming), `GET /metrics`. Errors use the OpenAI-compatible
`{"error": {"message", "type", "code"}}` envelope.

See [docs/SERVER.md](docs/SERVER.md) for the full guide.

---

## 🔭 Observability (Langfuse)

The `langfuse` extra records every routed operation as a Langfuse trace through
the router middleware — no routing code changes needed:

```bash
pip install llmrouterx[langfuse]
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com
```

Tracing activates automatically when those variables are present (or via
`LangfuseMiddleware` in a `RouterConfig.middleware` list). Each
`chat`/`stream`/`embeddings` call becomes a Langfuse `generation` — its own
auto-created trace — with input, output, model, provider, masked key suffix,
retry count, latency, and `ERROR` levels on failures. Tracing is fail-open and
never interrupts routing. The gateway flushes pending traces on clean shutdown.
See [docs/SERVER.md](docs/SERVER.md#4-langfuse-tracing).

`GET /dashboard` serves a zero-config auto-refreshing observability UI, and the
`MetricsCollector` exposes global + per-provider token counters
(`tokens.prompt.total`, `tokens.completion.total`, `tokens.total`). See
[docs/SERVER.md](docs/SERVER.md#5-observability-errors--token-tracking).

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "No healthy providers" | Check the `Failure sequence` in the `NoHealthyClientError` message to see each failing provider/key and its underlying error |
| 401 / 403 from the gateway | Configure an API key / admin token (set `LLMROUTER_API_KEYS` / `LLMROUTER_ADMIN_TOKEN` or pass `create_app(..., admin_token=, api_keys=)`) |
| Requests are slow | Use `LeastBusyScheduler`, check metrics for latency outliers |
| Token counters are 0 | Token usage is extracted automatically from provider responses; custom adapters must call `context.set("usage", {...})` (see `BaseProviderAdapter._record_usage`) |
| Same key always used | Check that scheduler is set to `RoundRobin` or `LeastBusy` |
| Errors not retrying | `max_retries=N` allows N retries after the initial call; non-retryable errors (4xx, bad request) are not retried |
| ModuleNotFoundError: llmrouterx | Run `pip install -e .` if developing from source |
| Tests hang | Update `setuptools>=45` and run `pip install -e .` again |

---

## 📚 API Reference

### LLMRouter

```python
class LLMRouter:
    def __init__(
        self,
        composite_router: CompositeRouter,
        *,
        retry: BaseRetry | None = None,
        metrics: MetricsCollector | None = None,
        middleware: list[BaseMiddleware] | None = None,
        max_retries: int = 3,
        circuit_breaker: CircuitBreaker | None = None,
        max_concurrent_requests: int | None = None,
        total_timeout: float | None = None,
    ):
        """Initialize router with composite, retry policy, and middleware."""

    async def chat(self, prompt: str, **kwargs) -> str:
        """Send chat request to best available provider."""

    async def stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream response chunks from best available provider."""

    async def embeddings(self, text: str, **kwargs) -> list:
        """Get embeddings from best available provider."""

    async def responses(self, *args, **kwargs):
        """Call the provider's Responses API, when supported."""

    async def close(self, timeout: float = 10.0) -> None:
        """Wait for in-flight tasks, then cancel and await leftovers."""

    def get_metrics(self) -> dict:
        """Return metrics snapshot (counters, timings, labeled variants)."""
```

### CompositeRouter

```python
class CompositeRouter:
    def __init__(self, providers: list):
        """Initialize with list of ProviderRouters."""

    async def handle(self, op, payload, **kwargs):
        """Try providers in order; fall back to the next on failure."""

    async def health(self, timeout: float = 5.0) -> list:
        """Check every provider with a per-provider timeout."""

    async def is_healthy(self) -> bool:
        """Check if any provider is healthy."""

    @property
    def last_provider(self) -> str | None:
        """Name of the provider that served the last request."""

    @property
    def last_api_key(self) -> str | None:
        """API key that served the last request (masked nowhere, use with care)."""
```

### ProviderRouter

```python
class ProviderRouter:
    def __init__(self, name: str, clients: list, scheduler=None):
        """Initialize provider with name, clients, and scheduler."""

    async def select_client(self) -> ClientNode:
        """Select best client using scheduler."""

    async def is_healthy(self) -> bool:
        """Check if any client is healthy."""

    @property
    def last_api_key(self) -> str | None:
        """API key that served the last request."""
```

### ClientNode

```python
class ClientNode:
    def __init__(
        self,
        api_key: str,
        client: BaseProviderAdapter,
        *,
        streaming: StreamingManager | None = None,
        timeout: float | None = 60.0,
        max_concurrent: int = 100,
        failure_threshold: int | None = None,
        cooldown_seconds: float | None = None,
        circuit_breaker_enabled: bool = True,
        weight: float = 1.0,
        priority: int = 100,
        count_transient_failures: bool = False,
    ):
        """Initialize with identifier and client instance."""

    async def is_healthy(self, force: bool = False) -> bool:
        """Check if client is healthy."""

    async def send(self, op, payload, **kwargs):
        """Dispatch a request, respecting timeouts and concurrency slots."""

    async def stream(self, prompt, **kwargs) -> AsyncGenerator[str, None]:
        """Stream tokens while holding a concurrency slot."""

    @property
    def is_saturated(self) -> bool:
        """True when every concurrency slot is in use."""
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Write tests** for your changes
4. **Run tests**: `pytest tests/ -v`
5. **Commit with clear messages**: `git commit -m "feat: add my feature"`
6. **Push** to your fork: `git push origin feature/my-feature`
7. **Open a PR** with a clear description

### Development Checklist
- [ ] Code follows style guidelines (`ruff check . && ruff format --check .`)
- [ ] Type checks pass (`mypy`)
- [ ] Tests pass (`pytest tests/ -v`)
- [ ] New features include tests
- [ ] Documentation is updated
- [ ] Commit messages are descriptive

### Areas for Contribution
- Cost-aware routing
- Latency-aware routing
- Response caching
- Prometheus metrics export
- Additional provider support
- Performance optimizations
- Documentation and examples

---

## 📄 License

Apache 2.0 License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built with ❤️ for developers managing multiple LLM providers. Special thanks to the open-source community.

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/amar8737/LLMRouter/issues)
- **Discussions**: [GitHub Discussions](https://github.com/amar8737/LLMRouter/discussions)
- **Email**: babuamar455@gmail.com

---

## 🚀 Roadmap

**v0.1** (Current) ✅
- Basic routing and failover
- API key pooling
- Schedulers (round-robin, least-busy, random, weighted, priority)
- Retry logic
- Metrics collection
- Per-key circuit breakers
- Streaming + graceful shutdown
- Total request timeouts
- Retry middleware hook, labeled metrics, structured logging

**v0.2** (Planned)
- Cost-aware routing
- Latency-aware routing
- Response caching
- Prometheus metrics export

**v0.3** (Future)
- Region-aware routing
- Batch request optimization
- Advanced analytics dashboard

---

**Ready to get started?** Check out the [Quick Start](#-quick-start-5-minutes) section or explore the [examples](#-usage-examples) above!