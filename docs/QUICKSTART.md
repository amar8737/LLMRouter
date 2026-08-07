# LLMRouter Quick Reference

Copy-paste snippets for common scenarios.

---

## Setup (5 minutes)

```bash
# Install
git clone https://github.com/amar8737/LLMRouter.git
cd LLMRouter
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Test it works
PYTHONPATH=. python3 tests/run_tests.py
```

---

## 1. Single OpenAI Key

```python
import asyncio
from openai import AsyncOpenAI
from llmrouterx import LLMRouter
from llmrouterx.client import ClientNode
from llmrouterx.providers import ProviderRouter, CompositeRouter


async def main():
    client = AsyncOpenAI(api_key="sk-...")
    node = ClientNode("sk-...", client)
    provider = ProviderRouter("openai", [node])
    composite = CompositeRouter([provider])
    router = LLMRouter(composite)

    response = await router.chat(prompt="Hello!")
    print(response)


asyncio.run(main())
```

---

## 2. Multiple OpenAI Keys (Rate Limit Protection)

```python
import asyncio
from openai import AsyncOpenAI
from llmrouterx import LLMRouter
from llmrouterx.client import ClientNode
from llmrouterx.providers import ProviderRouter, CompositeRouter
from llmrouterx.scheduler import LeastBusyScheduler


async def main():
    keys = ["sk-key1", "sk-key2", "sk-key3"]
    nodes = [ClientNode(key, AsyncOpenAI(api_key=key)) for key in keys]

    provider = ProviderRouter(
        "openai",
        nodes,
        scheduler=LeastBusyScheduler(),  # Rotate across keys
    )
    composite = CompositeRouter([provider])
    router = LLMRouter(composite)

    response = await router.chat(prompt="Hello!")
    print(response)


asyncio.run(main())
```

---

## 3. OpenAI + Groq Fallback

```python
import asyncio
from openai import AsyncOpenAI
from groq import AsyncGroq
from llmrouterx import LLMRouter
from llmrouterx.client import ClientNode
from llmrouterx.providers import ProviderRouter, CompositeRouter
from llmrouterx.scheduler import LeastBusyScheduler


async def main():
    # OpenAI (primary)
    openai_node = ClientNode("sk-...", AsyncOpenAI(api_key="sk-..."))
    openai_provider = ProviderRouter("openai", [openai_node], scheduler=LeastBusyScheduler())

    # Groq (fallback)
    groq_node = ClientNode("gsk-...", AsyncGroq(api_key="gsk-..."))
    groq_provider = ProviderRouter("groq", [groq_node], scheduler=LeastBusyScheduler())

    # Try OpenAI first, fall back to Groq
    composite = CompositeRouter([openai_provider, groq_provider])
    router = LLMRouter(composite)

    response = await router.chat(prompt="Hello!")
    print(response)


asyncio.run(main())
```

---

## 4. Concurrent Requests

```python
import asyncio
from llmrouterx import LLMRouter


async def main():
    router = LLMRouter(composite)

    # Make 10 concurrent requests
    prompts = [f"Request {i}" for i in range(10)]
    responses = await asyncio.gather(*[router.chat(prompt=p) for p in prompts])

    print(f"Got {len(responses)} responses")


asyncio.run(main())
```

---

## 5. Custom Scheduling (Priority)

```python
from llmrouterx.scheduler import BaseScheduler


class PriorityScheduler(BaseScheduler):
    async def select(self, provider_router):
        candidates = [c for c in provider_router.clients if await c.is_healthy()]
        if not candidates:
            return None
        candidates.sort(key=lambda c: getattr(c, "priority", 0), reverse=True)
        return candidates[0]


# Use it
premium_node = ClientNode("premium-key", client)
premium_node.priority = 100

standard_node = ClientNode("standard-key", client)
standard_node.priority = 10

provider = ProviderRouter(
    "prioritized", [premium_node, standard_node], scheduler=PriorityScheduler()
)
```

---

## 6. Logging Middleware

```python
import logging
from llmrouterx.middleware import BaseMiddleware
from llmrouterx import LLMRouter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseMiddleware):
    async def before_request(self, operation, payload, context):
        logger.info(f"→ {operation}: {payload}")
        return payload

    async def after_response(self, operation, payload, response, context):
        logger.info(f"← {operation}: {response}")
        return response


router = LLMRouter(composite, middleware=[LoggingMiddleware()])
```

---

## 7. Retry Policy

```python
from llmrouterx.retry import ExponentialRetry
from llmrouterx import LLMRouter

# Retry with exponential backoff
retry = ExponentialRetry(
    max_retries=5,  # Try 5 times
    base=0.5,  # Start with 0.5 second wait
    factor=2.0,  # Double each time
    max_backoff=30.0,  # Cap at 30 seconds
)

router = LLMRouter(composite, retry=retry)
```

---

## 8. View Metrics

```python
import asyncio
from llmrouterx import LLMRouter


async def main():
    router = LLMRouter(composite)

    # Make some requests
    await router.chat(prompt="Request 1")
    await router.chat(prompt="Request 2")

    # View metrics
    metrics = router.metrics.get()
    print("Counters:", metrics["counters"])
    print("Timings:", metrics["timings"])


asyncio.run(main())
```

---

## 9. Error Handling

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

## 10. All Schedulers

```python
from llmrouterx.scheduler import (
    LeastBusyScheduler,  # Pick client with fewest active requests
    RoundRobinScheduler,  # Rotate through clients
    RandomScheduler,  # Pick random client
    WeightedScheduler,  # Pick by weight (node.weight)
    PriorityScheduler,  # Pick by priority (node.priority)
)

# Use it
provider = ProviderRouter(
    "name",
    clients,
    scheduler=LeastBusyScheduler(),  # or any scheduler above
)
```

---

## 11. Declarative Config + HTTP Gateway

Run an OpenAI-compatible server from a JSON config (no Python needed):

```bash
pip install llmrouterx[server]

# router.json
cat > router.json <<'EOF'
{
  "providers": [
    {
      "name": "openai",
      "clients": [
        { "client": "openai", "api_key_env": "OPENAI_API_KEY", "default_model": "gpt-4o" }
      ]
    }
  ]
}
EOF

llmrouterx serve --config router.json --port 8000
```

```bash
curl -X POST localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}'
```

### Securing the gateway

By default the gateway is open (handy for local dev). Lock it down for real
traffic with bearer-token auth:

- **Admin token** for `/dashboard` and `/metrics`.
- **API keys** for `/v1/*`.

Set them via env (works in multi-worker mode too) or pass flags:

```bash
export LLMROUTER_ADMIN_TOKEN=admin-secret
export LLMROUTER_API_KEYS=sk-openai-1,gsk-groq-1,sk-ant-1

llmrouterx serve --config router.json --no-docs
curl -H "Authorization: Bearer sk-openai-1" \
  -X POST localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}'
```

Or programmatically:

```python
app = create_app(
    config_path="router.json",
    admin_token="admin-secret",
    api_keys=["sk-openai-1", "gsk-groq-1"],
    docs_enabled=False,
)
```

See [SERVER.md](SERVER.md) for the endpoint auth matrix and diagnostic headers.

Or load the config in code:

```python
from llmrouterx.config import RouterConfig
from llmrouterx.router.factory import RouterFactory

config = RouterConfig.from_file("router.json")   # resolves keys automatically
router = RouterFactory.build(config)
```

Keys can be a literal (`api_key`), an environment variable (`api_key_env`), or
a file (`api_key_file`), so secrets stay out of config. See
[SERVER.md](SERVER.md).

---

## 12. Quick-Start Fallback Chain

```python
from llmrouterx import LLMRouter

router = LLMRouter.from_cascade([
    "openai:sk-...",
    "anthropic:sk-ant-...",
    "groq:gsk-...",
])
```

Each `"provider:api_key"` becomes a fallback provider in order. The default
client is `AsyncOpenAI` (OpenAI-compatible providers) / `AsyncAnthropic`
(`anthropic`); pass `client_factory(provider, api_key)` to customize.

---

## 13. Langfuse Tracing

```bash
pip install llmrouterx[langfuse]
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com
```

Tracing is enabled automatically when those variables are present — no code
change. Each request is recorded as a Langfuse generation.

```python
import asyncio
from llmrouterx import LLMRouter


async def main():
    # No extra setup needed; the env vars above activate tracing.
    response = await router.chat(prompt="Hello, trace me!")


asyncio.run(main())
```

See [SERVER.md](SERVER.md#4-langfuse-tracing).

---

## Common Patterns

### Stream responses

```python
async for chunk in router.stream(prompt="Tell me a story"):
    print(chunk, end="")
```

### Check if provider is healthy

```python
for provider in composite.providers:
    health = await provider.is_healthy()
    print(f"{provider.name}: {'✓' if health else '✗'}")
```

### Get embeddings

```python
embeddings = await router.embeddings(text="Hello world")
```

### Track tokens from provider usage

Tokens are recorded automatically: every adapter extracts `usage` from the
provider response and the router records it in the global/per-provider token
counters (`tokens.prompt.total`, `tokens.completion.total`,
`tokens.total{provider="..."}`). You can also record manually:

```python
from llmrouterx.metrics import MetricsCollector

metrics = MetricsCollector()
metrics.track_tokens("openai", prompt_tokens=120, completion_tokens=45)
snapshot = metrics.snapshot()["counters"]
snapshot["tokens.prompt.total"]          # global prompt tokens
snapshot["tokens.total{provider=openai}"]  # per-provider total
```

### Handle multi-provider failures

```python
from llmrouterx.exceptions import NoHealthyClientError

try:
    response = await router.chat(prompt="Hello")
except NoHealthyClientError as exc:
    print(exc)          # message includes the "Failure sequence" block
    for provider, api_key, error in exc.errors:
        print(provider, api_key, error)
```

### Custom provider response handling

```python
class MyMiddleware(BaseMiddleware):
    async def after_response(self, operation, payload, response, context):
        # Transform response
        if "data" in response:
            response["transformed"] = response["data"]
        return response
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No healthy providers" | Check if providers are up, verify API keys |
| Requests are slow | Use `LeastBusyScheduler`, check latency in metrics |
| Same key always used | Check scheduler, should be RoundRobin/LeastBusy |
| Errors not retrying | Check `should_retry()` logic, might be permanent error |
| High memory usage | Check if metrics timings list is growing unbounded |

---

## Tests

```bash
# Smoke test (quick)
PYTHONPATH=. python3 tests/run_tests.py

# Unit tests
pytest tests/ -v

# Your own test
# Put this in test_my_router.py:
import pytest
from llmrouterx import LLMRouter

@pytest.mark.asyncio
async def test_chat():
    router = LLMRouter(composite)
    response = await router.chat(prompt="Hello")
    assert response is not None
```

---

## Next Steps

1. ✅ Copy one of the snippets above
2. ✅ Replace API keys with real ones
3. ✅ Run it: `python3 your_script.py`
4. ✅ Check metrics: `router.metrics.get()`
5. ✅ Run the HTTP gateway: `llmrouterx serve --config router.json`
6. ✅ Enable tracing: set `LANGFUSE_*` env vars
7. ✅ Read full [README.md](README.md) for more details
