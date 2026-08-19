# LLMRouter Quick Reference

Copy-paste snippets for common scenarios.

---

## ⚡ Fastest Start: HTTP Gateway (No Python)

Run an OpenAI-compatible server in seconds:

```bash
pip install llmrouterx[server]

# Option 1: Zero-config (uses env vars)
export OPENAI_API_KEY=sk-...
llmrouterx serve --port 8000

# Option 2: Quick provider flag (no config file)
llmrouterx serve --provider openai:sk-...@gpt-4o --port 8000

# Option 3: Fallback chain (first healthy wins)
llmrouterx serve --fallback openai:sk-1 --fallback groq:gsk-2 --port 8000

# Option 4: Config file (YAML or JSON)
cat > router.yaml <<'EOF'
providers:
  - name: openai
    clients:
      - client: openai
        api_key_env: OPENAI_API_KEY
        default_model: gpt-4o
EOF
llmrouterx serve --config router.yaml --port 8000
```

Test it:
```bash
curl -X POST localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}'
```

---

## 🐍 Python API: Recommended Entry Points

### 1. Fallback Chain (`from_cascade`) — Simplest

```python
from llmrouterx import LLMRouter

router = LLMRouter.from_cascade([
    "openai:sk-...",
    "anthropic:sk-ant-...",
    "groq:gsk-...",
])

response = await router.chat(prompt="Hello!")
```

Each `"provider:api_key"` becomes a fallback provider in order. Keys can be literals or env var names (`OPENAI_API_KEY`). Default models are provider-specific.

### 2. Multiple Providers with Options (`from_providers`) — Most Flexible

```python
from llmrouterx import LLMRouter

router = LLMRouter.from_providers([
    {"provider": "openai", "key_env": "OPENAI_API_KEY", "model": "gpt-4o"},
    {"provider": "groq", "key_env": "GROQ_API_KEY", "model": "llama-3.3-70b-versatile"},
    {"provider": "anthropic", "key": "sk-ant-...", "model": "claude-3-5-sonnet-20241022"},
])

response = await router.chat(prompt="Hello!")
```

Per-provider options:
| Field | Description |
|-------|-------------|
| `provider` | Adapter name: `openai`, `groq`, `anthropic`, `cohere`, `together`, `gemini`, `mistral`, `xai`, `deepseek`, `ollama` |
| `key` | Literal API key |
| `key_env` | Env var name (scans `_1`, `_2`, ... for key rotation) |
| `key_file` | Path to file containing key |
| `model` | Default chat model |
| `embedding_model` | Default embedding model |
| `base_url` | OpenAI-compatible base URL override |
| `scheduler` | Scheduler instance for key rotation |
| `clients` | List of client dicts for multiple keys with per-key options |

### 3. Synchronous Wrapper (`LLMRouterSync`)

```python
from llmrouterx import LLMRouterSync

router = LLMRouterSync.from_cascade(["openai:sk-..."])

# Blocking calls - no async/await needed
response = router.chat("Hello!")
for chunk in router.stream_chunks("Tell me a story"):
    print(chunk, end="")
```

---

## 🔧 Advanced: Low-Level Composition

> **Note**: The patterns below are for advanced use cases. Most users should use `from_cascade` or `from_providers` above.

### Single Provider with Multiple Keys (Rate Limit Protection)

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

### Provider Fallback with Custom Scheduling

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

### Custom Priority Scheduling

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

## 🛠 Common Patterns

### Stream Responses

```python
async for chunk in router.stream(prompt="Tell me a story"):
    print(chunk, end="")
```

### Get Embeddings

```python
embeddings = await router.embeddings(text="Hello world")
```

### Re-rank Documents

```python
results = await router.rerank(
    query="machine learning",
    documents=["doc1...", "doc2...", "doc3..."],
    top_n=2,
)
```

### View Metrics

```python
metrics = router.metrics.get()
print("Counters:", metrics["counters"])
print("Timings:", metrics["timings"])
```

### Error Handling

```python
from llmrouterx.exceptions import NoHealthyClientError

try:
    response = await router.chat(prompt="Hello!")
except NoHealthyClientError as e:
    print(f"All providers down: {e}")
    print(e.errors)  # List of (provider, api_key, error) tuples
except Exception as e:
    print(f"Request failed: {e}")
```

### Logging Middleware

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

### Retry Policy

```python
from llmrouterx.retry import ExponentialRetry
from llmrouterx import LLMRouter

retry = ExponentialRetry(
    max_retries=5,
    base=0.5,
    factor=2.0,
    max_backoff=30.0,
)

router = LLMRouter(composite, retry=retry)
```

### Track Tokens

```python
from llmrouterx.metrics import MetricsCollector

metrics = MetricsCollector()
metrics.track_tokens("openai", prompt_tokens=120, completion_tokens=45)
snapshot = metrics.snapshot()["counters"]
snapshot["tokens.prompt.total"]  # global prompt tokens
snapshot["tokens.total{provider=openai}"]  # per-provider total
```

---

## 🔐 Securing the Gateway

```bash
# Admin token for /dashboard and /metrics
export LLMROUTER_ADMIN_TOKEN=admin-secret

# API keys for /v1/* endpoints
export LLMROUTER_API_KEYS=sk-openai-1,gsk-groq-1,sk-ant-1

llmrouterx serve --config router.yaml --no-docs
```

```bash
curl -H "Authorization: Bearer sk-openai-1" \
  -X POST localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}'
```

Or programmatically:
```python
from llmrouterx.server.app import create_app

app = create_app(
    config_path="router.yaml",
    admin_token="admin-secret",
    api_keys=["sk-openai-1", "gsk-groq-1"],
    docs_enabled=False,
)
```

---

## 📊 Observability

### Dashboard
Open `http://localhost:8000/dashboard` in a browser for a live auto-refreshing view of provider health, latency, and token usage.

### Metrics
```bash
# JSON format
curl localhost:8000/metrics

# Prometheus format
curl -H "Accept: text/plain; version=0.0.4" localhost:8000/metrics
# or
curl localhost:8000/metrics/prometheus
```

### Health Checks
```bash
# Liveness (process alive)
curl localhost:8000/live

# Readiness (router + providers healthy)
curl localhost:8000/ready
```

### Langfuse Tracing
```bash
pip install llmrouterx[langfuse]
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com
```
Tracing is enabled automatically — no code changes needed.

---

## 🧪 Tests

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
    router = LLMRouter.from_cascade(["openai:sk-..."])
    response = await router.chat(prompt="Hello")
    assert response is not None
```

---

## 📚 Next Steps

1. ✅ Copy a snippet above
2. ✅ Replace API keys with real ones
3. ✅ Run it: `python3 your_script.py`
4. ✅ Check metrics: `router.metrics.get()`
5. ✅ Run the HTTP gateway: `llmrouterx serve --config router.yaml`
6. ✅ Enable tracing: set `LANGFUSE_*` env vars
7. ✅ Read [SERVER.md](SERVER.md) for gateway details
8. ✅ Read [MIGRATION.md](MIGRATION.md) for low-level → high-level migration