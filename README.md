# LLMRouter 🚀

**Intelligent routing, load balancing, and failover for LLM providers**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
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
- 🔌 **Extensible Design** — Custom schedulers, retry policies, middleware
- 🔍 **Health Checks** — Monitor provider availability in real-time
- 📡 **Streaming Support** — Stream responses from any provider
- 🛡️ **Error Handling** — Graceful degradation with meaningful exceptions

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
from llmrouterx.client import ClientNode
from llmrouterx.providers import ProviderRouter, CompositeRouter


async def main():
    # Create an OpenAI client
    client = AsyncOpenAI(api_key="sk-your-key-here")

    # Wrap it in a ClientNode (tracks identity and health)
    node = ClientNode("key-1", client)

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

### Example 2: Multiple API Keys (Rate Limit Protection)

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
from llmrouterx.client import ClientNode
from llmrouterx.providers import ProviderRouter, CompositeRouter
from llmrouterx.scheduler import LeastBusyScheduler


async def main():
    # Primary: OpenAI
    openai_node = ClientNode("sk-...", AsyncOpenAI(api_key="sk-..."))
    openai_provider = ProviderRouter("openai", [openai_node], scheduler=LeastBusyScheduler())

    # Fallback: Groq
    groq_node = ClientNode("gsk-...", AsyncGroq(api_key="gsk-..."))
    groq_provider = ProviderRouter("groq", [groq_node], scheduler=LeastBusyScheduler())

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
    print("Total requests:", metrics["counters"].get("total_requests", 0))
    print("Total errors:", metrics["counters"].get("total_errors", 0))
    print(
        "Average latency:",
        sum(metrics["timings"]) / len(metrics["timings"]) if metrics["timings"] else 0,
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
    async def before_request(self, op, payload):
        logger.info(f"→ {op}: {payload}")
        return payload

    async def after_response(self, op, payload, response):
        logger.info(f"← {op}: response received")
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

## 🔑 Core Concepts

### ClientNode
Represents a single LLM client (e.g., one OpenAI API key). Tracks health, active requests, and metadata.

```python
node = ClientNode("identifier", client_instance)
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
metrics = router.metrics.get()

# Counters
print(metrics["counters"]["total_requests"])
print(metrics["counters"]["total_errors"])
print(metrics["counters"]["total_successes"])

# Timings (list of request latencies in seconds)
print(metrics["timings"])  # [0.45, 0.52, 0.38, ...]
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
    async def before_request(self, op, payload):
        # Modify request before sending
        return payload

    async def after_response(self, op, payload, response):
        # Transform response after receiving
        return response


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
pytest tests/test_streaming.py -v
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
from llmrouterx.middleware import BaseMiddleware


class LogMiddleware(BaseMiddleware):
    async def before_request(self, op, payload):
        print(f"Sending: {op}")
        return payload


retry = ExponentialRetry(max_retries=5, base=1.0, factor=2.0)

router = LLMRouter(
    composite,
    retry=retry,
    middleware=[LogMiddleware()],
    # Additional config options as needed
)
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "No healthy providers" | Check if providers are up, verify API keys are valid |
| Requests are slow | Use `LeastBusyScheduler`, check metrics for latency outliers |
| Same key always used | Check that scheduler is set to `RoundRobin` or `LeastBusy` |
| Errors not retrying | Check `should_retry()` logic in retry policy, some errors are permanent |
| ModuleNotFoundError: llmrouterx | Run `pip install -e .` if developing from source |
| Tests hang | Update `setuptools>=45` and run `pip install -e .` again |

---

## 📚 API Reference

### LLMRouter

```python
class LLMRouter:
    def __init__(self, composite, retry=None, middleware=None):
        """Initialize router with composite, retry policy, and middleware."""
    
    async def chat(self, prompt: str, **kwargs) -> str:
        """Send chat request to best available provider."""
    
    async def stream(self, prompt: str, **kwargs):
        """Stream response chunks from best available provider."""
    
    async def embeddings(self, text: str, **kwargs) -> list:
        """Get embeddings from best available provider."""
    
    def get_metrics(self) -> dict:
        """Return metrics (counters and timings)."""
```

### CompositeRouter

```python
class CompositeRouter:
    def __init__(self, providers: list):
        """Initialize with list of ProviderRouters."""
    
    async def select(self) -> ProviderRouter:
        """Select best provider, trying in order."""
    
    async def is_healthy(self) -> bool:
        """Check if any provider is healthy."""
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
```

### ClientNode

```python
class ClientNode:
    def __init__(self, identifier: str, client):
        """Initialize with identifier and client instance."""
    
    async def is_healthy(self) -> bool:
        """Check if client is healthy."""
    
    def increment_active(self) -> None:
        """Increment active request count."""
    
    def decrement_active(self) -> None:
        """Decrement active request count."""
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
- [ ] Code follows style guidelines (use `black` or `autopep8`)
- [ ] Tests pass (`pytest tests/ -v`)
- [ ] New features include tests
- [ ] Documentation is updated
- [ ] Commit messages are descriptive

### Areas for Contribution
- ✅ Cost-aware routing
- ✅ Latency-aware routing  
- ✅ Response caching
- ✅ Prometheus metrics export
- ✅ Additional provider support
- ✅ Performance optimizations
- ✅ Documentation and examples

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

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