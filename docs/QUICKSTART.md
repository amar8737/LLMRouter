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
5. ✅ Read full [README.md](README.md) for more details
