# LLMRouter

**Intelligent routing, load balancing, and failover for LLM providers.**

LLMRouter is a Python package that acts as a routing layer between your application and LLM providers (OpenAI, Groq, Together AI, etc.). It handles provider selection, API key rotation, automatic failover, retries, and metrics—so you don't have to.

```python
# Instead of this:
client = OpenAI(api_key="sk-...")
response = client.chat.completions.create(model="gpt-4", messages=[...])

# Do this:
router = LLMRouter(providers=[openai_provider, groq_provider])
response = await router.chat(prompt="Hello!")
# Router automatically picks the best provider and key, retries on failure, tracks metrics
```

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Streaming](docs/STREAMING.md)
- [API Reference](#api-reference)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Advanced Patterns](#advanced-patterns)
- [Testing](#testing)
- [Troubleshooting]

---

## Features

✨ **Core Features (v0.1)**
- 🔄 **Automatic provider failover** — Switch to backup provider if primary fails
- 🔑 **API key pooling** — Rotate across multiple keys to avoid rate limits
- ⏱️ **Intelligent scheduling** — Pick the best client (least busy, round-robin, etc.)
- 🔁 **Automatic retries** — Exponential backoff with jitter
- 🚦 **Per-key circuit breakers** — Isolate a failing key
- ⏳ **Total request timeouts** — Hard deadline across retries and failover
- 📊 **Metrics collection** — Track requests, errors, latency, throughput
- 🏷️ **Labeled metrics** — Per-provider counters and timings
- 📝 **Structured logging** — JSON formatter included
- 🔌 **Extensible design** — Custom schedulers, retry policies, middleware
- 📡 **Streaming support** — Async and synchronous streaming with graceful shutdown

🚀 **Planned Features (v0.2+)**
- 📈 Cost-aware routing (pick cheapest provider)
- ⚡ Latency-aware routing (pick fastest provider)
- 🌍 Region-aware routing
- 💾 Response caching
- 🔐 Authentication gateway
- 📉 Prometheus metrics export

---

## Installation

### From PyPI
```bash
pip install llmrouterx
```

### From source (development)
```bash
git clone https://github.com/amar8737/LLMRouter.git
cd LLMRouter
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### Requirements
- Python 3.10+ (tested on 3.10, 3.11, 3.12)
- No external LLM SDKs required (you provide the client)

---

## Quick Start

### 1. Install and setup
```bash
pip install -e .
python3 tests/run_tests.py  # Run smoke test
```

### 2. Create a minimal example
```python
import asyncio
from llmrouterx import LLMRouter
from llmrouterx.client import ClientNode
from llmrouterx.providers import ProviderRouter, CompositeRouter, StubClient
from llmrouterx.scheduler import RoundRobinScheduler


async def main():
    # Create a stub client for testing
    client = StubClient("stub")
    node = ClientNode("k1", client)
    provider = ProviderRouter("stub", [node], scheduler=RoundRobinScheduler())
    composite = CompositeRouter([provider])
    router = LLMRouter(composite)

    resp = await router.chat(prompt="Hello from enhanced README")
    print(resp)


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Usage Examples

See `docs/QUICKSTART.md` for copy-paste examples including multiple providers, rate-limit protection, and middleware usage.

---

## Contributing

Please follow standard GitHub workflows: fork, create a branch, implement changes, add tests, and open a PR. See `docs/` for guides and the [README](README.md) for the roadmap.
