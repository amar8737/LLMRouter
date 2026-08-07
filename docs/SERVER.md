# LLMRouter Server & Gateway

The `server` extra ships an OpenAI-compatible HTTP gateway plus a CLI that
wires up a router from a declarative config file. It is the easiest way to run
LLMRouter in front of your application without writing any Python.

```bash
pip install llmrouterx[server]
```

---

## 1. HTTP Gateway

Start the gateway with the CLI (Uvicorn):

```bash
llmrouterx serve --host 0.0.0.0 --port 8000
# options:
#   --config PATH     path to a JSON router config (recommended)
#   --workers N       number of worker processes
#   --reload          auto-reload on code changes (development)
#   --log-level info  debug | info | warning | error | critical
```

`serve` without `--config` builds the router from environment variables (see
the README configuration table). For production, pass a config file that
declares providers and keys.

### Endpoints

| Method | Path                    | Description                                        | Auth          |
|--------|-------------------------|----------------------------------------------------|---------------|
| GET    | `/health`               | Provider health + healthy/total counts + uptime    | none          |
| GET    | `/v1/models`            | List configured providers as models                | API key       |
| POST   | `/v1/chat/completions`  | Chat completions (streaming and non-streaming)     | API key       |
| GET    | `/metrics`              | In-memory router metrics snapshot                  | admin token   |
| GET    | `/dashboard`            | Zero-config auto-refreshing observability UI       | admin token   |

Open `/dashboard` in a browser for a live view of health and metrics.

### Authentication

The gateway supports two bearer-token tiers, both optional (the gateway is open
until a token is configured):

- **Admin token** (`admin_token=` arg, or `LLMROUTER_ADMIN_TOKEN` env):
  required for the observability endpoints `/dashboard` and `/metrics`.
- **API keys** (`api_keys=` arg, or `LLMROUTER_API_KEYS` env, comma-separated):
  required for the LLM endpoints `/v1/models` and `/v1/chat/completions`.

The CLI exposes `--admin-token`, `--api-key` (repeatable), and `--no-docs`.
Interactive docs (`/docs`) are served unless disabled.

```bash
export LLMROUTER_ADMIN_TOKEN=super-secret-admin
export LLMROUTER_API_KEYS=sk-openai-1,gsk-groq-1
llmrouterx serve --config router.json
```

In code, or to embed in an existing app:

```python
from llmrouterx.server.app import create_app

app = create_app(
    router=my_router,
    admin_token="super-secret-admin",
    api_keys=["sk-openai-1", "gsk-groq-1"],
    docs_enabled=False,
)
```
`/metrics` includes dedicated token counters (`tokens.prompt.total`,
`tokens.completion.total`, and a per-provider `tokens.total`) when token
tracking is wired in.

Examples:

```bash
# Health
curl localhost:8000/health
# => {"status":"ok","providers":{"openai":true},"healthy_count":1,
#     "total_providers":1,"uptime_seconds":0.12}

# Chat (non-streaming)
curl -X POST localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}]}'

# Chat (streaming)
curl -N -X POST localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

### Embedding the gateway in an existing app

The gateway is a normal FastAPI app via the `create_app` factory:

```python
from fastapi import FastAPI
from llmrouterx.server.app import create_app

app = create_app(config_path="router.json")   # build from file
# or
app = create_app(router=my_existing_router)    # embed an existing router
```

---

## 2. Declarative Config File

A router config is a JSON object matching the `RouterConfig` fields:

```json
{
  "providers": [
    {
      "name": "openai",
      "clients": [
        { "client": "openai", "api_key_env": "OPENAI_API_KEY", "default_model": "gpt-4o" },
        { "client": "openai", "api_key_file": "/run/secrets/openai-2", "default_model": "gpt-4o" }
      ]
    },
    {
      "name": "groq",
      "clients": [
        { "client": "groq", "api_key": "gsk-plaintext-...", "default_model": "llama-3.1-70b" }
      ]
    }
  ],
  "timeout": 60,
  "max_retries": 3,
  "max_concurrent_per_key": 100,
  "total_timeout": 30.0,
  "enable_circuit_breaker": true,
  "circuit_breaker_threshold": 5,
  "circuit_breaker_reset_timeout": 30
}
```

Load it in code with `RouterConfig.from_file` / `from_dict`:

```python
from llmrouterx.config import RouterConfig
from llmrouterx.router.factory import RouterFactory

config = RouterConfig.from_file("router.json")   # resolves keys automatically
router = RouterFactory.build(config)
```

---

## 3. Loading API Keys

A client's key can be provided three ways; precedence is
`api_key` > `api_key_env` > `api_key_file`:

| Field           | Meaning                                                     |
|-----------------|-------------------------------------------------------------|
| `api_key`       | The literal key (plaintext)                                 |
| `api_key_env`   | Name of an environment variable holding the key             |
| `api_key_file`  | Path to a file whose trimmed contents are the key (secrets) |

```json
{ "client": "openai", "api_key_env": "OPENAI_API_KEY" }
{ "client": "openai", "api_key_file": "/run/secrets/openai-key" }
{ "client": "openai", "api_key": "sk-plaintext-..." }
```

When a config is loaded with `from_file`/`from_dict`, keys are resolved
immediately: `api_key_env` and `api_key_file` are replaced by the literal value
under `api_key`. If no source is present, or an env var/file is missing, a
`KeyResolutionError` is raised with a clear message.

```python
from llmrouterx.config.secrets import resolve_key, KeyResolutionError

key = resolve_key({"api_key_env": "OPENAI_API_KEY"})   # reads the env var
```

This keeps secrets out of config files and works with container secrets
(Docker/K8s mounted files) and CI/CD environments.

---

## 4. Langfuse Tracing

The `langfuse` extra records every routed operation as a Langfuse trace. It
plugs into the router middleware system, so **no routing code changes** are
needed.

```bash
pip install llmrouterx[langfuse]
```

Set the environment and the router picks up tracing automatically:

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com   # or LANGFUSE_BASE_URL
```

Every `chat` / `stream` / `embeddings` call is recorded as a Langfuse
`generation` (each request becomes its own auto-created trace) with:

- **input** (the prompt or text), **output**, and the **model**
- **metadata**: `request_id`, `provider`, masked `api_key_suffix`, `retry_count`,
  and measured `latency_ms`
- failures logged at `ERROR` level with the exception type/message as
  `status_message`

Tracing is fail-open: if Langfuse errors, the failure is logged and routing is
never affected. The `RouterFactory` auto-enables tracing when the credentials
are present, but you can attach the middleware explicitly:

```python
from llmrouterx.middleware.langfuse_trace import LangfuseMiddleware

config = RouterConfig(...)
config.middleware.append(LangfuseMiddleware())
```

The gateway flushes pending traces on clean shutdown:

```python
# in a FastAPI lifespan shutdown hook
for m in router._middleware:
    if hasattr(m, "flush"):
        m.flush()
```

---

## 5. Observability, Errors & Token Tracking

### Dashboard

`GET /dashboard` serves a zero-config, auto-refreshing (2s) HTML page that
polls `/health` and `/metrics` — open it in a browser for a live view of
provider health, healthy/total counts, and token counters. No extra setup or
dependencies required.

### Token tracking

Provider adapters automatically extract token usage from successful
`chat`/`responses`/`embeddings` responses (via the request context) and feed
it into the metrics collector, so these counters are populated out of the box
on the gateway:

- `tokens.prompt.total` — total prompt tokens across all providers
- `tokens.completion.total` — total completion tokens across all providers
- `tokens.total{provider="..."}` — per-provider total tokens

You can also record usage manually:

```python
from llmrouterx.metrics import MetricsCollector

metrics = MetricsCollector()
metrics.track_tokens("openai", prompt_tokens=120, completion_tokens=45)
metrics.snapshot()["counters"]["tokens.prompt.total"]  # 120
```

### Actionable error reporting

When every provider fails, `chat`/`stream`/`embeddings` raises
`NoHealthyClientError`. Its message includes a `Failure sequence:` block listing
each provider, the failing key, and the underlying error, so you can diagnose
multi-provider outages at a glance:

```python
from llmrouterx.exceptions import NoHealthyClientError

try:
    response = await router.chat(prompt="Hello")
except NoHealthyClientError as exc:
    print(exc)          # includes the Failure sequence block
    print(exc.errors)   # list of (provider, api_key, error) tuples
```

---

## 6. Performance

A few built-in optimizations reduce the *avoidable* overhead between your
application and the provider (the network round-trip to the provider itself
still dominates):

- **Shared HTTP/2 connection pool.** Every provider SDK client created by
  `LLMRouter.from_cascade` shares one process-wide `httpx.AsyncClient` with
  HTTP/2 multiplexing and a large keep-alive pool. Requests to the same host
  reuse a warm connection instead of paying a TCP + TLS handshake each time.
  Degrades gracefully to HTTP/1.1 keep-alive if the optional `h2` package is
  missing.
- **uvloop event loop.** `llmrouterx serve` installs uvloop (Cython/libuv) as
  the asyncio event-loop policy when available, reducing I/O context-switch
  overhead under high concurrency. Falls back to the standard loop on unsupported
  platforms (e.g. Windows) automatically.

Both ship in the `server` extra:

```bash
pip install llmrouterx[server]   # brings in httpx[http2] and uvloop
```

The gateway also relies on FastAPI's built-in Pydantic JSON serialization for
typed responses (the fastest path in current FastAPI), rather than a custom
response class.

---

## 7. Next Steps

- See [QUICKSTART.md](QUICKSTART.md) for programmatic routing examples.
- See [STREAMING.md](STREAMING.md) for streaming behaviour and failover rules.
