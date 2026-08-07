# Streaming: Async, Synchronous, and Callback Approaches

LLMRouter supports streaming in three styles. Pick the one that matches your
application:

- **Async generator streaming** — best for FastAPI and async applications
- **Synchronous (blocking) streaming** — best for CLI and scripts
- **Callback-based streaming** — best for GUI apps (Tkinter, PyQt)

Two layers exist:

- `LLMRouter.stream()` — the high-level router API. It picks a provider, holds
  a concurrency slot, and streams tokens. Failover happens *before* the first
  token; retries are intentionally **not** applied to streams (switching
  providers mid-stream would duplicate output).
- `StreamingManager` (`llmrouterx.streaming`) — the lower-level engine used by
  the router. It wraps a single provider adapter and adds sync wrappers,
  callbacks, tokenization, and cancellation.

---

## 1. Async Generator Streaming

Best for: FastAPI, async applications.

### Using the router

```python
import asyncio
from llmrouterx import LLMRouter

router = LLMRouter(composite)


async def main():
    async for token in router.stream(prompt="Explain AI in 100 words"):
        print(token, end="", flush=True)
    print()


asyncio.run(main())
```

### Using StreamingManager directly

```python
import asyncio
from llmrouterx.streaming import StreamingManager
from llmrouterx.adapters import AdapterFactory

adapter = AdapterFactory.create(
    provider="openai",
    client=client,  # e.g. AsyncOpenAI(...)
    default_model="gpt-4",
)
manager = StreamingManager(adapter)


async def main():
    text = await manager.stream_to_text("Explain AI", on_chunk=lambda t: print(t, end=""))
    print()
    print(text)


asyncio.run(main())
```

---

## 2. Synchronous Streaming

Best for: CLI tools, scripts, and non-async code.

> **Note:** Each active synchronous stream runs on its own daemon thread with a
> private event loop. This is convenient for interactive use but does not
> scale to many concurrent streams — prefer `StreamingManager.stream()` (async)
> in servers and batch jobs.

```python
from llmrouterx.streaming import StreamingManager

manager = StreamingManager(adapter)

for token in manager.stream_sync("Explain AI"):
    print(token, end="", flush=True)
print()
```

To cancel a long-running stream from another thread, pass a `threading.Event`:

```python
import threading
from llmrouterx.streaming import AsyncStreamEngine, SyncStreamEngine

engine = SyncStreamEngine(AsyncStreamEngine(adapter))

stop_event = threading.Event()


def run_stream():
    for token in engine.stream("Explain AI", stop_event=stop_event):
        print(token, end="", flush=True)


thread = threading.Thread(target=run_stream)
thread.start()
# ... later, cancel the stream:
stop_event.set()
thread.join()
```

---

## 3. Callback-Based Streaming

Best for: GUI applications and event handlers.

### Tkinter example

```python
import asyncio
import tkinter as tk


class AIApp:
    def __init__(self):
        self.window = tk.Tk()
        self.text_widget = tk.Text(self.window)
        self.text_widget.pack()

    def on_token(self, token):
        self.text_widget.insert(tk.END, token)
        self.text_widget.update()

    def ask_ai(self):
        asyncio.run(self._stream_response())

    async def _stream_response(self):
        async for token in manager.stream("Explain AI", on_chunk=self.on_token):
            pass


app = AIApp()
```

`on_chunk` is also honored by the sync engine, so the same callback works with
`stream_sync(...)`.

---

## 4. Real-World Example: Chat CLI (sync streaming)

```python
from llmrouterx.streaming import StreamingManager

manager = StreamingManager(adapter)

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    print("Bot: ", end="", flush=True)
    for token in manager.stream_sync(user_input):
        print(token, end="", flush=True)
    print()
```

---

## 5. Real-World Example: FastAPI SSE Endpoint (async streaming)

```python
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llmrouterx import LLMRouter

app = FastAPI()
router = LLMRouter(composite)


@app.post("/chat/stream")
async def chat_stream(prompt: str):
    async def generate():
        async for token in router.stream(prompt):
            yield f"data: {json.dumps({'token': token})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

The frontend consumes the SSE stream and appends tokens as they arrive.

---

## 6. Provider Wiring

Provider adapters implement `BaseProviderAdapter.stream(prompt, *, model, **kwargs)`
and yield string tokens. Adapters are constructed with `AdapterFactory.create()`
and attached to a router through `ClientNode` + `ProviderRouter`, or used
directly via `StreamingManager`.

```python
from llmrouterx.adapters import AdapterFactory

adapter = AdapterFactory.create(
    provider="openai",
    client=AsyncOpenAI(api_key="sk-..."),
    default_model="gpt-4",
)
```

```python
from llmrouterx.client import ClientNode
from llmrouterx.providers import ProviderRouter, CompositeRouter
from llmrouterx.streaming import StreamingManager

node = ClientNode(
    "sk-...",
    adapter,
    streaming=StreamingManager(adapter),
)
provider = ProviderRouter("openai", [node])
composite = CompositeRouter([provider])
router = LLMRouter(composite)
```

If an adapter does not expose a real streaming API, tokenization fallbacks
(whitespace / character) simulate streaming — useful for a streaming UI, but
not true low-latency chunks.

---

## 7. Comparison Table

| Feature | Async Generator | Sync Streaming | Callback |
|---|---:|---:|---:|
| Use Case | FastAPI, async apps | CLI, scripts | GUI, event handlers |
| Syntax | `async for token in stream()` | `for token in stream_sync()` | `on_chunk=callback` |
| Blocking | No | Yes (blocking wrapper) | No |
| Threading | Single async loop | One thread per stream | Single async loop |
| Complexity | Medium | Low | Medium |
| Performance | High (if provider streams) | Medium | High |
| GUI Compatibility | No | No | Yes |

---

## 8. Common Mistakes & Fixes

- Forgetting `flush=True` when printing tokens.
- Storing all tokens in memory (use unbounded lists only for short responses).
- Not catching exceptions during streaming — partial output plus graceful
  failure is better than a crash.
- Leaving a `router.stream()` generator open — `close()` it early to release
  the concurrency slot and stop the provider stream.
- Treating the sync engine as thread-safe — one `SyncStreamEngine.stream`
  call per thread.

---

## 9. Testing Streaming

See `tests/test_streaming_shutdown.py` and `tests/test_streaming_components.py`
for tests covering async generators, sync wrappers, callbacks, and graceful
shutdown.

---

## Next Steps

- Wire true provider-level streaming by implementing
  `BaseProviderAdapter.stream()` for each provider SDK you use.
- Use `StreamingManager.tokenize()` and custom tokenizers for simulated
  streaming fallbacks.
