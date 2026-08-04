# Streaming Approaches: Side-by-Side Comparison

This document compares three streaming approaches you can use with LLMRouter:

- Async generator streaming — best for FastAPI and async applications
- Synchronous (blocking) streaming — best for CLI and scripts
- Callback-based streaming — best for GUI apps (Tkinter, PyQt)

The examples below assume the `StreamingLLMRouter` implemented in `llmrouter.router.streaming`.

---

## Problem

Original LLMRouter:

- ✅ Works with async/await
- ❌ No streaming (waits for entire response)
- ❌ Can't use in blocking code
- ❌ No real-time output

```py
# Old way - blocks until response is complete
response = await router.handle_request("Explain AI")
print(response)  # Nothing printed for several seconds
```


---

## Solution 1: Async Generator Streaming

Best for: FastAPI, async applications, modern Python 3.8+

### Before (No Streaming)

```py
import asyncio
from llmrouter import LLMRouter

async def main():
    router = LLMRouter(...)
    # Blocks until complete response
    response = await router.handle_request("Explain AI in 100 words")
    print(response)

asyncio.run(main())
```

### After (Async Streaming)

```py
import asyncio
from llmrouter.router.streaming import StreamingLLMRouter

async def main():
    router = StreamingLLMRouter(composite)
    async for token in router.stream("Explain AI in 100 words"):
        print(token, end="", flush=True)
    print()

asyncio.run(main())
```

Benefit: real-time output, responsive UI.

Key usage:

- `async for token in router.stream(prompt)` — incremental tokens
- `await router.stream_until_complete(prompt)` — gather and return full text


---

## Solution 2: Synchronous Streaming

Best for: CLI tools, scripts, non-async code

### Before (Blocking, No Streaming)

```py
from llmrouter import LLMRouter

def main():
    router = LLMRouter(...)
    response = router.handle_request_sync("Explain AI")
    print(response)

main()
```

### After (Sync Streaming)

```py
from llmrouter.router.streaming import StreamingLLMRouter

def main():
    router = StreamingLLMRouter(composite)
    for token in router.stream_sync("Explain AI"):
        print(token, end="", flush=True)
    print()

main()
```

Note: `stream_sync()` is a wrapper that runs the async stream flow and yields tokens synchronously. For true sub-second incremental streaming with provider-driven chunks you need provider clients that expose streaming APIs.


---

## Solution 3: Callback-Based Streaming

Best for: GUI applications, event handlers, PyQt/Tkinter

### GUI before (blocks)

```py
import tkinter as tk

class AIApp:
    def __init__(self):
        self.window = tk.Tk()
        self.text_widget = tk.Text(self.window)
        self.text_widget.pack()

    def ask_ai(self):
        response = router.handle_request_sync("...")
        self.text_widget.insert(tk.END, response)

app = AIApp()
```

### GUI after (non-blocking with callbacks)

```py
import tkinter as tk
import asyncio

class AIApp:
    def __init__(self):
        self.window = tk.Tk()
        self.text_widget = tk.Text(self.window)
        self.text_widget.pack()
        self.router = StreamingLLMRouter(composite)

    def on_token(self, token):
        self.text_widget.insert(tk.END, token)
        self.text_widget.update()

    def ask_ai(self):
        asyncio.run(self._stream_response())

    async def _stream_response(self):
        async for token in self.router.stream("Explain AI", on_chunk=self.on_token):
            pass
```

Result: GUI stays responsive, text updates in real-time.


---

## Real-World Example: Chat CLI (sync streaming)

```py
#!/usr/bin/env python
from llmrouter.router.streaming import StreamingLLMRouter

router = StreamingLLMRouter(composite)

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    print("Bot: ", end="", flush=True)
    # Streams tokens
    for token in router.stream_sync(user_input):
        print(token, end="", flush=True)
    print()
```


---

## Real-World Example: FastAPI SSE endpoint (async streaming)

```py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llmrouter.router.streaming import StreamingLLMRouter
import json

app = FastAPI()
router = StreamingLLMRouter(composite)

@app.post("/chat/stream")
async def chat_stream(prompt: str):
    async def generate():
        async for token in router.stream(prompt):
            yield f"data: {json.dumps({'token': token})}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

Frontend can consume SSE and append tokens as they arrive.


---

## Comparison Table

| Feature | Async Generator | Sync Streaming | Callback |
|---|---:|---:|---:|
| Use Case | FastAPI, async apps | CLI, scripts | GUI, event handlers |
| Syntax | `async for token in stream()` | `for token in stream_sync()` | `on_chunk=callback` |
| Blocking | No | Yes (blocking wrapper) | No |
| Threading | Single async loop | Uses `asyncio.run` internally | Single async loop |
| Complexity | Medium | Low | Medium |
| Performance | High (if provider streams) | Medium | High |
| GUI Compatibility | No | No | Yes |


---

## Token Streaming Performance (notes)

- Provider-driven streaming is the most efficient and lowest-latency approach — work with provider clients that implement streaming.
- Tokenization fallbacks (word/sentence splitting) are useful when provider does not stream, but they provide a simulated streaming UI rather than true low-latency chunks.


---

## Integration Patterns

### Gradual migration

1. Keep `LLMRouter` for non-streaming code path.
2. Add `StreamingLLMRouter` in parallel and switch critical paths.
3. Migrate handlers to streaming gradually.

### Wrapper

Provide a small wrapper API that hides streaming vs non-streaming:

```py
async def ask_ai(prompt: str, stream: bool = True):
    if stream:
        resp = ""
        async for token in router.stream(prompt):
            print(token, end="", flush=True)
            resp += token
        return resp
    else:
        return await router.handle_request(prompt)
```


---

## Common mistakes & fixes

- Forgetting `flush=True` when printing tokens
- Storing all tokens in memory (unbounded lists)
- Not catching exceptions during streaming (partial output plus graceful failure is better)


---

## Testing streaming

See `tests/test_streaming.py` for unit tests that verify async generator, callback and sync wrappers.


---

## Next steps

- Wire true provider-level streaming (clients provide `stream` or an async iterator). See `llmrouter.router.streaming` for the current fallback implementation.
- Add examples for major providers showing how to hook provider streaming APIs.


---

File created from a side-by-side comparison provided by the project owner. If you want this mirrored into `README_ENHANCED.md` or linked from the top-level `README.md`, I can update those next.
