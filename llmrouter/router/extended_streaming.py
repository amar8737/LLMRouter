import asyncio
import inspect
import queue
import threading
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class StreamMode(Enum):
    NONE = "none"
    ASYNC = "async"
    SYNC = "sync"
    CALLBACK = "callback"


@dataclass
class StreamConfig:
    mode: StreamMode = StreamMode.ASYNC
    buffer_size: int = 100
    collect_full: bool = False
    timeout: float = 60.0
    retry_on_incomplete: bool = True


def _default_tokenize(text: str):
    parts = text.split(" ")
    for i, p in enumerate(parts):
        if i + 1 < len(parts):
            yield p + " "
        else:
            yield p


class ExtendedStreamingLLMRouter:
    """Provider-agnostic extended router with async+sync methods and streaming.

    This implementation uses provider clients from the `providers` mapping.
    A provider client may expose an async `request(op, payload, **kwargs)` method
    (as `StubClient` does) or provider-specific APIs. The router will call the
    client's `request` method when present, awaiting if necessary.
    """

    def __init__(self, providers: Dict[str, Any], config: StreamConfig = None):
        self.providers = providers
        self.config = config or StreamConfig()
        self.tokenizer = _default_tokenize

    # ----------------- provider helpers -----------------
    async def _maybe_await(self, value_or_coro):
        if inspect.isawaitable(value_or_coro):
            return await value_or_coro
        return value_or_coro

    def _select_provider(self) -> str:
        return list(self.providers.keys())[0]

    def _extract_text(self, result: Any) -> str:
        if isinstance(result, dict):
            return result.get("response") or result.get("text") or str(result)
        return str(result)

    # ----------------- ASYNC CHAT -----------------
    async def chat(self, prompt: str, model: str = None, **kwargs) -> str:
        provider = self._select_provider()
        return await self._chat_with_provider(provider, prompt, model, **kwargs)

    async def _chat_with_provider(self, provider: str, prompt: str, model: str = None, **kwargs) -> str:
        cfg = self.providers.get(provider, {})
        client = cfg.get("client")

        # prefer a generic request API
        if client and hasattr(client, "request"):
            try:
                res = client.request("chat", {"prompt": prompt}, **kwargs)
                res = await self._maybe_await(res)
                return self._extract_text(res)
            except Exception:
                # fallback to other provider-specific flows
                pass

        # provider-specific attempt: openai-like
        if client and hasattr(client, "chat"):
            # Try calling an async chat completion if available
            try:
                func = getattr(client.chat, "completions", None)
                if func and hasattr(func, "create"):
                    out = func.create(model=model or cfg.get("default_model"), messages=[{"role": "user", "content": prompt}], stream=False)
                    out = await self._maybe_await(out)
                    return self._extract_text(out)
            except Exception:
                pass

        raise NotImplementedError("No supported chat method found for provider: %s" % provider)

    # ----------------- ASYNC STREAMING -----------------
    async def chat_stream(self, prompt: str, model: str = None, on_chunk: Optional[Callable] = None, **kwargs) -> AsyncGenerator[str, None]:
        provider = self._select_provider()
        # attempt provider streaming paths
        cfg = self.providers.get(provider, {})
        client = cfg.get("client")

        # If client exposes async request streaming (rare), try it
        if client and hasattr(client, "request"):
            # call and then fallback to full response tokenization
            try:
                res = client.request("chat", {"prompt": prompt}, stream=True, **kwargs)
                res = await self._maybe_await(res)
                # If res is an async iterator
                if hasattr(res, "__aiter__"):
                    async for chunk in res:
                        text = self._extract_text(chunk)
                        for tok in self.tokenizer(text):
                            if on_chunk:
                                maybe = on_chunk(tok)
                                if inspect.isawaitable(maybe):
                                    await maybe
                            yield tok
                    return
            except Exception:
                pass

        # Fallback: get full response then stream tokens
        full = await self._chat_with_provider(provider, prompt, model, **kwargs)
        for tok in self.tokenizer(full):
            if on_chunk:
                maybe = on_chunk(tok)
                if inspect.isawaitable(maybe):
                    await maybe
            yield tok

    async def chat_stream_complete(self, prompt: str, model: str = None, print_output: bool = False, **kwargs) -> str:
        out = ""
        async for t in self.chat_stream(prompt, model, **kwargs):
            out += t
            if print_output:
                print(t, end="", flush=True)
        if print_output:
            print()
        return out

    # ----------------- SYNC WRAPPERS -----------------
    def chat_sync(self, prompt: str, model: str = None, **kwargs) -> str:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.chat(prompt, model, **kwargs))
        finally:
            loop.close()

    def chat_stream_sync(self, prompt: str, model: str = None, on_chunk: Optional[Callable] = None, stop_event: Optional[threading.Event] = None, **kwargs):
        q: "queue.Queue[Tuple[str, Optional[str]]]" = queue.Queue()
        err = [None]

        def producer():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def run():
                try:
                    async for tok in self.chat_stream(prompt, model, on_chunk=on_chunk, **kwargs):
                        # allow cooperative cancellation via stop_event
                        if stop_event and stop_event.is_set():
                            break
                        q.put(("token", tok))
                    q.put(("done", None))
                except Exception as e:
                    err[0] = e
                    q.put(("error", e))

            task = loop.create_task(run())

            # watcher to cancel the task from another thread if stop_event is set
            def watcher():
                if not stop_event:
                    return
                stop_event.wait()
                if not task.done():
                    loop.call_soon_threadsafe(task.cancel)

            watch_thread = threading.Thread(target=watcher, daemon=True)
            watch_thread.start()

            try:
                loop.run_until_complete(task)
            finally:
                loop.close()

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        try:
            while True:
                typ, data = q.get()
                if typ == "error":
                    raise err[0]
                if typ == "done":
                    break
                if typ == "token":
                    yield data
        finally:
            thread.join(timeout=5)

    def chat_stream_sync_complete(self, prompt: str, model: str = None, print_output: bool = False, **kwargs) -> str:
        out = ""
        for tok in self.chat_stream_sync(prompt, model, **kwargs):
            out += tok
            if print_output:
                print(tok, end="", flush=True)
        if print_output:
            print()
        return out

    # ----------------- EMBEDDINGS -----------------
    async def embeddings(self, text: str, model: str = None, **kwargs) -> List[float]:
        provider = self._select_provider()
        cfg = self.providers.get(provider, {})
        client = cfg.get("client")

        if client and hasattr(client, "embeddings"):
            try:
                call = client.embeddings.create(model=model or cfg.get("embedding_model"), input=text, **kwargs)
                res = await self._maybe_await(call)
                return res.data[0].embedding
            except Exception:
                pass

        if client and hasattr(client, "request"):
            res = client.request("embeddings", {"text": text}, **kwargs)
            res = await self._maybe_await(res)
            # Expect dict-like
            return res.get("embedding") or res.get("data", [])[0].get("embedding")

        raise NotImplementedError("Embeddings not implemented for provider")

    def embeddings_sync(self, text: str, model: str = None, **kwargs) -> List[float]:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.embeddings(text, model, **kwargs))
        finally:
            loop.close()

    # ----------------- RESPONSES -----------------
    async def responses(self, *args, model: str = None, **kwargs) -> Any:
        provider = self._select_provider()
        return await self._responses_with_provider(provider, args, model, **kwargs)

    async def _responses_with_provider(self, provider: str, args: Tuple, model: str = None, **kwargs) -> Any:
        cfg = self.providers.get(provider, {})
        client = cfg.get("client")
        if client and hasattr(client, "request"):
            res = client.request("responses", {"args": args}, **kwargs)
            res = await self._maybe_await(res)
            return res
        raise NotImplementedError("Responses not implemented for provider")

    async def responses_stream(self, *args, model: str = None, on_chunk: Optional[Callable] = None, **kwargs) -> AsyncGenerator[str, None]:
        provider = self._select_provider()
        # fallback to full response tokenization
        res = await self._responses_with_provider(provider, args, model, **kwargs)
        text = self._extract_text(res)
        for tok in self.tokenizer(text):
            if on_chunk:
                maybe = on_chunk(tok)
                if inspect.isawaitable(maybe):
                    await maybe
            yield tok

    def responses_sync(self, *args, model: str = None, **kwargs) -> Any:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.responses(*args, model=model, **kwargs))
        finally:
            loop.close()

    def responses_stream_sync(self, *args, model: str = None, on_chunk: Optional[Callable] = None, **kwargs):
        stop_event: Optional[threading.Event] = kwargs.pop("stop_event", None)
        q: "queue.Queue[Tuple[str, Optional[str]]]" = queue.Queue()
        err = [None]

        def producer():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def run():
                try:
                    async for tok in self.responses_stream(*args, model=model, on_chunk=on_chunk, **kwargs):
                        if stop_event and stop_event.is_set():
                            break
                        q.put(("token", tok))
                    q.put(("done", None))
                except Exception as e:
                    err[0] = e
                    q.put(("error", e))

            task = loop.create_task(run())

            def watcher():
                if not stop_event:
                    return
                stop_event.wait()
                if not task.done():
                    loop.call_soon_threadsafe(task.cancel)

            watch_thread = threading.Thread(target=watcher, daemon=True)
            watch_thread.start()

            try:
                loop.run_until_complete(task)
            finally:
                loop.close()

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        try:
            while True:
                typ, data = q.get()
                if typ == "error":
                    raise err[0]
                if typ == "done":
                    break
                if typ == "token":
                    yield data
        finally:
            thread.join(timeout=5)
