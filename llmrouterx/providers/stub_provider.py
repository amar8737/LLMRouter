import asyncio


class StubClient:
    """A tiny in-process provider client used for testing and examples."""

    def __init__(self, name: str = "stub"):
        self.name = name

    async def request(self, op: str, payload: dict, api_key: str | None = None, **kwargs):
        await asyncio.sleep(0)
        # payload may include prompt or text etc.
        body = payload.get("prompt") or payload.get("text") or payload.get("args")
        return {
            "provider": self.name,
            "api_key": api_key,
            "op": op,
            "response": f"Echo from {self.name}: {body}",
        }


class StubStreamingClient:
    """A stub client that can stream chunks asynchronously or return a full response.

    Use `request(..., stream=True)` to get an async iterator of chunk dicts
    with a `response` field. For non-streaming calls `request(..., stream=False)`
    returns an awaitable coroutine producing the full response dict.
    """

    def __init__(self, name: str = "stub-stream", chunks: list | None = None, delay: float = 0.01):
        self.name = name
        self.chunks = chunks or ["Hello", " world", " from", " stub"]
        self.delay = delay

    def request(
        self, op: str, payload: dict, api_key: str | None = None, stream: bool = False, **kwargs
    ):
        if stream:

            async def gen():
                for part in self.chunks:
                    await asyncio.sleep(self.delay)
                    yield {"provider": self.name, "api_key": api_key, "op": op, "response": part}

            return gen()

        async def full():
            await asyncio.sleep(0)
            return {
                "provider": self.name,
                "api_key": api_key,
                "op": op,
                "response": "".join(self.chunks),
            }

        return full()
