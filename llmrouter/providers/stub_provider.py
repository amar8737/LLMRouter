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
