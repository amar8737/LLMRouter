class ClientNode:
    """Represents a single API key / client instance."""

    def __init__(self, api_key: str, client):
        self.api_key = api_key
        self.client = client
        self._healthy = True
        self.active_requests = 0
        self.last_success = None
        self.last_failure = None
        self.failures = 0

    async def is_healthy(self) -> bool:
        return bool(self._healthy)

    async def send(self, op: str, prompt: str, **kwargs):
        # Track active requests for scheduling
        self.active_requests += 1
        try:
            resp = await self.client.request(op, prompt, api_key=self.api_key, **kwargs)
            self.last_success = True
            self.failures = 0
            return resp
        except Exception as e:
            self.last_failure = True
            self.failures += 1
            raise
        finally:
            self.active_requests = max(0, self.active_requests - 1)
