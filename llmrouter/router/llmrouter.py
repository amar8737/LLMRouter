import asyncio
import time

from ..metrics.metrics import MetricsCollector
from ..middleware.base import BaseMiddleware
from ..providers.composite_router import CompositeRouter
from ..retry.exponential import ExponentialRetry


class LLMRouter:
    """Public router interface for applications.

    Orchestrates middleware, retries, and metrics around the CompositeRouter.
    """

    def __init__(self, composite_router: CompositeRouter, *, retry: ExponentialRetry = None, metrics: MetricsCollector = None, middleware: list[BaseMiddleware] | None = None, max_retries: int = 3):
        self._composite = composite_router
        self.retry = retry or ExponentialRetry(max_retries=max_retries)
        self.metrics = metrics or MetricsCollector()
        self.middleware = middleware or []

    async def _run_middlewares_before(self, op: str, payload: dict):
        for m in self.middleware:
            payload = await m.before_request(op, payload) or payload
        return payload

    async def _run_middlewares_after(self, op: str, payload: dict, response: dict):
        for m in self.middleware:
            response = await m.after_response(op, payload, response) or response
        return response

    async def _route_with_retry(self, op: str, payload: dict):
        attempt = 0
        start = time.monotonic()
        while True:
            try:
                payload = await self._run_middlewares_before(op, payload)
                resp = await self._composite.handle(op, payload,)
                resp = await self._run_middlewares_after(op, payload, resp)
                elapsed = time.monotonic() - start
                self.metrics.incr(f"requests.{op}")
                self.metrics.timing(f"latency.{op}", elapsed)
                return resp
            except Exception as e:
                attempt += 1
                self.metrics.incr(f"errors.{op}")
                if self.retry.should_retry(e, attempt):
                    backoff = self.retry.get_backoff(attempt)
                    await asyncio.sleep(backoff)
                    continue
                raise

    async def chat(self, prompt: str, **kwargs):
        payload = {"prompt": prompt, **kwargs}
        return await self._route_with_retry("chat", payload)

    async def embeddings(self, text: str, **kwargs):
        payload = {"text": text, **kwargs}
        return await self._route_with_retry("embeddings", payload)

    async def responses(self, *args, **kwargs):
        payload = {"args": args, **kwargs}
        return await self._route_with_retry("responses", payload)

