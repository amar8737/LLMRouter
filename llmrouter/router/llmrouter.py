import asyncio
import time

from ..metrics.metrics import MetricsCollector
from ..middleware.base import BaseMiddleware
from ..providers.composite_router import CompositeRouter
from ..providers.provider_router import ProviderRouter
from ..client.client_node import ClientNode
from ..config.config import RouterConfig
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

        # Run before-middlewares once to avoid side-effects across retries
        payload = await self._run_middlewares_before(op, payload)

        while True:
            try:
                resp = await self._composite.handle(op, payload)
                resp = await self._run_middlewares_after(op, payload, resp)
                elapsed = time.monotonic() - start
                self.metrics.incr(f"requests.{op}")
                self.metrics.timing(f"latency.{op}", elapsed)
                return resp
            except Exception as e:
                # Immediately propagate cancellation
                if isinstance(e, asyncio.CancelledError):
                    raise
                attempt += 1
                self.metrics.incr(f"errors.{op}")
                if self.retry.should_retry(e, attempt):
                    backoff = self.retry.get_backoff(e, attempt)
                    # record backoff in metrics
                    try:
                        self.metrics.timing(f"retry.backoff.{op}", backoff)
                    except Exception:
                        pass
                    await asyncio.sleep(backoff)
                    continue
                # exhausted or non-retryable
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

    @classmethod
    def from_config(cls, config: RouterConfig):
        """Build an `LLMRouter` from a `RouterConfig` instance.

        `config.providers` may contain already-built `ProviderRouter` objects
        or dict-like provider descriptions of the form:

        {"name": "openai", "clients": [{"api_key": "sk-..", "client": client_obj}, ...], "scheduler": scheduler}
        """
        providers = []
        for p in config.providers:
            if isinstance(p, ProviderRouter):
                providers.append(p)
                continue
            # assume mapping-like
            name = p.get("name") or p.get("provider")
            raw_clients = p.get("clients", [])
            scheduler = p.get("scheduler", config.scheduler)
            clients = []
            for rc in raw_clients:
                api_key = rc.get("api_key") or rc.get("key")
                client_obj = rc.get("client")
                clients.append(ClientNode(api_key, client_obj, max_concurrent=config.max_concurrent_per_key))
            providers.append(ProviderRouter(name, clients, scheduler=scheduler))

        composite = CompositeRouter(providers)
        return cls(composite, retry=config.retry, metrics=None, middleware=config.middleware, max_retries=config.max_retries)

