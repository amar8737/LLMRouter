"""Langfuse tracing middleware.

Records every routed operation as a Langfuse *generation* (the request itself
is auto-created as its own trace). It integrates with LLMRouter's existing
middleware hooks, so enabling tracing requires no changes to the routing core.

Setup
-----
Configure Langfuse through the environment::

    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com

Then add the middleware to a ``RouterConfig.middleware`` list (the
``RouterFactory`` also auto-enables it when those variables are present).
"""

from __future__ import annotations

import logging
import time
from typing import Any

try:
    from langfuse import get_client
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Langfuse tracing requires the 'langfuse' package. "
        "Install it with: pip install llmrouterx[langfuse]"
    ) from exc

from ..context import RequestContext
from .base import BaseMiddleware

logger = logging.getLogger("llmrouterx.middleware.langfuse")

_STATE_KEY = "langfuse_observation"


def is_configured() -> bool:
    """Whether Langfuse credentials/host are present in the environment."""
    import os

    has_host = bool(os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL"))
    has_keys = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
    return bool(has_host or has_keys)


class LangfuseMiddleware(BaseMiddleware):
    """Trace each routed operation as a Langfuse generation."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client

    def _resolve_client(self) -> Any:
        if self._client is None:
            self._client = get_client()
        return self._client

    def _start(
        self,
        operation: str,
        payload: dict[str, Any],
        context: RequestContext,
    ) -> None:
        try:
            client = self._resolve_client()
            model = context.model or payload.get("model")
            input_ = payload.get("prompt") or payload.get("text") or payload

            cm = client.start_as_current_observation(
                name=operation,
                as_type="generation",
                input=input_,
                model=model,
                metadata={
                    "request_id": context.request_id,
                    "operation": operation,
                    "model": model,
                    "tenant": context.tenant,
                },
            )
            observation = cm.__enter__()
            context.set(_STATE_KEY, (cm, observation, time.perf_counter()))
        except Exception:  # pragma: no cover - tracing must never break routing
            self._client = None
            logger.warning(
                "Langfuse tracing disabled (client unavailable); "
                "will not add observations for this request."
            )

    def _end(
        self,
        context: RequestContext,
        *,
        output: Any | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        state = context.get(_STATE_KEY)
        if not state:
            return
        cm, observation, started = state
        try:
            update: dict[str, Any] = {}
            if output is not None:
                update["output"] = output
            if level is not None:
                update["level"] = level
            if status_message is not None:
                update["status_message"] = status_message

            update["metadata"] = {
                "request_id": context.request_id,
                "operation": context.operation,
                "model": context.model,
                "provider": context.provider,
                "api_key_suffix": (context.api_key[-4:] if context.api_key else None),
                "retry_count": context.retry_count,
                "latency_ms": round(
                    ((context.finished_at or time.perf_counter()) - started) * 1000, 2
                ),
            }
            observation.update(**update)
            observation.end()
        finally:
            cm.__exit__(None, None, None)

    # ------------------------------------------------------------------
    # BaseMiddleware hooks
    # ------------------------------------------------------------------

    async def before_request(
        self,
        operation: str,
        payload: dict[str, Any],
        context: RequestContext,
    ) -> None:
        self._start(operation, payload, context)
        return None

    async def after_response(
        self,
        operation: str,
        payload: dict[str, Any],
        response: Any,
        context: RequestContext,
    ) -> None:
        self._end(context, output=_coerce_output(response))
        return None

    async def on_exception(
        self,
        operation: str,
        payload: dict[str, Any],
        exception: BaseException,
        context: RequestContext,
    ) -> None:
        self._end(
            context,
            level="ERROR",
            status_message=f"{type(exception).__name__}: {exception}",
        )


    def flush(self) -> None:
        """Flush pending traces (call on application shutdown)."""
        try:
            self._resolve_client().flush()
        except Exception:  # pragma: no cover
            logger.exception("Langfuse flush failed.")


def _coerce_output(response: Any) -> Any:
    """Reduce a provider response to something JSON-serialisable."""
    if response is None or isinstance(response, (str, int, float, bool)):
        return response
    if isinstance(response, dict):
        return response.get("response", response)
    return str(response)
