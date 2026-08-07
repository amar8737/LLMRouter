from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Literal

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Server dependencies missing. Install them with: pip install llmrouterx[server]"
    ) from exc

from llmrouterx.config.config import RouterConfig
from llmrouterx.exceptions import NoHealthyClientError
from llmrouterx.router.factory import RouterFactory
from llmrouterx.router.llmrouter import LLMRouter

logger = logging.getLogger("llmrouterx.server")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int | None = Field(default=None, gt=0)


class CompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[CompletionChoice]


class ModelEntry(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelEntry]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    providers: dict[str, bool]
    healthy_count: int
    total_providers: int
    uptime_seconds: float


class ErrorEnvelope(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorEnvelope


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_prompt(messages: list[ChatMessage]) -> str:
    """Use the most recent message content as the routing prompt."""
    if not messages:
        raise HTTPException(
            status_code=400,
            detail="messages must contain at least one message",
        )
    return messages[-1].content


def _coerce_text(result: Any) -> str:
    """Normalize a provider result to a string.

    Real adapters return ``str``; some test stubs return ``{"response": ...}``.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return str(result.get("response", result))
    return str(result)


def _error_payload(
    message: str,
    error_type: str,
    *,
    param: str | None = None,
    code: str | None = None,
) -> ErrorResponse:
    return ErrorResponse(
        error=ErrorEnvelope(message=message, type=error_type, param=param, code=code)
    )


def _sse_chunk(text: str, request_id: str, created: int) -> str:
    data = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(data)}\n\n"


def _sse_done(request_id: str, created: int) -> str:
    data = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(data)}\n\n"


class _AccessLogMiddleware(BaseHTTPMiddleware):
    """Inject a request ID and emit a per-request access log line."""

    def __init__(self, app, enabled: bool = True) -> None:
        super().__init__(app)
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "%s %s [req_id=%s] failed",
                request.method,
                request.url.path,
                request_id,
            )
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers["X-Request-ID"] = request_id
        if self.enabled:
            logger.info(
                "%s %s %d %.1fms [req_id=%s]",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
                request_id,
            )
        return response


def create_app(
    router: LLMRouter | None = None,
    config: RouterConfig | None = None,
    *,
    config_path: str | os.PathLike[str] | None = None,
    cors_origins: list[str] | None = None,
    enable_access_log: bool = True,
) -> FastAPI:
    """Application factory for the standalone HTTP Gateway.

    Resolution order for the underlying router: an explicit ``router`` wins,
    otherwise ``config``, otherwise ``config_path`` (JSON file), otherwise
    ``RouterConfig.from_env()``.

    Args:
        router: An already-built :class:`LLMRouter` to embed.
        config: A :class:`RouterConfig` used to build a fresh router.
        config_path: Path to a JSON config file (see ``RouterConfig.from_file``).
        cors_origins: If provided, enable CORS for these origins.
        enable_access_log: Whether to emit per-request access log lines.
    """
    start_time = time.monotonic()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if router is not None:
            app.state.llm_router = router
            app.state.owns_router = False
        else:
            cfg = config
            if cfg is None and config_path is not None:
                cfg = RouterConfig.from_file(config_path)
            cfg = cfg or RouterConfig.from_env()
            app.state.llm_router = RouterFactory.build(cfg)
            app.state.owns_router = True

        logger.info("LLMRouter Gateway API initialized.")
        yield

        if app.state.owns_router and app.state.llm_router is not None:
            await app.state.llm_router.close()
            _flush_observability(app.state.llm_router)
            logger.info("LLMRouter Gateway API shut down cleanly.")

    app = FastAPI(title="LLMRouter Gateway", version="0.1.0", lifespan=lifespan)

    app.add_middleware(_AccessLogMiddleware, enabled=enable_access_log)

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ------------------------------------------------------------------
    # Error handling: OpenAI-compatible error envelope
    # ------------------------------------------------------------------

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        message = detail if isinstance(detail, str) else "Request failed"
        error_type = {
            400: "invalid_request_error",
            404: "invalid_request_error",
            422: "invalid_request_error",
            429: "rate_limit_error",
            500: "server_error",
            503: "server_error",
        }.get(exc.status_code, "api_error")
        payload = _error_payload(message, error_type, code=str(exc.status_code))
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        payload = _error_payload(
            "Invalid request parameters",
            "invalid_request_error",
            code="422",
        )
        return JSONResponse(status_code=422, content=payload.model_dump())

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health_check(request: Request) -> HealthResponse:
        rt: LLMRouter = request.app.state.llm_router
        provider_health = await rt.health()
        total = len(provider_health)
        healthy = sum(1 for value in provider_health.values() if value)
        return HealthResponse(
            status="ok" if total and healthy == total else "degraded",
            providers=provider_health,
            healthy_count=healthy,
            total_providers=total,
            uptime_seconds=time.monotonic() - start_time,
        )

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models(request: Request) -> ModelListResponse:
        rt: LLMRouter = request.app.state.llm_router
        models: list[ModelEntry] = []
        for provider in getattr(rt, "providers", []):
            name = getattr(provider, "name", None)
            if name:
                models.append(ModelEntry(id=name, owned_by=name))
        return ModelListResponse(data=models)

    @app.get("/metrics")
    async def metrics(request: Request) -> dict[str, Any]:
        rt: LLMRouter = request.app.state.llm_router
        snapshot = rt.get_metrics()
        snapshot["started_at"] = start_time
        return snapshot

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionRequest, request: Request
    ) -> Any:
        rt: LLMRouter = request.app.state.llm_router
        prompt = _extract_prompt(req.messages)

        if req.stream:
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())

            async def sse_generator() -> AsyncGenerator[str, None]:
                try:
                    async for chunk in rt.stream(prompt=prompt, model=req.model):
                        yield _sse_chunk(_coerce_text(chunk), request_id, created)
                except NoHealthyClientError as exc:
                    yield _sse_error(str(exc), error_type="server_error", code="503")
                    return
                except Exception as exc:
                    logger.exception("Error streaming chat completion")
                    yield _sse_error(str(exc), error_type="server_error", code="500")
                    return
                yield _sse_done(request_id, created)
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse_generator(), media_type="text/event-stream")

        try:
            text = _coerce_text(await rt.chat(prompt=prompt, model=req.model))
        except NoHealthyClientError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Error serving chat completion request")
            raise HTTPException(status_code=500, detail="Internal Router Error") from exc

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=req.model,
            choices=[CompletionChoice(message=ChatMessage(role="assistant", content=text))],
        )

    return app


def _sse_error(message: str, *, error_type: str, code: str) -> str:
    data = _error_payload(message, error_type, code=code).model_dump()
    return f"data: {json.dumps(data)}\n\n"


def _flush_observability(router: LLMRouter) -> None:
    """Flush any observability middleware with a ``flush()`` on shutdown."""
    for middleware in getattr(router, "middleware", []):
        flush = getattr(middleware, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception:  # pragma: no cover
                logger.exception("Observability middleware flush failed.")
