from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Any, Literal

try:
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Server dependencies missing. Install them with: pip install llmrouterx[server]"
    ) from exc

from llmrouterx.config.config import RouterConfig
from llmrouterx.exceptions import NoHealthyClientError, StreamError
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


def _make_bearer_guard(token: str | Sequence[str] | None) -> Any:
    """Build a FastAPI dependency that requires ``Bearer <token>``.

    ``token`` may be a single string or a collection of accepted values.
    When it is None the guard is a no-op (auth disabled), so the gateway
    remains usable out of the box and is only locked down when an admin/API
    token is configured.
    """
    if token is None:
        accepted: set[str] | None = None
    elif isinstance(token, str):
        accepted = {token}
    else:
        accepted = set(token)

    async def _guard(request: Request) -> None:
        if accepted is None:
            return
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and auth[len("Bearer ") :] in accepted:
            return
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return _guard


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
    admin_token: str | None = None,
    api_keys: Sequence[str] | None = None,
    docs_enabled: bool = True,
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
        admin_token: If set, require ``Authorization: Bearer <token>`` on the
            admin endpoints (``/dashboard``, ``/metrics``). None disables auth.
        api_keys: If set, require ``Authorization: Bearer <key>`` on the LLM
            endpoints (``/v1/*``) using one of these keys. None disables auth.
        docs_enabled: Whether to expose the interactive OpenAPI docs.
    """
    start_time = time.monotonic()

    # Fall back to environment configuration so both inline-config and
    # multi-worker (factory) modes pick up auth/docs settings consistently.
    if admin_token is None:
        admin_token = os.getenv("LLMROUTER_ADMIN_TOKEN") or None
    if api_keys is None:
        raw_keys = os.getenv("LLMROUTER_API_KEYS")
        api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()] if raw_keys else None
    docs_env = os.getenv("LLMROUTER_DOCS")
    if docs_env is not None:
        docs_enabled = docs_env.strip().lower() not in {"0", "false", "no"}

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
            from llmrouterx.router.llmrouter import shutdown_shared_http_client

            await shutdown_shared_http_client()
            logger.info("LLMRouter Gateway API shut down cleanly.")

    app = FastAPI(
        title="LLMRouter Gateway",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
    )

    admin_guard = _make_bearer_guard(admin_token)
    api_key_guard = _make_bearer_guard(list(api_keys) if api_keys else None)

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

    @app.get("/v1/models", response_model=ModelListResponse, dependencies=[Depends(api_key_guard)])
    async def list_models(request: Request) -> ModelListResponse:
        rt: LLMRouter = request.app.state.llm_router
        models: list[ModelEntry] = []
        for provider in getattr(rt, "providers", []):
            name = getattr(provider, "name", None)
            if name:
                models.append(ModelEntry(id=name, owned_by=name))
        return ModelListResponse(data=models)

    @app.get("/dashboard", response_class=HTMLResponse, dependencies=[Depends(admin_guard)])
    async def dashboard(request: Request) -> HTMLResponse:
        """Zero-config observability dashboard (auto-refreshing)."""
        html_content = _DASHBOARD_HTML
        return HTMLResponse(content=html_content)

    @app.get("/metrics", dependencies=[Depends(admin_guard)])
    async def metrics(request: Request) -> dict[str, Any]:
        rt: LLMRouter = request.app.state.llm_router
        snapshot = rt.get_metrics()
        snapshot["started_at"] = start_time
        return snapshot

    @app.post(
        "/v1/chat/completions",
        dependencies=[Depends(api_key_guard)],
    )
    async def chat_completions(req: ChatCompletionRequest, request: Request) -> Any:
        rt: LLMRouter = request.app.state.llm_router
        prompt = _extract_prompt(req.messages)

        kwargs: dict[str, Any] = {}
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.max_tokens is not None:
            kwargs["max_tokens"] = req.max_tokens

        if req.stream:
            request_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())

            async def sse_generator() -> AsyncGenerator[str, None]:
                try:
                    async for chunk in rt.stream(prompt=prompt, model=req.model, **kwargs):
                        yield _sse_chunk(_coerce_text(chunk), request_id, created)
                except NoHealthyClientError as exc:
                    yield _sse_error(str(exc), error_type="server_error", code="503")
                except StreamError:
                    # A provider died mid-stream; the underlying cause is
                    # re-raised by the router as a 503-family error.
                    yield _sse_error(
                        "Provider stream interrupted",
                        error_type="server_error",
                        code="503",
                    )
                except Exception as exc:
                    logger.exception("Error streaming chat completion")
                    yield _sse_error(str(exc), error_type="server_error", code="500")
                finally:
                    # Signal end-of-stream (with or without an error) so
                    # OpenAI-compatible clients finalize cleanly.
                    yield _sse_done(request_id, created)
                    yield "data: [DONE]\n\n"

            return StreamingResponse(sse_generator(), media_type="text/event-stream")

        try:
            text = _coerce_text(await rt.chat(prompt=prompt, model=req.model, **kwargs))
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


_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLMRouter Operations</title>
    <style>
        body {
            font-family: system-ui, sans-serif;
            background: #121212; color: #e0e0e0; padding: 2rem;
        }
        h1 { margin-top: 0; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        .card {
            background: #1e1e1e; padding: 1.5rem; border-radius: 8px;
            margin-bottom: 1rem; border: 1px solid #333;
        }
        pre {
            background: #000; padding: 1rem; overflow-x: auto;
            border-radius: 4px; color: #4af626;
        }
        h2 { margin-top: 0; }
        @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <h1>LLMRouter Gateway</h1>
    <div class="grid">
        <div class="card">
            <h2>Live Health</h2>
            <pre id="health-data">Loading...</pre>
        </div>
        <div class="card">
            <h2>Live Metrics</h2>
            <pre id="metrics-data">Loading...</pre>
        </div>
    </div>
    <script>
        async function poll() {
            try {
                const [healthRes, metricsRes] =
                    await Promise.all([fetch('/health'), fetch('/metrics')]);
                const health = JSON.stringify(await healthRes.json(), null, 2);
                const metrics = JSON.stringify(await metricsRes.json(), null, 2);
                document.getElementById('health-data').textContent = health;
                document.getElementById('metrics-data').textContent = metrics;
            } catch (e) {
                console.error("Polling failed", e);
            }
        }
        poll();
        setInterval(poll, 2000);
    </script>
</body>
</html>
"""


def _flush_observability(router: LLMRouter) -> None:
    """Flush any observability middleware with a ``flush()`` on shutdown."""
    for middleware in getattr(router, "middleware", []):
        flush = getattr(middleware, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception:  # pragma: no cover
                logger.exception("Observability middleware flush failed.")
