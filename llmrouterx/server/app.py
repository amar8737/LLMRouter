# ruff: noqa: E501
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from statistics import mean
from typing import Any, Literal

try:
    from fastapi import Depends, FastAPI, HTTPException, Request, status
    from fastapi.exceptions import RequestValidationError
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Server dependencies missing. Install them with: pip install llmrouterx[server]"
    ) from exc

from llmrouterx.config.api_keys import (
    KEY_PREFIX,
    VALID_SCOPES,
    create_api_key,
    get_api_key_db,
    validate_api_key,
)
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
# API Key Management Models
# ---------------------------------------------------------------------------


class APIKeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    scopes: list[str] = Field(default_factory=lambda: ["chat", "embeddings"])
    expires_in: str | None = Field(
        default=None,
        pattern=r"^\d+[hdm]$",  # e.g., "24h", "7d", "30m"
    )


class APIKeyCreateResponse(BaseModel):
    key: str  # Full key (only returned once)
    prefix: str
    name: str
    scopes: list[str]
    created_at: str
    expires_at: str | None


class APIKeyListResponse(BaseModel):
    keys: list[dict[str, Any]]


class APIKeyRevokeResponse(BaseModel):
    revoked: bool
    prefix: str


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


def _make_bearer_guard(
    token: str | Sequence[str] | None,
    *,
    check_db: bool = False,
    db_path: str | None = None,
) -> Any:
    """Build a FastAPI dependency that requires ``Bearer <token>``.

    ``token`` may be a single string or a collection of accepted values.
    When it is None the guard is a no-op (auth disabled), so the gateway
    remains usable out of the box and is only locked down when an admin/API
    token is configured.

    If ``check_db`` is True, also validate against the API key database
    (for /v1/* endpoints).
    """
    if token is None:
        accepted: set[str] | None = None
    elif isinstance(token, str):
        accepted = {token}
    else:
        accepted = set(token)

    async def _guard(request: Request) -> None:
        if accepted is None and not check_db:
            return

        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            bearer_token = auth[len("Bearer ") :]
            if accepted is not None and bearer_token in accepted:
                return
            if check_db:
                # Check API key database
                record = validate_api_key(bearer_token, db_path)
                if (
                    record
                    and record.is_valid
                    and ("chat" in record.scopes or "embeddings" in record.scopes)
                ):
                    return

        # Also check query param for token
        query_token = request.query_params.get("token")
        if query_token:
            if accepted is not None and query_token in accepted:
                return
            if check_db:
                record = validate_api_key(query_token, db_path)
                if (
                    record
                    and record.is_valid
                    and ("chat" in record.scopes or "embeddings" in record.scopes)
                ):
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
    api_keys_db_path: str | None = None,
    docs_enabled: bool = True,
    health_timeout: float | None = None,
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
            admin endpoints (``/dashboard``, ``/metrics``, ``/admin/*``). None disables auth.
        api_keys: If set, require ``Authorization: Bearer <key>`` on the LLM
            endpoints (``/v1/*``) using one of these keys. None disables auth.
        api_keys_db_path: Path to the API key database file. If None, uses default location.
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
    if api_keys_db_path is None:
        api_keys_db_path = os.getenv("LLMROUTER_API_KEYS_DB")
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
        app.state.health_timeout = health_timeout

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
    # api_key_guard checks both env var keys and database keys
    api_key_guard = _make_bearer_guard(
        list(api_keys) if api_keys else None,
        check_db=True,
        db_path=api_keys_db_path,
    )

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
        health_timeout = request.app.state.health_timeout

        if health_timeout is not None:
            try:
                provider_health = await asyncio.wait_for(rt.health(), timeout=health_timeout)
            except asyncio.TimeoutError:
                provider_health = {}
        else:
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
        html_content = _DASHBOARD_HTML.replace("__ADMIN_TOKEN__", admin_token or "")
        return HTMLResponse(content=html_content)

    @app.get("/metrics", dependencies=[Depends(admin_guard)])
    async def metrics(request: Request) -> dict[str, Any]:
        rt: LLMRouter = request.app.state.llm_router
        snapshot = rt.get_metrics()

        # Compute derived metrics for dashboard
        provider_health = await rt.health()
        derived = _aggregate_per_provider(snapshot, rt.providers, provider_health)

        # Collect circuit breaker state
        circuit_breakers = _collect_circuit_breaker_state(rt.providers)

        return {
            "snapshot": snapshot,
            "derived": derived,
            "circuit_breakers": circuit_breakers,
            "uptime_seconds": time.monotonic() - start_time,
            "timestamp": time.time(),
        }

    # ------------------------------------------------------------------
    # API Key Management (Admin only)
    # ------------------------------------------------------------------

    @app.get(
        "/admin/api-keys", response_model=APIKeyListResponse, dependencies=[Depends(admin_guard)]
    )
    async def list_api_keys(request: Request) -> APIKeyListResponse:
        """List all API keys (admin only)."""
        db = get_api_key_db()
        keys = db.list_keys(include_revoked=True)
        return APIKeyListResponse(
            keys=[
                {
                    "id": k.id,
                    "prefix": k.prefix,
                    "name": k.name,
                    "scopes": list(k.scopes),
                    "created_at": k.created_at.isoformat(),
                    "expires_at": k.expires_at.isoformat() if k.expires_at else None,
                    "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
                    "revoked": k.revoked,
                    "is_valid": k.is_valid,
                }
                for k in keys
            ]
        )

    @app.post(
        "/admin/api-keys", response_model=APIKeyCreateResponse, dependencies=[Depends(admin_guard)]
    )
    async def create_api_key_endpoint(request: Request) -> APIKeyCreateResponse:
        """Create a new API key (admin only)."""
        body = await request.json()
        # Validate scopes
        for scope in body.get("scopes", ["chat", "embeddings"]):
            if scope not in VALID_SCOPES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid scope: {scope}. Valid scopes: {VALID_SCOPES}",
                )

        full_key, record = create_api_key(
            name=body["name"],
            scopes=tuple(body.get("scopes", ["chat", "embeddings"])),
            expires_in=body.get("expires_in"),
        )

        return APIKeyCreateResponse(
            key=full_key,
            prefix=record.prefix,
            name=record.name,
            scopes=list(record.scopes),
            created_at=record.created_at.isoformat(),
            expires_at=record.expires_at.isoformat() if record.expires_at else None,
        )

    @app.delete(
        "/admin/api-keys/{prefix}",
        response_model=APIKeyRevokeResponse,
        dependencies=[Depends(admin_guard)],
    )
    async def revoke_api_key(prefix: str) -> APIKeyRevokeResponse:
        """Revoke an API key by prefix (admin only)."""
        db = get_api_key_db()
        if not prefix.startswith(KEY_PREFIX):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid key prefix. Must start with {KEY_PREFIX}",
            )

        revoked = db.revoke_key(prefix)
        if not revoked:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"API key with prefix {prefix} not found",
            )

        return APIKeyRevokeResponse(revoked=True, prefix=prefix)

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


_DASHBOARD_HTML = """\  # ruff: noqa: E501
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLMRouter Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #f5f5f5;
            color: #333;
            padding: 2rem;
            margin: 0;
        }
        h1 { margin-top: 0; margin-bottom: 2rem; }
        h2 { margin: 0 0 1rem; font-size: 16px; }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }
        .grid-2 {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1rem;
        }
        
        .card {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        
        .metric {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        .metric-label { font-size: 12px; color: #999; margin-bottom: 8px; }
        .metric-value { font-size: 28px; font-weight: 500; margin: 0; }
        .metric-detail { font-size: 12px; color: #999; margin-top: 4px; }
        
        .status-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px;
            background: #f9f9f9;
            border-radius: 4px;
            margin-bottom: 8px;
        }
        .status-item-name { display: flex; align-items: center; gap: 8px; font-weight: 500; }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; background: #4caf50; }
        .status-item-time { font-size: 12px; color: #999; }
        
        .error-row {
            display: flex;
            justify-content: space-between;
            padding: 8px;
            background: #f9f9f9;
            border-radius: 4px;
            margin-bottom: 8px;
            font-size: 14px;
        }
        .error-type { }
        .error-count { font-weight: 500; }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 1rem;
        }
        
        .latency-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            font-size: 13px;
        }
        .latency-label { flex: 1; }
        .latency-chart {
            flex: 2;
            height: 20px;
            background: #f0f0f0;
            border-radius: 4px;
            overflow: hidden;
            margin: 0 12px;
        }
        .latency-fill { height: 100%; background: linear-gradient(to right, #4caf50, #ff9800); }
        .latency-value { flex: 0.5; text-align: right; font-weight: 500; }
        
        .loading { color: #999; }
        .error { color: #d32f2f; }
        .success { color: #4caf50; }
        
        @media (max-width: 1200px) {
            .grid-2 { grid-template-columns: 1fr; }
        }
        
        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            body { background: #121212; color: #e0e0e0; }
            .card {
                background: #1e1e1e;
                border-color: #333;
                box-shadow: 0 1px 2px rgba(0,0,0,0.2);
            }
            .metric { background: #1e1e1e; border-color: #333; }
            .metric-label { color: #999; }
            .metric-detail { color: #999; }
            .status-item { background: #2d2d2d; }
            .status-item-time { color: #999; }
            .error-row { background: #2d2d2d; }
            .sparkline-card { background: #1e1e1e; border-color: #333; }
            .sparkline-title { color: #e0e0e0; }
            .latency-table th { color: #999; }
            .latency-table th, .latency-table td { border-color: #333; }
            .loading { color: #999; }
        }
        
        .latency-table {
            width: 100%;
            font-size: 13px;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        .latency-table th, .latency-table td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        .latency-table th {
            font-weight: 500;
            color: #666;
        }

        /* API Key panel styles */
        .api-key-table {
            width: 100%;
            font-size: 13px;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        .api-key-table th, .api-key-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        .api-key-table th {
            font-weight: 500;
            color: #666;
        }
        .api-key-table tr:hover {
            background: #f5f5f5;
        }
        .scope-badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 500;
            margin-right: 4px;
        }
        .scope-chat { background: #e3f2fd; color: #1565c0; }
        .scope-embeddings { background: #f3e5f5; color: #7b1fa2; }
        .scope-admin { background: #fff3e0; color: #e65100; }
        .status-badge {
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
        }
        .status-valid { background: #e8f5e9; color: #2e7d32; }
        .status-revoked { background: #fdeaea; color: #c62828; }
        .status-expired { background: #fff3e0; color: #e65100; }
        .btn {
            padding: 6px 12px;
            border: none;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn-primary { background: #1976d2; color: white; }
        .btn-primary:hover { background: #1565c0; }
        .btn-danger { background: #d32f2f; color: white; }
        .btn-danger:hover { background: #b71c1c; }
        .btn-secondary { background: #757575; color: white; }
        .btn-secondary:hover { background: #616161; }
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        .modal.show { display: flex; }
        .modal-content {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            width: 90%;
            max-width: 500px;
            max-height: 90vh;
            overflow-y: auto;
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        .modal-title { font-size: 16px; font-weight: 500; }
        .modal-close { background: none; border: none; font-size: 20px; cursor: pointer; color: #999; }
        .form-group { margin-bottom: 1rem; }
        .form-label { display: block; font-size: 12px; font-weight: 500; margin-bottom: 4px; }
        .form-input { width: 100%; padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px; font-size: 13px; }
        .form-input:focus { outline: none; border-color: #1976d2; }
        .checkbox-group { display: flex; flex-wrap: wrap; gap: 8px; }
        .checkbox-label { display: flex; align-items: center; gap: 6px; font-size: 13px; cursor: pointer; }
        .key-display {
            background: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 1rem;
            margin: 1rem 0;
            font-family: monospace;
            font-size: 13px;
            word-break: break-all;
        }
        .copy-btn { margin-left: 8px; }
        .api-key-row { display: flex; align-items: center; justify-content: space-between; padding: 12px 0; }
        .api-key-info { flex: 1; }
        .api-key-name { font-weight: 500; }
        .api-key-meta { font-size: 12px; color: #999; margin-top: 4px; }
        .api-key-actions { display: flex; gap: 8px; }
        .empty-state { text-align: center; padding: 3rem; color: #999; }
    </style>
</head>
<body>
    <h1>LLMRouter Dashboard</h1>
    
    <!-- Summary metrics -->
    <div class="grid">
        <div class="metric">
            <div class="metric-label">Healthy Providers</div>
            <p class="metric-value" id="healthy-count">-</p>
            <div class="metric-detail">of <span id="total-providers">-</span></div>
        </div>
        <div class="metric">
            <div class="metric-label">Success Rate</div>
            <p class="metric-value" id="success-rate">-</p>
            <div class="metric-detail" id="success-detail">-</div>
        </div>
        <div class="metric">
            <div class="metric-label">Avg Latency</div>
            <p class="metric-value" id="avg-latency">-</p>
            <div class="metric-detail" id="latency-detail">-</div>
        </div>
        <div class="metric">
            <div class="metric-label">Total Requests</div>
            <p class="metric-value" id="total-requests">-</p>
            <div class="metric-detail">All time</div>
        </div>
    </div>
    
    <!-- Sparklines -->
    <div class="sparkline-grid" style="margin-top: 1.5rem;">
        <div class="sparkline-card">
            <div class="sparkline-header">
                <span class="sparkline-title">Latency (p50 / p95)</span>
                <span class="sparkline-value" id="sparkline-latency-value">-</span>
            </div>
            <canvas class="sparkline-canvas" id="sparkline-latency"></canvas>
        </div>
        <div class="sparkline-card">
            <div class="sparkline-header">
                <span class="sparkline-title">Error Rate</span>
                <span class="sparkline-value" id="sparkline-error-value">-</span>
            </div>
            <canvas class="sparkline-canvas" id="sparkline-error-rate"></canvas>
        </div>
        <div class="sparkline-card">
            <div class="sparkline-header">
                <span class="sparkline-title">Request Rate (req/s)</span>
                <span class="sparkline-value" id="sparkline-rps-value">-</span>
            </div>
            <canvas class="sparkline-canvas" id="sparkline-rps"></canvas>
        </div>
    </div>
    
    <!-- Main panels -->
    <div style="margin-top: 2rem;">
        <div class="grid-2">
            
            <!-- Health Panel -->
            <div class="card">
                <h2>Provider Health</h2>
                <div id="health-panel" class="loading">Loading...</div>
            </div>
            
            <!-- Latency Panel -->
            <div class="card">
                <h2>Latency (p95)</h2>
                <div id="latency-panel" class="loading">Loading...</div>
            </div>
            
        </div>
        
        <!-- Latency Details -->
        <div class="card" style="margin-top: 1rem;">
            <h2>Latency Breakdown (all providers)</h2>
            <table class="latency-table">
                <tr>
                    <th>Min</th>
                    <th>P50</th>
                    <th>P95</th>
                    <th>P99</th>
                    <th>Mean</th>
                    <th>Max</th>
                </tr>
                <tr>
                    <td><span id="global-min">-</span>ms</td>
                    <td><span id="global-p50">-</span>ms</td>
                    <td><span id="global-p95">-</span>ms</td>
                    <td><span id="global-p99">-</span>ms</td>
                    <td><span id="global-mean">-</span>ms</td>
                    <td><span id="global-max">-</span>ms</td>
                </tr>
            </table>
        </div>
        
        <!-- Error Panel -->
        <div class="card" style="margin-top: 1rem;">
            <h2>Error Breakdown (all time)</h2>
            <div id="error-panel" class="loading">Loading...</div>
        </div>
        
        <!-- Circuit Breaker Panel -->
        <div class="card" style="margin-top: 1rem;">
            <h2>Circuit Breakers</h2>
            <div id="circuit-breaker-panel" class="loading">Loading...</div>
        </div>
        
        <!-- API Keys Panel -->
        <div class="card" style="margin-top: 1rem;">
            <div class="modal-header">
                <h2>API Keys</h2>
                <button class="btn btn-primary" onclick="ApiKeys.openCreateModal()">Create Key</button>
            </div>
            <div id="api-keys-panel" class="loading">Loading...</div>
        </div>

    </div>
    
    <script>
        // Dashboard state & polling
        const Dashboard = {
            state: {
                derived: null,
                health: null,
                lastUpdate: null,
                // History for sparklines (max 60 points = 2 minutes at 2s interval)
                history: {
                    timestamps: [],
                    latency_p50: [],
                    latency_p95: [],
                    error_rate: [],
                    request_rate: [],
                    prev_request_count: 0,
                    prev_time: null
                },
                charts: {}
            },
            
            async fetch() {
                try {
                    const [metricsRes, healthRes] = await Promise.all([
                        fetch('/metrics'),
                        fetch('/health')
                    ]);
                    if (!metricsRes.ok) throw new Error(`HTTP ${metricsRes.status}`);
                    if (!healthRes.ok) throw new Error(`HTTP ${healthRes.status}`);
                    
                    const metricsData = await metricsRes.json();
                    const healthData = await healthRes.json();
                    
                    this.state.derived = metricsData.derived;
                    this.state.circuitBreakers = metricsData.circuit_breakers || {};
                    this.state.health = healthData.providers || {};
                    this.state.lastUpdate = new Date();
                    this.updateHistory();
                    this.render();
                } catch (err) {
                    console.error('Dashboard fetch failed:', err);
                    document.getElementById('health-panel').innerHTML = 
                        '<span class="error">Failed to load metrics</span>';
                }
            },
            
            updateHistory() {
                const g = this.state.derived.global;
                const now = this.state.lastUpdate;
                
                // Add timestamp (format as HH:MM:SS)
                const timeStr = now.toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                });
                this.state.history.timestamps.push(timeStr);
                
                // Latency percentiles
                this.state.history.latency_p50.push(Math.round(g.latency_stats.p50));
                this.state.history.latency_p95.push(Math.round(g.latency_stats.p95));
                
                // Error rate (percentage)
                this.state.history.error_rate.push((g.error_rate * 100).toFixed(1));
                
                // Request rate (requests per second)
                let rps = 0;
                if (this.state.history.prev_request_count > 0 && this.state.history.prev_time) {
                    const timeDiff = (now - this.state.history.prev_time) / 1000;
                    const countDiff = g.request_count - this.state.history.prev_request_count;
                    rps = timeDiff > 0 ? (countDiff / timeDiff).toFixed(1) : 0;
                }
                this.state.history.request_rate.push(rps);
                
                // Update prev values
                this.state.history.prev_request_count = g.request_count;
                this.state.history.prev_time = now;
                
                // Keep only last 60 points
                const MAX_POINTS = 60;
                const historyKeys = [
                    'timestamps',
                    'latency_p50',
                    'latency_p95',
                    'error_rate',
                    'request_rate',
                ];
                for (const key of historyKeys) {
                    if (this.state.history[key].length > MAX_POINTS) {
                        this.state.history[key].shift();
                    }
                }
                
                // Initialize or update charts
                this.initCharts();
                this.updateCharts();
            },
            
            initCharts() {
                if (this.state.charts.latency) return; // Already initialized
                
                const commonOptions = {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 300 },
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            enabled: true,
                            backgroundColor: 'rgba(0,0,0,0.7)',
                            padding: 8,
                            titleFont: { size: 11 },
                            bodyFont: { size: 11 }
                        }
                    },
                    scales: {
                        x: { display: false },
                        y: { 
                            display: false,
                            beginAtZero: true,
                            grace: '10%'
                        }
                    },
                    elements: {
                        line: { tension: 0.3, borderWidth: 2 },
                        point: { radius: 0, hoverRadius: 4 }
                    }
                };
                
                // Latency sparkline (p50 + p95)
                this.state.charts.latency = new Chart(
                    document.getElementById('sparkline-latency'),
                    {
                        type: 'line',
                        data: {
                            labels: this.state.history.timestamps,
                            datasets: [
                                {
                                    label: 'p50',
                                    data: this.state.history.latency_p50,
                                    borderColor: '#4caf50',
                                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                                    fill: true
                                },
                                {
                                    label: 'p95',
                                    data: this.state.history.latency_p95,
                                    borderColor: '#ff9800',
                                    backgroundColor: 'rgba(255, 152, 0, 0.1)',
                                    fill: true
                                }
                            ]
                        },
                        options: commonOptions
                    }
                );
                
                // Error rate sparkline
                this.state.charts.errorRate = new Chart(
                    document.getElementById('sparkline-error-rate'),
                    {
                        type: 'line',
                        data: {
                            labels: this.state.history.timestamps,
                            datasets: [{
                                label: 'Error %',
                                data: this.state.history.error_rate,
                                borderColor: '#f44336',
                                backgroundColor: 'rgba(244, 67, 54, 0.1)',
                                fill: true
                            }]
                        },
                        options: commonOptions
                    }
                );
                
                // Request rate sparkline
                this.state.charts.rps = new Chart(
                    document.getElementById('sparkline-rps'),
                    {
                        type: 'line',
                        data: {
                            labels: this.state.history.timestamps,
                            datasets: [{
                                label: 'req/s',
                                data: this.state.history.request_rate,
                                borderColor: '#2196f3',
                                backgroundColor: 'rgba(33, 150, 243, 0.1)',
                                fill: true
                            }]
                        },
                        options: commonOptions
                    }
                );
            },
            
            updateCharts() {
                if (!this.state.charts.latency) return;
                
                const h = this.state.history;
                
                // Update latency chart
                this.state.charts.latency.data.labels = h.timestamps;
                this.state.charts.latency.data.datasets[0].data = h.latency_p50;
                this.state.charts.latency.data.datasets[1].data = h.latency_p95;
                this.state.charts.latency.update('none');
                
                // Update error rate chart
                this.state.charts.errorRate.data.labels = h.timestamps;
                this.state.charts.errorRate.data.datasets[0].data = h.error_rate;
                this.state.charts.errorRate.update('none');
                
                // Update RPS chart
                this.state.charts.rps.data.labels = h.timestamps;
                this.state.charts.rps.data.datasets[0].data = h.request_rate;
                this.state.charts.rps.update('none');
                
                // Update current values
                const lastP50 = h.latency_p50.length
                    ? h.latency_p50[h.latency_p50.length - 1]
                    : null;
                const lastP95 = h.latency_p95.length
                    ? h.latency_p95[h.latency_p95.length - 1]
                    : null;
                document.getElementById('sparkline-latency-value').textContent =
                    lastP50 !== null ? `p50: ${lastP50}ms | p95: ${lastP95}ms` : '-';
                document.getElementById('sparkline-error-value').textContent =
                    h.error_rate.length ? `${h.error_rate[h.error_rate.length - 1]}%` : '-';
                document.getElementById('sparkline-rps-value').textContent =
                    h.request_rate.length ? `${h.request_rate[h.request_rate.length - 1]}` : '-';
            },
            
            render() {
                this.renderSummary();
                this.renderHealth();
                this.renderLatency();
                this.renderErrors();
                this.renderCircuitBreakers();
            },
            
            renderSummary() {
                const g = this.state.derived.global;
                
                document.getElementById('healthy-count').textContent = g.healthy_providers;
                document.getElementById('total-providers').textContent = g.total_providers;
                document.getElementById('success-rate').textContent = 
                    (g.success_rate * 100).toFixed(1) + '%';
                document.getElementById('success-detail').textContent = 
                    g.success_count + ' of ' + g.request_count + ' requests';
                document.getElementById('avg-latency').textContent = 
                    Math.round(g.latency_stats.mean) + 'ms';
                document.getElementById('latency-detail').textContent = 
                    'p95: ' + Math.round(g.latency_stats.p95) + 'ms';
                document.getElementById('total-requests').textContent = 
                    g.request_count.toLocaleString();
            },
            
            renderHealth() {
                const providers = this.state.derived.providers;
                const health = this.state.health;
                let html = '';
                
                for (const [name, data] of Object.entries(providers)) {
                    const isHealthy = health[name] === true;
                    const statusDot = isHealthy 
                        ? '<span class="status-dot"></span>'
                        : '<span class="status-dot" style="background: #f44336;"></span>';
                    const statusText = isHealthy ? 'Healthy' : 'Down';
                    
                    html += `
                        <div class="status-item">
                            <div class="status-item-name">
                                ${statusDot}
                                <span>${name}</span>
                                <span class="status-badge" style="
                                    background: ${isHealthy ? '#e8f5e9' : '#fdeaea'};
                                    color: ${isHealthy ? '#2e7d32' : '#c62828'};
                                    padding: 2px 8px;
                                    border-radius: 12px;
                                    font-size: 11px;
                                    font-weight: 500;
                                    margin-left: 8px;
                                ">${statusText}</span>
                            </div>
                            <div class="status-item-time">${data.request_count} requests</div>
                        </div>
                    `;
                }
                
                if (!html) html = '<span class="loading">No provider data</span>';
                
                document.getElementById('health-panel').innerHTML = html;
            },
            
            renderLatency() {
                const providers = this.state.derived.providers;
                const g = this.state.derived.global;
                
                let html = '';
                
                // Per-provider p95
                const maxP95 = Math.max(
                    ...Object.values(providers).map(p => p.latency_stats.p95 || 0)
                );
                
                for (const [name, data] of Object.entries(providers)) {
                    if (!data.latency_stats.p95) continue;
                    
                    const width = (data.latency_stats.p95 / maxP95) * 100;
                    html += `
                        <div class="latency-bar">
                            <div class="latency-label">${name}</div>
                            <div class="latency-chart">
                                <div class="latency-fill" style="width: ${width}%;"></div>
                            </div>
                            <div class="latency-value">${Math.round(data.latency_stats.p95)}ms</div>
                        </div>
                    `;
                }
                
                document.getElementById('latency-panel').innerHTML = html;
                
                // Global stats
                const gs = g.latency_stats;
                document.getElementById('global-min').textContent = Math.round(gs.min);
                document.getElementById('global-p50').textContent = Math.round(gs.p50);
                document.getElementById('global-p95').textContent = Math.round(gs.p95);
                document.getElementById('global-p99').textContent = Math.round(gs.p99);
                document.getElementById('global-mean').textContent = Math.round(gs.mean);
                document.getElementById('global-max').textContent = Math.round(gs.max);
            },
            
            renderErrors() {
                const g = this.state.derived.global;
                let html = '';
                
                const sortedErrors = Object.entries(g.error_breakdown)
                    .sort((a, b) => b[1] - a[1]);
                
                for (const [type, count] of sortedErrors) {
                    const pct = g.error_count > 0 
                        ? ((count / g.error_count) * 100).toFixed(0) 
                        : 0;
                    
                    html += `
                        <div class="error-row">
                            <span class="error-type">${type}</span>
                            <span class="error-count">${count} (${pct}%)</span>
                        </div>
                    `;
                }
                
                if (!html) html = '<span class="success">No errors</span>';
                
                document.getElementById('error-panel').innerHTML = html;
            },
            
            renderCircuitBreakers() {
                const cb = this.state.circuitBreakers || {};
                
                let html = '';
                
                for (const [providerName, clients] of Object.entries(cb)) {
                    if (!clients || clients.length === 0) continue;
                    
                    for (const client of clients) {
                        const state = client.state || 'UNKNOWN';
                        const isOpen = state === 'OPEN';
                        const isHalfOpen = state === 'HALF_OPEN';
                        const isClosed = state === 'CLOSED';
                        
                        let stateClass = 'status-dot';
                        let stateColor = '#4caf50'; // closed = green
                        let stateText = 'Closed';
                        
                        if (isOpen) {
                            stateColor = '#f44336'; // red
                            stateText = 'Open';
                        } else if (isHalfOpen) {
                            stateColor = '#ff9800'; // orange
                            stateText = 'Half-Open';
                        }
                        
                        html += `
                            <div class="status-item">
                                <div class="status-item-name">
                                    <span class="status-dot"
                                        style="background: ${stateColor};"></span>
                                    <span>${providerName}</span>
                                    <span class="status-badge" style="
                                        background: ${isOpen
                                            ? '#fdeaea'
                                            : isHalfOpen
                                                ? '#fff3e0'
                                                : '#e8f5e9'};
                                        color: ${isOpen
                                            ? '#c62828'
                                            : isHalfOpen
                                                ? '#e65100'
                                                : '#2e7d32'};
                                        padding: 2px 8px;
                                        border-radius: 12px;
                                        font-size: 11px;
                                        font-weight: 500;
                                        margin-left: 8px;
                                    ">${stateText}</span>
                                </div>
                                <div class="status-item-time">
                                    Key: ...${client.api_key_suffix}
                                    | Failures: ${client.failure_count}
                                </div>
                            </div>
                        `;
                    }
                }
                
                if (!html) html = '<span class="loading">No circuit breaker data</span>';
                
                document.getElementById('circuit-breaker-panel').innerHTML = html;
            },
            
            start() {
                this.fetch();
                setInterval(() => this.fetch(), 2000);
            }
        };
        
        // API Keys management
        const ApiKeys = {
            state: {
                keys: [],
                selectedPrefix: null,
            },
            
            async fetch() {
                try {
                    const res = await fetch('/admin/api-keys', {
                        headers: {
                            'Authorization': `Bearer ${window.__ADMIN_TOKEN__ || ''}`
                        }
                    });
                    if (!res.ok) throw new Error(`HTTP ${res.status}`);
                    const data = await res.json();
                    this.state.keys = data.keys || [];
                    this.render();
                } catch (err) {
                    console.error('Failed to load API keys:', err);
                    document.getElementById('api-keys-panel').innerHTML = 
                        '<span class="error">Failed to load API keys</span>';
                }
            },
            
            render() {
                const keys = this.state.keys;
                if (!keys.length) {
                    document.getElementById('api-keys-panel').innerHTML = 
                        '<div class="empty-state">No API keys. Click "Create Key" to get started.</div>';
                    return;
                }
                
                let html = '<table class="api-key-table"><thead><tr>';
                html += '<th>Name</th><th>Prefix</th><th>Scopes</th><th>Created</th><th>Last Used</th><th>Status</th><th>Actions</th>';
                html += '</tr></thead><tbody>';
                
                for (const key of keys) {
                    const isExpired = key.expires_at && new Date(key.expires_at) < new Date();
                    const isValid = key.is_valid && !isExpired;
                    const statusClass = key.revoked ? 'status-revoked' : (isExpired ? 'status-expired' : 'status-valid');
                    const statusText = key.revoked ? 'Revoked' : (isExpired ? 'Expired' : 'Valid');
                    
                    const scopesHtml = (key.scopes || []).map(s => 
                        `<span class="scope-badge scope-${s}">${s}</span>`
                    ).join('');
                    
                    const created = new Date(key.created_at).toLocaleString();
                    const lastUsed = key.last_used_at ? new Date(key.last_used_at).toLocaleString() : 'Never';
                    
                    html += `
                        <tr>
                            <td>${this.escapeHtml(key.name)}</td>
                            <td><code>${this.escapeHtml(key.prefix)}</code></td>
                            <td>${scopesHtml}</td>
                            <td>${created}</td>
                            <td>${lastUsed}</td>
                            <td><span class="status-badge ${statusClass}">${statusText}</span></td>
                            <td>
                                <div class="api-key-actions">
                                    ${!key.revoked ? `<button class="btn btn-danger" onclick="ApiKeys.confirmRevoke('${key.prefix}')">Revoke</button>` : ''}
                                </div>
                            </td>
                        </tr>
                    `;
                }
                
                html += '</tbody></table>';
                document.getElementById('api-keys-panel').innerHTML = html;
            },
            
            openCreateModal() {
                const modalHtml = `
                    <div class="modal show" id="create-key-modal" onclick="ApiKeys.closeModal(event)">
                        <div class="modal-content" onclick="event.stopPropagation()">
                            <div class="modal-header">
                                <span class="modal-title">Create API Key</span>
                                <button class="modal-close" onclick="ApiKeys.closeModal()">&times;</button>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Name</label>
                                <input type="text" class="form-input" id="key-name" placeholder="e.g., Production API" required maxlength="100">
                            </div>
                            <div class="form-group">
                                <label class="form-label">Scopes</label>
                                <div class="checkbox-group">
                                    <label class="checkbox-label"><input type="checkbox" value="chat" checked> Chat</label>
                                    <label class="checkbox-label"><input type="checkbox" value="embeddings" checked> Embeddings</label>
                                    <label class="checkbox-label"><input type="checkbox" value="admin"> Admin</label>
                                </div>
                            </div>
                            <div class="form-group">
                                <label class="form-label">Expires In (optional)</label>
                                <input type="text" class="form-input" id="key-expires" placeholder="e.g., 24h, 7d, 30d (leave empty for no expiry)">
                                <small style="color: #999;">Format: number + h/d/m (hours/days/minutes)</small>
                            </div>
                            <div style="display: flex; gap: 8px; justify-content: flex-end;">
                                <button class="btn btn-secondary" onclick="ApiKeys.closeModal()">Cancel</button>
                                <button class="btn btn-primary" onclick="ApiKeys.createKey()">Create Key</button>
                            </div>
                        </div>
                    </div>
                `;
                document.body.insertAdjacentHTML('beforeend', modalHtml);
            },
            
            closeModal(event) {
                if (event && event.target !== event.currentTarget) return;
                const modal = document.getElementById('create-key-modal');
                if (modal) modal.remove();
                
                const keyModal = document.getElementById('show-key-modal');
                if (keyModal) keyModal.remove();
            },
            
            async createKey() {
                const name = document.getElementById('key-name').value.trim();
                if (!name) {
                    alert('Please enter a name');
                    return;
                }
                
                const scopes = Array.from(document.querySelectorAll('#create-key-modal input[type="checkbox"]:checked'))
                    .map(cb => cb.value);
                
                const expiresIn = document.getElementById('key-expires').value.trim() || null;
                
                try {
                    const res = await fetch('/admin/api-keys', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${window.__ADMIN_TOKEN__ || ''}`
                        },
                        body: JSON.stringify({ name, scopes, expires_in: expiresIn })
                    });
                    
                    if (!res.ok) {
                        const err = await res.json();
                        throw new Error(err.detail || `HTTP ${res.status}`);
                    }
                    
                    const data = await res.json();
                    this.closeModal();
                    this.showKeyModal(data);
                    await this.fetch();
                } catch (err) {
                    console.error('Failed to create API key:', err);
                    alert('Failed to create key: ' + err.message);
                }
            },
            
            showKeyModal(data) {
                const modalHtml = `
                    <div class="modal show" id="show-key-modal" onclick="ApiKeys.closeModal(event)">
                        <div class="modal-content" onclick="event.stopPropagation()">
                            <div class="modal-header">
                                <span class="modal-title">API Key Created</span>
                                <button class="modal-close" onclick="ApiKeys.closeModal()">&times;</button>
                            </div>
                            <p style="color: #d32f2f; font-weight: 500;">This is the only time the full key will be shown. Copy it now!</p>
                            <div class="key-display">
                                ${data.key}
                                <button class="btn btn-secondary copy-btn" onclick="ApiKeys.copyKey('${data.key}')">Copy</button>
                            </div>
                            <p><strong>Prefix:</strong> <code>${data.prefix}</code></p>
                            <p><strong>Name:</strong> ${this.escapeHtml(data.name)}</p>
                            <p><strong>Scopes:</strong> ${data.scopes.join(', ')}</p>
                            <p><strong>Created:</strong> ${data.created_at}</p>
                            ${data.expires_at ? `<p><strong>Expires:</strong> ${data.expires_at}</p>` : ''}
                            <div style="display: flex; gap: 8px; justify-content: flex-end; margin-top: 1rem;">
                                <button class="btn btn-primary" onclick="ApiKeys.closeModal()">Done</button>
                            </div>
                        </div>
                    </div>
                `;
                document.body.insertAdjacentHTML('beforeend', modalHtml);
            },
            
            copyKey(key) {
                navigator.clipboard.writeText(key).then(() => {
                    const btn = document.querySelector('.copy-btn');
                    const original = btn.textContent;
                    btn.textContent = 'Copied!';
                    setTimeout(() => btn.textContent = original, 2000);
                });
            },
            
            confirmRevoke(prefix) {
                if (!confirm(`Revoke API key ${prefix}? This action cannot be undone.`)) return;
                this.revokeKey(prefix);
            },
            
            async revokeKey(prefix) {
                try {
                    const res = await fetch(`/admin/api-keys/${encodeURIComponent(prefix)}`, {
                        method: 'DELETE',
                        headers: {
                            'Authorization': `Bearer ${window.__ADMIN_TOKEN__ || ''}`
                        }
                    });
                    
                    if (!res.ok) {
                        const err = await res.json();
                        throw new Error(err.detail || `HTTP ${res.status}`);
                    }
                    
                    await this.fetch();
                } catch (err) {
                    console.error('Failed to revoke API key:', err);
                    alert('Failed to revoke key: ' + err.message);
                }
            },
            
            escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            },
            
            start() {
                this.fetch();
            }
        };
        
        // Expose admin token to JavaScript
        window.__ADMIN_TOKEN__ = "__ADMIN_TOKEN__";
        
        // Start polling on page load
        document.addEventListener('DOMContentLoaded', () => {
            Dashboard.start();
            ApiKeys.start();
        });
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


def _collect_circuit_breaker_state(providers: list[Any]) -> dict[str, Any]:
    """Collect circuit breaker state from all provider routers and their clients."""
    result: dict[str, list[dict[str, Any]]] = {}
    for provider in providers:
        provider_name = getattr(provider, "name", None)
        if not provider_name:
            continue

        clients = getattr(provider, "clients", [])
        if not clients:
            continue

        result[provider_name] = []
        for client in clients:
            cb = getattr(client, "circuit_breaker", None)
            if cb:
                result[provider_name].append(
                    {
                        "api_key_suffix": getattr(client, "api_key", "")[-4:]
                        if getattr(client, "api_key", "")
                        else "unknown",
                        "state": cb.state.value if hasattr(cb.state, "value") else str(cb.state),
                        "failure_count": cb.failure_count,
                        "half_open_calls": getattr(cb, "_half_open_calls", 0),
                        "half_open_successes": getattr(cb, "_half_open_successes", 0),
                    }
                )
    return result


# ---------------------------------------------------------------------------
# Dashboard metrics aggregation
# ---------------------------------------------------------------------------


def _compute_percentiles(values: list[float]) -> dict[str, float]:
    """Compute latency percentiles from a list of float values (seconds)."""
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return {
        "count": n,
        "min": round(sorted_vals[0] * 1000, 2),
        "max": round(sorted_vals[-1] * 1000, 2),
        "mean": round(mean(sorted_vals) * 1000, 2),
        "median": round(sorted_vals[n // 2] * 1000, 2),
        "p50": round(sorted_vals[int(n * 0.50)] * 1000, 2),
        "p95": round(sorted_vals[int(n * 0.95)] * 1000, 2),
        "p99": round(sorted_vals[int(n * 0.99)] * 1000, 2),
    }


def _aggregate_per_provider(
    snapshot: dict[str, Any],
    providers: list[Any],
    provider_health: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """
    Aggregate labeled metrics into per-provider summaries.

    Input: snapshot from MetricsCollector (counters, labeled_counters, timings, labeled_timings)
    Output: {
        "providers": {
            "openai": {
                "request_count": 10234,
                "success_count": 10156,
                "error_count": 78,
                "error_rate": 0.0076,
                "success_rate": 0.9924,
                "latency_stats": {...percentiles...},
                "error_breakdown": {"rate_limit_error": 35, ...}
            },
            ...
        },
        "global": {
            "healthy_providers": 4,
            "total_providers": 4,
            "success_rate": 0.987,
            "latency_stats": {...percentiles...},
            "total_errors": 662
        }
    }
    """
    result: dict[str, Any] = {"providers": {}, "global": {}}

    labeled_counters = snapshot.get("labeled_counters", {})
    labeled_timings = snapshot.get("labeled_timings", {})
    counters = snapshot.get("counters", {})
    timings = snapshot.get("timings", {})

    # -----------------------------------------------------------------------
    # Per-Provider Aggregation
    # -----------------------------------------------------------------------

    provider_names = [getattr(p, "name", None) for p in providers]
    provider_names = [n for n in provider_names if n]

    for provider_name in provider_names:
        label_key = f"provider={provider_name}"

        # Request counts
        request_count = labeled_counters.get("request.count", {}).get(label_key, 0)
        success_count = labeled_counters.get("request.success", {}).get(label_key, 0)
        error_count = labeled_counters.get("request.error", {}).get(label_key, 0)

        # Latency percentiles
        latency_values = labeled_timings.get("request.latency", {}).get(label_key, [])
        latency_stats = _compute_percentiles(latency_values)

        # Error breakdown (per provider)
        error_breakdown = {}
        for label, count in labeled_counters.get("request.error", {}).items():
            if label.startswith(f"{label_key},"):
                # Parse "provider=openai,error_type=rate_limit_error"
                parts = dict(p.split("=") for p in label.split(","))
                error_type = parts.get("error_type", "unknown")
                error_breakdown[error_type] = count

        result["providers"][provider_name] = {
            "request_count": request_count,
            "success_count": success_count,
            "error_count": error_count,
            "error_rate": error_count / request_count if request_count > 0 else 0,
            "success_rate": success_count / request_count if request_count > 0 else 0,
            "latency_stats": latency_stats,
            "error_breakdown": error_breakdown,
        }

    # -----------------------------------------------------------------------
    # Global Aggregation
    # -----------------------------------------------------------------------

    global_request_count = counters.get("request.count", 0)
    global_success_count = counters.get("request.success", 0)
    global_error_count = counters.get("request.error", 0)

    global_latency_values = timings.get("request.latency", [])
    global_latency_stats = _compute_percentiles(global_latency_values)

    # Error breakdown (global)
    global_error_breakdown: dict[str, int] = {}
    for label, count in labeled_counters.get("request.error", {}).items():
        parts = dict(p.split("=") for p in label.split(","))
        error_type = parts.get("error_type", "unknown")
        global_error_breakdown[error_type] = global_error_breakdown.get(error_type, 0) + count

    # Count healthy providers
    if provider_health:
        healthy_count = sum(1 for v in provider_health.values() if v)
        total_providers = len(provider_health)
    else:
        healthy_count = len(provider_names)
        total_providers = len(provider_names)

    result["global"] = {
        "healthy_providers": healthy_count,
        "total_providers": total_providers,
        "request_count": global_request_count,
        "success_count": global_success_count,
        "error_count": global_error_count,
        "success_rate": (
            global_success_count / global_request_count if global_request_count > 0 else 0
        ),
        "error_rate": (
            global_error_count / global_request_count if global_request_count > 0 else 0
        ),
        "latency_stats": global_latency_stats,
        "error_breakdown": global_error_breakdown,
    }

    return result
