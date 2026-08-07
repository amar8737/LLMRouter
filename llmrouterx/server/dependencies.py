"""Authentication dependencies and guards."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from llmrouterx.config.api_keys import validate_api_key


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Inject a request ID and emit a per-request access log line."""

    def __init__(self, app, enabled: bool = True) -> None:
        super().__init__(app)
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next):
        import time
        import uuid

        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            import logging

            logger = logging.getLogger("llmrouterx.server")
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
            import logging

            logger = logging.getLogger("llmrouterx.server")
            logger.info(
                "%s %s %d %.1fms [req_id=%s]",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
                request_id,
            )
        return response


def make_bearer_guard(
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


def get_admin_guard(admin_token: str | None) -> Any:
    """Get admin authentication guard."""
    return make_bearer_guard(admin_token)


def get_api_key_guard(
    api_keys: Sequence[str] | None,
    api_keys_db_path: str | None = None,
) -> Any:
    """Get API key authentication guard (checks both env keys and database)."""
    return make_bearer_guard(
        list(api_keys) if api_keys else None,
        check_db=True,
        db_path=api_keys_db_path,
    )
