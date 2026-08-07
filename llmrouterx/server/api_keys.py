"""API Key management endpoints (admin only)."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Request, status

from llmrouterx.config.api_keys import (
    KEY_PREFIX,
    VALID_SCOPES,
    create_api_key,
    get_api_key_db,
)


class APIKeyCreateRequest:
    """Request model for creating an API key."""

    def __init__(
        self,
        name: str,
        scopes: list[str] | None = None,
        expires_in: str | None = None,
    ):
        self.name = name
        self.scopes = scopes or ["chat", "embeddings"]
        self.expires_in = expires_in


class APIKeyListResponse:
    """Response model for listing API keys."""

    def __init__(self, keys: list[dict[str, Any]]):
        self.keys = keys


class APIKeyCreateResponse:
    """Response model for creating an API key."""

    def __init__(
        self,
        key: str,
        prefix: str,
        name: str,
        scopes: list[str],
        created_at: str,
        expires_at: str | None = None,
    ):
        self.key = key
        self.prefix = prefix
        self.name = name
        self.scopes = scopes
        self.created_at = created_at
        self.expires_at = expires_at


class APIKeyRevokeResponse:
    """Response model for revoking an API key."""

    def __init__(self, revoked: bool, prefix: str):
        self.revoked = revoked
        self.prefix = prefix


def create_list_keys_endpoint(admin_guard: Any):
    """Create list API keys endpoint."""

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

    return Depends(admin_guard)(list_api_keys)


def create_create_key_endpoint(admin_guard: Any):
    """Create create API key endpoint."""

    async def create_api_key_endpoint(
        request: Request,
    ) -> APIKeyCreateResponse:
        """Create a new API key (admin only)."""
        body = await request.json()
        # Validate scopes
        scopes = body.get("scopes", ["chat", "embeddings"])
        for scope in scopes:
            if scope not in VALID_SCOPES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid scope: {scope}. Valid scopes: {VALID_SCOPES}",
                )

        full_key, record = create_api_key(
            name=body["name"],
            scopes=tuple(scopes),
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

    return Depends(admin_guard)(create_api_key_endpoint)


def create_revoke_key_endpoint(admin_guard: Any):
    """Create revoke API key endpoint."""

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

    return Depends(admin_guard)(revoke_api_key)


def register_api_key_routes(app: Any, admin_guard: Any) -> None:
    """Register all API key management routes."""
    app.get("/admin/api-keys", response_model=APIKeyListResponse)(
        create_list_keys_endpoint(admin_guard)
    )
    app.post("/admin/api-keys", response_model=APIKeyCreateResponse)(
        create_create_key_endpoint(admin_guard)
    )
    app.delete("/admin/api-keys/{prefix}", response_model=APIKeyRevokeResponse)(
        create_revoke_key_endpoint(admin_guard)
    )
