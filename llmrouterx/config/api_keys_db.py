"""API Key database models and storage (SQLite)."""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from llmrouterx.utils.masking import mask_api_key

DB_PATH = Path("api_keys.db")
DEFAULT_SCOPES = ["chat", "embeddings"]
ADMIN_SCOPE = "admin"
ALL_SCOPES = [*DEFAULT_SCOPES, ADMIN_SCOPE]
KEY_PREFIX = "lrk_"
KEY_LENGTH = 32


@dataclass
class APIKey:
    id: str
    name: str
    prefix: str
    key_hash: str
    created_at: str
    last_used: str | None
    expires_at: str | None
    scopes: list[str]
    revoked: bool

    def to_dict(self, include_key: bool = False) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "prefix": self.prefix,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "expires_at": self.expires_at,
            "scopes": self.scopes,
            "revoked": self.revoked,
            **({"key": mask_api_key(self.prefix + self.id)} if include_key else {}),
        }

    @property
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        return datetime.fromisoformat(self.expires_at) < datetime.utcnow()


def _get_db_path() -> Path:
    return Path(os.getenv("LLMROUTER_API_KEYS_DB", "api_keys.db"))


@contextmanager
def _connect() -> Any:
    conn = sqlite3.connect(_get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                prefix TEXT NOT NULL UNIQUE,
                key_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_used TEXT,
                expires_at TEXT,
                scopes TEXT NOT NULL,
                revoked INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(prefix)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_api_keys_revoked ON api_keys(revoked)
        """)
        conn.commit()


def _hash_key(key: str) -> str:
    import hashlib

    return hashlib.sha256(key.encode()).hexdigest()


def generate_key() -> tuple[str, str]:
    """Generate a new API key. Returns (full_key, prefix)."""
    random_part = secrets.token_urlsafe(KEY_LENGTH)
    full_key = f"{KEY_PREFIX}{random_part}"
    prefix = full_key[:8]
    return full_key, prefix


def create_key(
    name: str,
    scopes: list[str] | None = None,
    expires_in_seconds: int | None = None,
) -> tuple[APIKey, str]:
    """Create a new API key. Returns (APIKey object, full_key)."""
    _init_db()

    full_key, prefix = generate_key()
    key_hash = _hash_key(full_key)
    now = datetime.utcnow().isoformat()
    expires_at = (
        datetime.utcfromtimestamp(time.time() + expires_in_seconds).isoformat()
        if expires_in_seconds
        else None
    )
    scope_list = scopes or DEFAULT_SCOPES

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO api_keys
            (id, name, prefix, key_hash, created_at, expires_at, scopes, revoked)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (full_key, name, prefix, key_hash, now, expires_at, json.dumps(scope_list)),
        )
        conn.commit()

    key = APIKey(
        id=full_key,
        name=name,
        prefix=prefix,
        key_hash=key_hash,
        created_at=now,
        last_used=None,
        expires_at=expires_at,
        scopes=scope_list,
        revoked=False,
    )
    return key, full_key


def list_keys(include_revoked: bool = False) -> list[APIKey]:
    _init_db()
    with _connect() as conn:
        query = "SELECT * FROM api_keys"
        if not include_revoked:
            query += " WHERE revoked = 0"
        query += " ORDER BY created_at DESC"
        rows = conn.execute(query).fetchall()

    return [
        APIKey(
            id=row["id"],
            name=row["name"],
            prefix=row["prefix"],
            key_hash=row["key_hash"],
            created_at=row["created_at"],
            last_used=row["last_used"],
            expires_at=row["expires_at"],
            scopes=json.loads(row["scopes"]),
            revoked=bool(row["revoked"]),
        )
        for row in rows
    ]


def get_key_by_prefix(prefix: str) -> APIKey | None:
    _init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM api_keys WHERE prefix = ? AND revoked = 0", (prefix,)
        ).fetchone()

    if not row:
        return None

    return APIKey(
        id=row["id"],
        name=row["name"],
        prefix=row["prefix"],
        key_hash=row["key_hash"],
        created_at=row["created_at"],
        last_used=row["last_used"],
        expires_at=row["expires_at"],
        scopes=json.loads(row["scopes"]),
        revoked=bool(row["revoked"]),
    )


def validate_key(full_key: str) -> APIKey | None:
    """Validate a full API key. Returns APIKey if valid, None otherwise."""
    if not full_key.startswith(KEY_PREFIX):
        return None

    prefix = full_key[:8]
    key = get_key_by_prefix(prefix)
    if not key:
        return None

    if key.is_expired:
        return None

    if key.key_hash != _hash_key(full_key):
        return None

    return key


def revoke_key(prefix: str) -> bool:
    _init_db()
    with _connect() as conn:
        cursor = conn.execute("UPDATE api_keys SET revoked = 1 WHERE prefix = ?", (prefix,))
        conn.commit()
        return cursor.rowcount > 0


def update_last_used(prefix: str) -> None:
    _init_db()
    with _connect() as conn:
        conn.execute(
            "UPDATE api_keys SET last_used = ? WHERE prefix = ?",
            (datetime.utcnow().isoformat(), prefix),
        )
        conn.commit()


import os  # noqa: E402
