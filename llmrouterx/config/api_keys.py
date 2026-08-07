"""API Key management with SQLite persistence.

Provides database-backed API key storage with:
- Key generation with lrk_ prefix
- Scoped permissions (chat, embeddings, admin)
- Expiration support
- Revocation
- Last-used tracking
"""

from __future__ import annotations

import atexit
import os
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .secrets import KeyResolutionError

# Key configuration
KEY_PREFIX = "lrk_"
KEY_BYTES = 32  # 32 bytes = 43 chars base64url

# Scopes
VALID_SCOPES = ("chat", "embeddings", "admin")


@dataclass(slots=True)
class APIKey:
    """Represents an API key record."""

    id: int
    key_hash: str  # SHA256 hash of the full key
    prefix: str  # First 8 chars for display (lrk_abcd)
    name: str
    scopes: tuple[str, ...]
    created_at: datetime
    expires_at: datetime | None
    last_used_at: datetime | None
    revoked: bool

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.utcnow() >= self.expires_at

    @property
    def is_valid(self) -> bool:
        return not self.revoked and not self.is_expired

    @property
    def scopes_list(self) -> list[str]:
        return list(self.scopes)


class APIKeyDatabase:
    """SQLite-backed API key store."""

    def __init__(self, db_path: str | os.PathLike[str]):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection with WAL mode."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None,  # Autocommit
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_hash TEXT NOT NULL UNIQUE,
                    prefix TEXT NOT NULL,
                    name TEXT NOT NULL,
                    scopes TEXT NOT NULL,  -- JSON array
                    created_at TEXT NOT NULL,  -- ISO 8601
                    expires_at TEXT,  -- ISO 8601, nullable
                    last_used_at TEXT,  -- ISO 8601, nullable
                    revoked INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(prefix)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_revoked ON api_keys(revoked)
            """)

    def close(self) -> None:
        """Close database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    @contextmanager
    def _transaction(self):
        """Context manager for database transactions with retry on lock."""
        conn = self._get_connection()
        max_retries = 3
        for attempt in range(max_retries):
            try:
                yield conn
                conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    conn.rollback()
                    time.sleep(0.1 * (attempt + 1))  # exponential backoff
                    continue
                conn.rollback()
                raise

    @staticmethod
    def _hash_key(key: str) -> str:
        """Generate SHA256 hash of the key."""
        import hashlib

        return hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def _generate_key() -> str:
        """Generate a new API key with lrk_ prefix."""
        return KEY_PREFIX + secrets.token_urlsafe(KEY_BYTES)

    @staticmethod
    def _get_prefix(key: str) -> str:
        """Get display prefix (first 8 chars after lrk_)."""
        return key[:12]  # lrk_ + 8 chars

    def create_key(
        self,
        name: str,
        scopes: tuple[str, ...] = ("chat", "embeddings"),
        expires_in: str | None = None,  # e.g., "24h", "7d", "30d"
    ) -> tuple[str, APIKey]:
        """Create a new API key. Returns (full_key, APIKey_record)."""
        # Validate scopes
        for scope in scopes:
            if scope not in VALID_SCOPES:
                raise ValueError(f"Invalid scope: {scope}. Valid: {VALID_SCOPES}")

        # Parse expiry
        expires_at = None
        if expires_in:
            expires_at = self._parse_expiry(expires_in)

        # Generate key
        full_key = self._generate_key()
        key_hash = self._hash_key(full_key)
        prefix = self._get_prefix(full_key)
        now = datetime.utcnow()

        import json

        scopes_json = json.dumps(list(scopes))

        with self._transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO api_keys
                (key_hash, prefix, name, scopes, created_at, expires_at, last_used_at, revoked)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    key_hash,
                    prefix,
                    name,
                    scopes_json,
                    now.isoformat(),
                    expires_at.isoformat() if expires_at else None,
                    None,
                ),
            )
            key_id = cursor.lastrowid

            record = APIKey(
                id=key_id,
                key_hash=key_hash,
                prefix=prefix,
                name=name,
                scopes=scopes,
                created_at=now,
                expires_at=expires_at,
                last_used_at=None,
                revoked=False,
            )

            return full_key, record

    def _parse_expiry(self, expires_in: str) -> datetime:
        """Parse expiry string like '24h', '7d', '30d'."""
        expires_in = expires_in.strip().lower()
        if expires_in.endswith("h"):
            hours = int(expires_in[:-1])
            return datetime.utcnow() + timedelta(hours=hours)
        elif expires_in.endswith("d"):
            days = int(expires_in[:-1])
            return datetime.utcnow() + timedelta(days=days)
        elif expires_in.endswith("m"):
            minutes = int(expires_in[:-1])
            return datetime.utcnow() + timedelta(minutes=minutes)
        else:
            raise ValueError(f"Invalid expiry format: {expires_in}. Use '24h', '7d', etc.")

    def get_key(self, key_hash: str) -> APIKey | None:
        """Get API key record by hash."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_key_by_prefix(self, prefix: str) -> APIKey | None:
        """Get API key record by prefix (for display/revocation)."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM api_keys WHERE prefix = ?", (prefix,)).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_keys(self, include_revoked: bool = False) -> list[APIKey]:
        """List all API keys."""
        with self._get_connection() as conn:
            if include_revoked:
                rows = conn.execute("SELECT * FROM api_keys ORDER BY created_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM api_keys WHERE revoked = 0 ORDER BY created_at DESC"
                ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def revoke_key(self, prefix: str) -> bool:
        """Revoke an API key by prefix. Returns True if found and revoked."""
        with self._transaction() as conn:
            cursor = conn.execute("UPDATE api_keys SET revoked = 1 WHERE prefix = ?", (prefix,))
            return cursor.rowcount > 0

    def update_last_used(self, key_hash: str) -> None:
        """Update last_used_at timestamp for a key."""
        now = datetime.utcnow().isoformat()
        with self._transaction() as conn:
            conn.execute("UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?", (now, key_hash))

    def validate_key(self, full_key: str) -> APIKey | None:
        """Validate a full API key and return record if valid."""
        if not full_key.startswith(KEY_PREFIX):
            return None

        key_hash = self._hash_key(full_key)
        record = self.get_key(key_hash)

        if record is None:
            return None
        if not record.is_valid:
            return None

        # Update last used
        self.update_last_used(key_hash)

        # Return updated record
        return self.get_key(key_hash)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> APIKey:
        """Convert database row to APIKey record."""
        import json

        return APIKey(
            id=row["id"],
            key_hash=row["key_hash"],
            prefix=row["prefix"],
            name=row["name"],
            scopes=tuple(json.loads(row["scopes"])),
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            last_used_at=datetime.fromisoformat(row["last_used_at"])
            if row["last_used_at"]
            else None,
            revoked=bool(row["revoked"]),
        )


# Global database instance
_db: APIKeyDatabase | None = None
_db_lock = threading.Lock()


def get_api_key_db(db_path: str | os.PathLike[str] | None = None) -> APIKeyDatabase:
    """Get or create the global API key database."""
    global _db
    with _db_lock:
        if _db is None:
            if db_path is None:
                # Default to working directory
                db_path = Path.cwd() / "api_keys.db"
            _db = APIKeyDatabase(db_path)
            # Register cleanup
            atexit.register(_db.close)
        return _db


def set_api_key_db(db: APIKeyDatabase) -> None:
    """Set the global API key database (for testing)."""
    global _db
    with _db_lock:
        if _db is not None:
            _db.close()
        _db = db


def create_api_key(
    name: str,
    scopes: tuple[str, ...] = ("chat", "embeddings"),
    expires_in: str | None = None,
    db_path: str | None = None,
) -> tuple[str, APIKey]:
    """Convenience function to create an API key."""
    db = get_api_key_db(db_path)
    return db.create_key(name, scopes, expires_in)


def validate_api_key(full_key: str, db_path: str | None = None) -> APIKey | None:
    """Validate an API key against the database."""
    db = get_api_key_db(db_path)
    return db.validate_key(full_key)


def resolve_api_key(
    client: dict[str, Any],
    db_path: str | None = None,
) -> str:
    """Resolve API key from client config, supporting database keys.

    Supports:
    - api_key: literal key (existing)
    - api_key_env: environment variable (existing)
    - api_key_file: file path (existing)
    - api_key_db: boolean, use database keys (NEW)
    """
    # First try existing resolution methods
    try:
        from .secrets import resolve_key as resolve_key_legacy

        return resolve_key_legacy(client)
    except KeyResolutionError:
        pass

    # Try database lookup if enabled
    if client.get("api_key_db"):
        db_path_ = client.get("api_key_db_path")
        get_api_key_db(db_path_)  # Initialize DB
        # For database mode, we expect api_key to be the full key
        if client.get("api_key"):
            record = validate_api_key(client["api_key"], db_path_)
            if record and record.is_valid:
                return record.key_hash  # Return hash for validation
            raise KeyResolutionError("Invalid or revoked API key")
        raise KeyResolutionError("api_key_db requires api_key field with the full key")

    raise KeyResolutionError(
        "Client is missing an API key. Provide 'api_key', 'api_key_env', "
        "'api_key_file', or enable 'api_key_db'."
    )
