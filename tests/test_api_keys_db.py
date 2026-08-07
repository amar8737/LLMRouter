"""Tests for API key database and management."""

from __future__ import annotations

import os
import tempfile
import threading

import pytest

from llmrouterx.config.api_keys import (
    KEY_PREFIX,
    VALID_SCOPES,
    APIKeyDatabase,
    create_api_key,
    get_api_key_db,
    validate_api_key,
)


class TestAPIKeyDatabase:
    """Test the APIKeyDatabase class."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        yield path
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def db(self, temp_db_path):
        """Create a database instance."""
        db = APIKeyDatabase(temp_db_path)
        yield db
        db.close()

    def test_create_key(self, db):
        """Test creating a new API key."""
        full_key, record = db.create_key("Test Key", ("chat", "embeddings"))

        assert full_key.startswith(KEY_PREFIX)
        assert len(full_key) > len(KEY_PREFIX)
        assert record.name == "Test Key"
        assert record.scopes == ("chat", "embeddings")
        assert record.revoked is False
        assert record.is_valid is True
        assert record.prefix == full_key[:12]

    def test_create_key_with_expiry(self, db):
        """Test creating a key with expiration."""
        _full_key, record = db.create_key("Expiring Key", ("chat",), expires_in="1h")

        assert record.expires_at is not None
        assert record.is_valid is True

    def test_create_key_invalid_scope(self, db):
        """Test creating a key with invalid scope raises error."""
        with pytest.raises(ValueError, match="Invalid scope"):
            db.create_key("Test", ("invalid_scope",))

    def test_get_key_by_hash(self, db):
        """Test retrieving a key by hash."""
        full_key, record = db.create_key("Test Key")
        key_hash = db._hash_key(full_key)

        retrieved = db.get_key(key_hash)
        assert retrieved is not None
        assert retrieved.id == record.id
        assert retrieved.prefix == record.prefix

    def test_get_key_by_prefix(self, db):
        """Test retrieving a key by prefix."""
        _full_key, record = db.create_key("Test Key")

        retrieved = db.get_key_by_prefix(record.prefix)
        assert retrieved is not None
        assert retrieved.id == record.id

    def test_list_keys(self, db):
        """Test listing keys."""
        db.create_key("Key 1")
        db.create_key("Key 2")
        db.create_key("Key 3")

        keys = db.list_keys()
        assert len(keys) == 3
        # Should be ordered by created_at DESC
        assert keys[0].name == "Key 3"

    def test_list_keys_excludes_revoked_by_default(self, db):
        """Test list_keys excludes revoked by default."""
        _full_key1, _ = db.create_key("Active Key")
        _full_key2, record2 = db.create_key("Revoked Key")
        db.revoke_key(record2.prefix)

        keys = db.list_keys()
        assert len(keys) == 1
        assert keys[0].name == "Active Key"

    def test_list_keys_include_revoked(self, db):
        """Test list_keys with include_revoked=True."""
        db.create_key("Active Key")
        _full_key2, record2 = db.create_key("Revoked Key")
        db.revoke_key(record2.prefix)

        keys = db.list_keys(include_revoked=True)
        assert len(keys) == 2

    def test_revoke_key(self, db):
        """Test revoking a key."""
        _full_key, record = db.create_key("Test Key")
        assert record.revoked is False

        result = db.revoke_key(record.prefix)
        assert result is True

        # Verify revoked
        retrieved = db.get_key_by_prefix(record.prefix)
        assert retrieved.revoked is True
        assert retrieved.is_valid is False

    def test_revoke_nonexistent_key(self, db):
        """Test revoking a non-existent key returns False."""
        result = db.revoke_key("lrk_nonexist")
        assert result is False

    def test_validate_key_success(self, db):
        """Test validating a valid key."""
        full_key, _ = db.create_key("Test Key")

        validated = db.validate_key(full_key)
        assert validated is not None
        assert validated.is_valid is True

    def test_validate_key_invalid_prefix(self, db):
        """Test validating a key with wrong prefix."""
        result = db.validate_key("invalid_key")
        assert result is None

    def test_validate_key_revoked(self, db):
        """Test validating a revoked key returns None."""
        full_key, record = db.create_key("Test Key")
        db.revoke_key(record.prefix)

        result = db.validate_key(full_key)
        assert result is None

    def test_validate_key_expired(self, db):
        """Test validating an expired key returns None."""
        from datetime import datetime, timedelta

        from llmrouterx.config.api_keys import APIKey

        # Create a key record that's already expired
        expired_record = APIKey(
            id=1,
            key_hash="hash",
            prefix="lrk_test",
            name="Test",
            scopes=("chat",),
            created_at=datetime.utcnow() - timedelta(hours=2),
            expires_at=datetime.utcnow() - timedelta(hours=1),
            last_used_at=None,
            revoked=False,
        )
        assert expired_record.is_expired is True
        assert expired_record.is_valid is False

    def test_validate_key_updates_last_used(self, db):
        """Test that validate_key updates last_used_at."""
        full_key, record = db.create_key("Test Key")
        assert record.last_used_at is None

        validated = db.validate_key(full_key)
        assert validated.last_used_at is not None

    def test_parse_expiry_formats(self, db):
        """Test parsing various expiry formats."""
        assert db._parse_expiry("24h") is not None
        assert db._parse_expiry("7d") is not None
        assert db._parse_expiry("30m") is not None
        assert db._parse_expiry("1H") is not None  # Case insensitive
        assert db._parse_expiry("1D") is not None

    def test_parse_expiry_invalid(self, db):
        """Test parsing invalid expiry format."""
        with pytest.raises(ValueError):
            db._parse_expiry("invalid")

    def test_concurrent_access(self, temp_db_path):
        """Test thread-safe database access."""
        errors = []

        def create_keys():
            # Each thread gets its own database instance
            thread_db = APIKeyDatabase(temp_db_path)
            try:
                for i in range(10):
                    thread_db.create_key(f"Key {i}")
            except Exception as e:
                errors.append(e)
            finally:
                thread_db.close()

        threads = [threading.Thread(target=create_keys) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Verify total keys created
        db = APIKeyDatabase(temp_db_path)
        keys = db.list_keys()
        assert len(keys) == 50
        db.close()


class TestGlobalDatabase:
    """Test the global database functions."""

    def test_get_api_key_db_singleton(self):
        """Test that get_api_key_db returns singleton."""
        db1 = get_api_key_db(":memory:")
        db2 = get_api_key_db()
        assert db1 is db2

    def test_create_api_key_convenience(self):
        """Test create_api_key convenience function."""
        full_key, record = create_api_key("Convenience Key", ("chat",), expires_in="1h")

        assert full_key.startswith(KEY_PREFIX)
        assert record.name == "Convenience Key"

    def test_validate_api_key_convenience(self):
        """Test validate_api_key convenience function."""
        full_key, _ = create_api_key("Validate Test")

        validated = validate_api_key(full_key)
        assert validated is not None
        assert validated.is_valid is True

    def test_valid_scopes_constant(self):
        """Test VALID_SCOPES contains expected values."""
        assert "chat" in VALID_SCOPES
        assert "embeddings" in VALID_SCOPES
        assert "admin" in VALID_SCOPES


class TestAPIKeyRecord:
    """Test the APIKey dataclass properties."""

    def test_is_expired_no_expiry(self):
        """Test is_expired returns False when no expiry."""
        from datetime import datetime

        from llmrouterx.config.api_keys import APIKey

        record = APIKey(
            id=1,
            key_hash="hash",
            prefix="lrk_test",
            name="Test",
            scopes=("chat",),
            created_at=datetime.utcnow(),
            expires_at=None,
            last_used_at=None,
            revoked=False,
        )
        assert record.is_expired is False

    def test_is_expired_future(self):
        """Test is_expired returns False for future expiry."""
        from datetime import datetime, timedelta

        from llmrouterx.config.api_keys import APIKey

        record = APIKey(
            id=1,
            key_hash="hash",
            prefix="lrk_test",
            name="Test",
            scopes=("chat",),
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=1),
            last_used_at=None,
            revoked=False,
        )
        assert record.is_expired is False

    def test_is_expired_past(self):
        """Test is_expired returns True for past expiry."""
        from datetime import datetime, timedelta

        from llmrouterx.config.api_keys import APIKey

        record = APIKey(
            id=1,
            key_hash="hash",
            prefix="lrk_test",
            name="Test",
            scopes=("chat",),
            created_at=datetime.utcnow() - timedelta(hours=2),
            expires_at=datetime.utcnow() - timedelta(hours=1),
            last_used_at=None,
            revoked=False,
        )
        assert record.is_expired is True

    def test_is_valid_combinations(self):
        """Test is_valid property combinations."""
        from datetime import datetime, timedelta

        from llmrouterx.config.api_keys import APIKey

        # Valid: not revoked, not expired
        record = APIKey(
            id=1,
            key_hash="hash",
            prefix="lrk_test",
            name="Test",
            scopes=("chat",),
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=1),
            last_used_at=None,
            revoked=False,
        )
        assert record.is_valid is True

        # Invalid: revoked
        record.revoked = True
        assert record.is_valid is False

        # Invalid: expired
        record.revoked = False
        record.expires_at = datetime.utcnow() - timedelta(hours=1)
        assert record.is_valid is False

    def test_scopes_list_property(self):
        """Test scopes_list property."""
        from datetime import datetime

        from llmrouterx.config.api_keys import APIKey

        record = APIKey(
            id=1,
            key_hash="hash",
            prefix="lrk_test",
            name="Test",
            scopes=("chat", "embeddings", "admin"),
            created_at=datetime.utcnow(),
            expires_at=None,
            last_used_at=None,
            revoked=False,
        )
        assert record.scopes_list == ["chat", "embeddings", "admin"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
