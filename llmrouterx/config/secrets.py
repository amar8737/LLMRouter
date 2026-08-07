"""API key resolution for router configuration.

Client keys may be provided in one of three ways inside a provider's client
dict:

* ``api_key``: the literal key (plaintext).
* ``api_key_env``: the name of an environment variable holding the key.
* ``api_key_file``: a path to a file whose contents (trimmed) are the key.

Precedence is ``api_key`` > ``api_key_env`` > ``api_key_file``.
"""

from __future__ import annotations

import os
from typing import Any


class KeyResolutionError(ValueError):
    """Raised when a client's API key cannot be resolved."""


def resolve_key(client: dict[str, Any]) -> str:
    """Resolve a single client dict to its API key string."""
    if client.get("api_key"):
        return str(client["api_key"])

    env_name = client.get("api_key_env")
    if env_name:
        value = os.getenv(env_name)
        if value is None:
            raise KeyResolutionError(
                f"Environment variable '{env_name}' referenced by api_key_env is not set."
            )
        return value

    key_path = client.get("api_key_file")
    if key_path:
        try:
            with open(key_path, encoding="utf-8") as handle:
                key = handle.read().strip()
        except OSError as exc:
            raise KeyResolutionError(
                f"Could not read api_key_file '{key_path}': {exc}"
            ) from exc
        if not key:
            raise KeyResolutionError(f"api_key_file '{key_path}' is empty.")
        return key

    raise KeyResolutionError(
        "Client is missing an API key. Provide 'api_key', 'api_key_env', "
        "or 'api_key_file'."
    )


def has_key_source(client: dict[str, Any]) -> bool:
    """Whether a client dict declares any API key source."""
    return bool(
        client.get("api_key") or client.get("api_key_env") or client.get("api_key_file")
    )
