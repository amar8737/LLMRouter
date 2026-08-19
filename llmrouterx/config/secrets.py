"""API key resolution for router configuration.

Client keys may be provided in one of three ways inside a provider's client
dict:

* ``api_key``: the literal key (plaintext).
* ``api_key_env``: the name of an environment variable holding the key. When
  key-name scanning is enabled (the default), the environment is also scanned
  for numbered variants of that name (``MY_KEY_1``, ``MY_KEY_2``, ...) and
  every found key becomes its own ClientNode, enabling key rotation.
* ``api_key_file``: a path to a file whose contents (trimmed) are the key.

Precedence is ``api_key`` > ``api_key_env`` > ``api_key_file``.
"""

from __future__ import annotations

import os
import re
from typing import Any

#: ``api_key_env`` values matching a real key never get scanned as a name.
_KEY_PREFIX_RE = re.compile(r"^(sk-|gsk-|sk-ant-|AIza|nvidia|eyJ|lrk_)", re.IGNORECASE)


class KeyResolutionError(ValueError):
    """Raised when a client's API key cannot be resolved."""


def expand_env_key_names(
    base_name: str,
    *,
    scan: bool = True,
    regex: str | None = None,
) -> list[str]:
    """Expand an env-var key name into the list of environment variables to read.

    The base variable is always considered first. When ``scan`` is enabled the
    numbered variants ``<base>_1``, ``<base>_2``, ... are appended in order,
    stopping at the first gap (a missing variable ends the scan). This lets a
    user provide a single key *name* such as ``OPEN_AI_KEY`` and have the
    router pick up ``OPEN_AI_KEY``, ``OPEN_AI_KEY_1`` and ``OPEN_AI_KEY_2``
    automatically — but not ``OPEN_AI_KEY_3`` when it is unset.

    ``regex`` (when given) overrides the prefix scan: every environment
    variable matching the pattern is returned, sorted so that numbered
    suffixes sort numerically (``OPEN_AI_KEY_2`` before ``OPEN_AI_KEY_10``).

    Returns the variable *names* to read; resolution of their values happens
    in :func:`resolve_keys`.
    """
    base_name = (base_name or "").strip()
    if not base_name:
        raise KeyResolutionError("api_key_env must be a non-empty variable name.")

    if regex:
        pattern = re.compile(regex)
        found = sorted(
            (name for name in os.environ if pattern.fullmatch(name)),
            key=lambda name: [
                int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)
            ],
        )
        if not found:
            raise KeyResolutionError(
                f"No environment variable matched api_key_env_regex '{regex}'."
            )
        return found

    names = [base_name]
    if scan:
        index = 1
        while True:
            candidate = f"{base_name}_{index}"
            if candidate not in os.environ:
                break
            names.append(candidate)
            index += 1

    if not any(name in os.environ for name in names):
        raise KeyResolutionError(
            f"Environment variable '{base_name}' referenced by api_key_env is not set."
        )
    return names


def _looks_like_literal_key(value: str) -> bool:
    """Whether a string is probably a real API key rather than an env-var name."""
    return bool(_KEY_PREFIX_RE.match(value))


def resolve_keys(client: dict[str, Any]) -> list[str]:
    """Resolve a single client dict to one or more API key strings.

    * ``api_key`` / ``api_key_file`` resolve to exactly one key.
    * ``api_key_env`` resolves to one key per found variable: the base name
      plus every numbered variant when scanning is enabled (or every variable
      matched by ``api_key_env_regex``). Scanning is on by default and can be
      disabled with ``"api_key_env_scan": false``.
    """
    if client.get("api_key"):
        return [str(client["api_key"])]

    if client.get("api_key_file"):
        key_path = client.get("api_key_file")
        if not isinstance(key_path, (str, os.PathLike)):
            raise KeyResolutionError("api_key_file must be a path string.")
        try:
            with open(key_path, encoding="utf-8") as handle:
                key = handle.read().strip()
        except OSError as exc:
            raise KeyResolutionError(f"Could not read api_key_file '{key_path}': {exc}") from exc
        if not key:
            raise KeyResolutionError(f"api_key_file '{key_path}' is empty.")
        return [key]

    env_name = client.get("api_key_env")
    if env_name:
        scan = bool(client.get("api_key_env_scan", True))
        names = expand_env_key_names(
            env_name,
            scan=scan,
            regex=client.get("api_key_env_regex"),
        )
        values: list[str] = []
        for name in names:
            value = os.getenv(name)
            if value is None:
                if name == env_name and names != [env_name]:
                    continue
                raise KeyResolutionError(
                    f"Environment variable '{name}' referenced by api_key_env is not set."
                )
            values.append(value)
        if not values:
            raise KeyResolutionError(
                f"Environment variable '{env_name}' referenced by api_key_env is not set."
            )
        return values

    raise KeyResolutionError(
        "Client is missing an API key. Provide 'api_key', 'api_key_env', or 'api_key_file'."
    )


def resolve_key(client: dict[str, Any]) -> str:
    """Resolve a single client dict to its API key string.

    Mirrors :func:`resolve_keys` for callers that expect exactly one key.
    """
    keys = resolve_keys(client)
    return keys[0]


def has_key_source(client: dict[str, Any]) -> bool:
    """Whether a client dict declares any API key source."""
    return bool(client.get("api_key") or client.get("api_key_env") or client.get("api_key_file"))
