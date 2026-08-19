from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is optional
    yaml = None


def _parse_bool(value: Any, default: bool = True) -> bool:
    """
    Parse a bool from a JSON value or an env/var string.

    Accepts ``True``/``False`` directly, and the strings ``"true"``/``"false"``
    (case-insensitive). ``None`` yields ``default`` so the field stays optional.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


@dataclass(slots=True)
class RouterConfig:
    """
    Configuration for LLMRouter.
    """

    providers: list[Any] = field(default_factory=list)

    scheduler: Any | None = None

    retry: Any | None = None

    middleware: list[Any] = field(default_factory=list)

    timeout: float = 60.0

    max_retries: int = 3

    max_concurrent_per_key: int = 100

    max_concurrent_requests: int | None = None

    total_timeout: float | None = None

    enable_circuit_breaker: bool = True

    circuit_breaker_threshold: int = 5

    circuit_breaker_reset_timeout: float = 30.0

    @classmethod
    def from_env(cls) -> RouterConfig:
        return cls(
            timeout=float(
                os.getenv(
                    "LLMROUTER_TIMEOUT",
                    "60",
                )
            ),
            max_retries=int(
                os.getenv(
                    "LLMROUTER_MAX_RETRIES",
                    "3",
                )
            ),
            max_concurrent_per_key=int(
                os.getenv(
                    "LLMROUTER_MAX_CONCURRENT",
                    "100",
                )
            ),
            max_concurrent_requests=(
                int(os.getenv("LLMROUTER_MAX_CONCURRENT_REQUESTS", "0")) or None
            ),
            total_timeout=(float(os.getenv("LLMROUTER_TOTAL_TIMEOUT", "0")) or None),
            enable_circuit_breaker=_parse_bool(os.getenv("LLMROUTER_CIRCUIT_BREAKER", "true")),
            circuit_breaker_threshold=int(os.getenv("LLMROUTER_CB_THRESHOLD", "5")),
            circuit_breaker_reset_timeout=float(os.getenv("LLMROUTER_CB_RESET_TIMEOUT", "30")),
        )

    def resolve_keys(self) -> RouterConfig:
        """Return a copy with every client's API key materialised.

        Each client dict's ``api_key``/``api_key_env``/``api_key_file`` source
        is replaced with the resolved literal key under ``api_key``. When
        ``api_key_env`` names a key prefix (e.g. ``OPEN_AI_KEY``) and key-name
        scanning is enabled, every numbered variant found in the environment
        (``OPEN_AI_KEY_1``, ``OPEN_AI_KEY_2``, ...) expands the client dict
        into one client per key, so each becomes its own ClientNode and the
        provider rotates across them.
        """
        from .secrets import resolve_keys

        clean: list[Any] = []
        for provider in self.providers:
            resolved_provider = dict(provider)
            expanded: list[Any] = []
            for client in provider.get("clients", []):
                for key in resolve_keys(client):
                    expanded.append({**client, "api_key": key})
            resolved_provider["clients"] = expanded
            clean.append(resolved_provider)
        return self.copy(providers=clean)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the numeric/string settings."""
        return {
            "providers": self.providers,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "max_concurrent_per_key": self.max_concurrent_per_key,
            "max_concurrent_requests": self.max_concurrent_requests,
            "total_timeout": self.total_timeout,
            "enable_circuit_breaker": self.enable_circuit_breaker,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "circuit_breaker_reset_timeout": self.circuit_breaker_reset_timeout,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RouterConfig:
        """Build a config from a dict, falling back to defaults for missing keys."""
        known = {
            "providers",
            "timeout",
            "max_retries",
            "max_concurrent_per_key",
            "max_concurrent_requests",
            "total_timeout",
            "enable_circuit_breaker",
            "circuit_breaker_threshold",
            "circuit_breaker_reset_timeout",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg = cls(
            providers=list(data.get("providers", [])),
            timeout=float(data.get("timeout", 60.0)),
            max_retries=int(data.get("max_retries", 3)),
            max_concurrent_per_key=int(data.get("max_concurrent_per_key", 100)),
            max_concurrent_requests=(
                int(data["max_concurrent_requests"])
                if data.get("max_concurrent_requests")
                else None
            ),
            total_timeout=(float(data["total_timeout"]) if data.get("total_timeout") else None),
            enable_circuit_breaker=_parse_bool(data.get("enable_circuit_breaker", True)),
            circuit_breaker_threshold=int(data.get("circuit_breaker_threshold", 5)),
            circuit_breaker_reset_timeout=float(data.get("circuit_breaker_reset_timeout", 30.0)),
        )
        return cfg.resolve_keys()

    @classmethod
    def _from_dict_no_resolve(cls, data: dict[str, Any]) -> RouterConfig:
        """Build a config from a dict without resolving API keys (for validation)."""
        known = {
            "providers",
            "timeout",
            "max_retries",
            "max_concurrent_per_key",
            "max_concurrent_requests",
            "total_timeout",
            "enable_circuit_breaker",
            "circuit_breaker_threshold",
            "circuit_breaker_reset_timeout",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg = cls(
            providers=list(data.get("providers", [])),
            timeout=float(data.get("timeout", 60.0)),
            max_retries=int(data.get("max_retries", 3)),
            max_concurrent_per_key=int(data.get("max_concurrent_per_key", 100)),
            max_concurrent_requests=(
                int(data["max_concurrent_requests"])
                if data.get("max_concurrent_requests")
                else None
            ),
            total_timeout=(float(data["total_timeout"]) if data.get("total_timeout") else None),
            enable_circuit_breaker=_parse_bool(data.get("enable_circuit_breaker", True)),
            circuit_breaker_threshold=int(data.get("circuit_breaker_threshold", 5)),
            circuit_breaker_reset_timeout=float(data.get("circuit_breaker_reset_timeout", 30.0)),
        )
        return cfg

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> RouterConfig:
        """Load a config from a JSON or YAML file."""
        path = os.fspath(path)
        with open(path, encoding="utf-8") as handle:
            if path.endswith((".yaml", ".yml")):
                if yaml is None:
                    raise ImportError(
                        "PyYAML is required for YAML config files. "
                        "Install with: pip install pyyaml"
                    )
                data = yaml.safe_load(handle)
            else:
                data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Config file must contain a JSON/YAML object.")
        return cls.from_dict(data)

    @classmethod
    def from_providers(cls, providers: list[dict[str, Any]]) -> RouterConfig:
        """Build a config from the ergonomic provider format.

        Each provider dict accepts:
        - provider (required): adapter name (openai, groq, anthropic, etc.)
        - key: literal API key
        - key_env: environment variable name (scans for numbered variants)
        - key_file: path to key file
        - model: default chat model
        - embedding_model: default embedding model
        - base_url: optional OpenAI-compatible base URL
        - scheduler: scheduler instance for key rotation
        - clients: list of client dicts (for multiple keys with per-key options)
        """
        internal_providers = []
        for p in providers:
            if "provider" not in p:
                raise ValueError("Each provider must have a 'provider' field")

            provider_name = p["provider"]
            clients = []

            # Handle inline client(s)
            if "clients" in p:
                # Explicit clients list provided
                for c in p["clients"]:
                    client_dict = dict(c)
                    client_dict.setdefault("client", provider_name)
                    clients.append(client_dict)
            else:
                # Build single client from provider fields
                client_dict: dict[str, Any] = {"client": provider_name}
                if p.get("key"):
                    client_dict["api_key"] = p["key"]
                if p.get("key_env"):
                    client_dict["api_key_env"] = p["key_env"]
                if p.get("key_file"):
                    client_dict["api_key_file"] = p["key_file"]
                if p.get("model"):
                    client_dict["default_model"] = p["model"]
                if p.get("embedding_model"):
                    client_dict["embedding_model"] = p["embedding_model"]
                if p.get("base_url"):
                    client_dict["base_url"] = p["base_url"]
                clients.append(client_dict)

            provider_dict: dict[str, Any] = {
                "name": provider_name,
                "clients": clients,
            }
            if "scheduler" in p and p["scheduler"] is not None:
                provider_dict["scheduler"] = p["scheduler"]

            internal_providers.append(provider_dict)

        return cls(providers=internal_providers)

    def validate(self) -> None:
        if not self.providers:
            raise ValueError("At least one provider must be configured.")

        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero.")

        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0.")

        if self.max_concurrent_per_key <= 0:
            raise ValueError("max_concurrent_per_key must be > 0.")

        if self.max_concurrent_requests is not None and self.max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be > 0 or None.")

        if self.total_timeout is not None and self.total_timeout <= 0:
            raise ValueError("total_timeout must be > 0 or None.")

        if self.circuit_breaker_threshold < 1:
            raise ValueError("circuit_breaker_threshold must be >= 1.")

        if self.circuit_breaker_reset_timeout <= 0:
            raise ValueError("circuit_breaker_reset_timeout must be > 0.")

        for provider in self.providers:
            if not isinstance(provider, dict):
                raise ValueError("Each provider must be a dict with at least a 'name' key.")
            if "name" not in provider:
                raise ValueError("Each provider dict must have a 'name' key.")
            if "clients" not in provider or not provider["clients"]:
                raise ValueError(f"Provider '{provider['name']}' must have at least one client.")
            for client in provider["clients"]:
                if not isinstance(client, dict):
                    raise ValueError(f"Provider '{provider['name']}' has a non-dict client.")
                if "client" not in client:
                    raise ValueError(
                        f"Provider '{provider['name']}' has a client without a 'client' field."
                    )

    def copy(self, **updates: Any) -> RouterConfig:
        values = {
            "providers": self.providers,
            "scheduler": self.scheduler,
            "retry": self.retry,
            "middleware": self.middleware,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "max_concurrent_per_key": self.max_concurrent_per_key,
            "max_concurrent_requests": self.max_concurrent_requests,
            "total_timeout": self.total_timeout,
            "enable_circuit_breaker": self.enable_circuit_breaker,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "circuit_breaker_reset_timeout": self.circuit_breaker_reset_timeout,
        }

        values.update(updates)

        return RouterConfig(**values)  # type: ignore[arg-type]
