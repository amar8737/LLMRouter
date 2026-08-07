from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from .secrets import resolve_key


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
            enable_circuit_breaker=_parse_bool(
                os.getenv("LLMROUTER_CIRCUIT_BREAKER", "true")
            ),
            circuit_breaker_threshold=int(os.getenv("LLMROUTER_CB_THRESHOLD", "5")),
            circuit_breaker_reset_timeout=float(os.getenv("LLMROUTER_CB_RESET_TIMEOUT", "30")),
        )

    def resolve_keys(self) -> RouterConfig:
        """Return a copy with every client's API key materialised.

        Each client dict's ``api_key``/``api_key_env``/``api_key_file`` source
        is replaced with the resolved literal key under ``api_key``.
        """
        clean: list[Any] = []
        for provider in self.providers:
            resolved_provider = dict(provider)
            resolved_provider["clients"] = [
                {**client, "api_key": resolve_key(client)}
                for client in provider.get("clients", [])
            ]
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
            total_timeout=(
                float(data["total_timeout"]) if data.get("total_timeout") else None
            ),
            enable_circuit_breaker=_parse_bool(data.get("enable_circuit_breaker", True)),
            circuit_breaker_threshold=int(data.get("circuit_breaker_threshold", 5)),
            circuit_breaker_reset_timeout=float(
                data.get("circuit_breaker_reset_timeout", 30.0)
            ),
        )
        return cfg.resolve_keys()

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> RouterConfig:
        """Load a config from a JSON file."""
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Config file must contain a JSON object.")
        return cls.from_dict(data)

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
                    raise ValueError(
                        f"Provider '{provider['name']}' has a non-dict client."
                    )
                if "client" not in client:
                    raise ValueError(
                        f"Provider '{provider['name']}' has a client without a "
                        f"'client' field."
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
