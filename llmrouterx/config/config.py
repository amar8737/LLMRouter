from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


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
            enable_circuit_breaker=(
                os.getenv("LLMROUTER_CIRCUIT_BREAKER", "true").lower() != "false"
            ),
            circuit_breaker_threshold=int(os.getenv("LLMROUTER_CB_THRESHOLD", "5")),
            circuit_breaker_reset_timeout=float(os.getenv("LLMROUTER_CB_RESET_TIMEOUT", "30")),
        )

    def validate(self) -> None:
        if not self.providers:
            raise ValueError(
                "At least one provider must be configured."
            )

        if self.timeout <= 0:
            raise ValueError(
                "timeout must be greater than zero."
            )

        if self.max_retries < 0:
            raise ValueError(
                "max_retries must be >= 0."
            )

        if self.max_concurrent_per_key <= 0:
            raise ValueError(
                "max_concurrent_per_key must be > 0."
            )

        if self.max_concurrent_requests is not None and self.max_concurrent_requests <= 0:
            raise ValueError(
                "max_concurrent_requests must be > 0 or None."
            )

        if self.circuit_breaker_threshold < 1:
            raise ValueError(
                "circuit_breaker_threshold must be >= 1."
            )

        if self.circuit_breaker_reset_timeout <= 0:
            raise ValueError(
                "circuit_breaker_reset_timeout must be > 0."
            )

        for provider in self.providers:
            if not isinstance(provider, dict):
                raise ValueError(
                    "Each provider must be a dict with at least a 'name' key."
                )
            if "name" not in provider:
                raise ValueError(
                    "Each provider dict must have a 'name' key."
                )
            if "clients" not in provider or not provider["clients"]:
                raise ValueError(
                    f"Provider '{provider['name']}' must have at least one client."
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
            "enable_circuit_breaker": self.enable_circuit_breaker,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "circuit_breaker_reset_timeout": self.circuit_breaker_reset_timeout,
        }

        values.update(updates)

        return RouterConfig(**values)