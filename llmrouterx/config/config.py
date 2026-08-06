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

    def copy(self, **updates: Any) -> RouterConfig:
        values = {
            "providers": self.providers,
            "scheduler": self.scheduler,
            "retry": self.retry,
            "middleware": self.middleware,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "max_concurrent_per_key": self.max_concurrent_per_key,
        }

        values.update(updates)

        return RouterConfig(**values)