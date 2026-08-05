from dataclasses import dataclass, field
from typing import List, Optional, Any
import os


@dataclass
class RouterConfig:
    providers: List[Any] = field(default_factory=list)
    scheduler: Optional[Any] = None
    retry: Optional[Any] = None
    middleware: List[Any] = field(default_factory=list)
    timeout: int = 60
    max_retries: int = 3
    max_concurrent_per_key: int = 100

    @classmethod
    def from_env(cls) -> "RouterConfig":
        return cls(
            providers=[],
            timeout=int(os.getenv("LLMROUTER_TIMEOUT", "60")),
            max_retries=int(os.getenv("LLMROUTER_MAX_RETRIES", "3")),
            max_concurrent_per_key=int(os.getenv("LLMROUTER_MAX_CONCURRENT", "100")),
        )

    def validate(self):
        if not self.providers:
            raise ValueError("RouterConfig: at least one provider must be configured")
        if self.timeout <= 0:
            raise ValueError("RouterConfig.timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("RouterConfig.max_retries must be non-negative")
        if self.max_concurrent_per_key <= 0:
            raise ValueError("RouterConfig.max_concurrent_per_key must be positive")
