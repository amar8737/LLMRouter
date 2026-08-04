from dataclasses import dataclass
from typing import Any


@dataclass
class RouterConfig:
    providers: list[Any]
    scheduler: Any = None
    retry: Any = None
    middleware: list[Any] = None
    timeout: int = 60
    max_retries: int = 3
