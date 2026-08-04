from dataclasses import dataclass
from typing import Any, List


@dataclass
class RouterConfig:
    providers: List[Any]
    scheduler: Any = None
    retry: Any = None
    middleware: List[Any] = None
    timeout: int = 60
    max_retries: int = 3
