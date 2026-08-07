from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RequestContext:
    """
    Carries request metadata throughout the routing pipeline.

    This object is shared by:

    - Router
    - Middleware
    - Retry
    - Metrics
    - Logging
    - Scheduler
    - Provider
    """

    # ----------------------------------------------------
    # Identity
    # ----------------------------------------------------

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    trace_id: str | None = None

    parent_id: str | None = None

    # ----------------------------------------------------
    # Request
    # ----------------------------------------------------

    operation: str = ""

    prompt: str | None = None

    model: str | None = None

    provider: str | None = None

    api_key: str | None = None

    tenant: str | None = None

    # ----------------------------------------------------
    # Retry
    # ----------------------------------------------------

    retry_count: int = 0

    max_retries: int = 0

    # ----------------------------------------------------
    # Timing
    # ----------------------------------------------------

    started_at: float = field(default_factory=time.perf_counter)

    finished_at: float | None = None

    # ----------------------------------------------------
    # Metadata
    # ----------------------------------------------------

    metadata: dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------
    # Helpers
    # ----------------------------------------------------

    def elapsed(self) -> float:

        end = self.finished_at

        if end is None:
            end = time.perf_counter()

        return end - self.started_at

    def finish(self) -> None:

        self.finished_at = time.perf_counter()

    def increment_retry(self) -> None:

        self.retry_count += 1

    def set(
        self,
        key: str,
        value: Any,
    ) -> None:

        self.metadata[key] = value

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:

        return self.metadata.get(
            key,
            default,
        )

    def copy(self) -> RequestContext:

        return RequestContext(
            request_id=self.request_id,
            trace_id=self.trace_id,
            parent_id=self.parent_id,
            operation=self.operation,
            prompt=self.prompt,
            model=self.model,
            provider=self.provider,
            api_key=self.api_key,
            tenant=self.tenant,
            retry_count=self.retry_count,
            max_retries=self.max_retries,
            started_at=self.started_at,
            finished_at=self.finished_at,
            metadata=self.metadata.copy(),
        )
