from typing import Any


class RouterError(Exception):
    pass


class ProviderError(RouterError):
    pass


class NoHealthyClientError(ProviderError):
    """Raised when no provider/client can serve a request.

    ``errors`` carries every failure observed while trying providers, so
    callers can surface the full failure sequence instead of a bare message.
    """

    def __init__(self, message: str, errors: list[Exception] | None = None) -> None:
        self.errors = list(errors or [])
        if self.errors:
            detail = "\n".join(f"  - {type(e).__name__}: {e}" for e in self.errors)
            message = f"{message}\nFailure sequence:\n{detail}"
        super().__init__(message)


class StreamError(ProviderError):
    """Raised when no provider/client can produce a streamed response."""


class ConfigurationError(RouterError):
    pass


class HTTPError(Exception):
    """Generic HTTP error."""

    def __init__(
        self,
        status_code: int,
        message: str = "HTTP error",
        *,
        headers: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{status_code}: {message}")
        self.status_code = status_code
        self.headers = headers or {}
