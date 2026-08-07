class RouterError(Exception):
    pass


class ProviderError(RouterError):
    pass


class RetryError(RouterError):
    pass


class SchedulerError(RouterError):
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


class AuthenticationError(RouterError):
    pass
