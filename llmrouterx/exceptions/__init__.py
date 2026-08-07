class RouterError(Exception):
    pass


class ProviderError(RouterError):
    pass


class RetryError(RouterError):
    pass


class SchedulerError(RouterError):
    pass


class NoHealthyClientError(ProviderError):
    pass


class StreamError(ProviderError):
    """Raised when no provider/client can produce a streamed response."""


class ConfigurationError(RouterError):
    pass


class AuthenticationError(RouterError):
    pass
