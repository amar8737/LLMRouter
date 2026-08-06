from __future__ import annotations

from typing import Any

from ..context import RequestContext


class MiddlewareResult:
    """
    Optional structured return value for middleware hooks.

    Hooks may instead return a plain ``dict`` (treated as the new payload or
    response) or ``None`` (leave the value unchanged). Returning a
    ``MiddlewareResult`` is only necessary when you want to short-circuit the
    request with ``stop=True``.
    """

    __slots__ = ("payload", "response", "stop")

    def __init__(
        self,
        *,
        payload: dict[str, Any] | None = None,
        response: Any | None = None,
        stop: bool = False,
    ) -> None:
        self.payload = payload
        self.response = response
        self.stop = stop

    def __repr__(self) -> str:
        return (
            f"MiddlewareResult(stop={self.stop}, "
            f"payload={'set' if self.payload is not None else 'none'}, "
            f"response={'set' if self.response is not None else 'none'})"
        )


class BaseMiddleware:
    """
    Hook points around every routed operation.

    Every hook receives the :class:`RequestContext` for the in-flight request.
    Subclasses only override the hooks they care about.
    """

    async def before_request(
        self,
        operation: str,
        payload: dict[str, Any],
        context: RequestContext,
    ) -> MiddlewareResult | dict[str, Any] | None:
        """
        Inspect or rewrite the payload before it is routed.

        Return ``None`` to leave the payload unchanged, a ``dict`` to replace
        it, or ``MiddlewareResult(response=..., stop=True)`` to short-circuit
        the request (for example, to serve it from a cache).
        """
        return None

    async def after_response(
        self,
        operation: str,
        payload: dict[str, Any],
        response: Any,
        context: RequestContext,
    ) -> MiddlewareResult | dict[str, Any] | None:
        """
        Inspect or rewrite the response before it reaches the caller.
        """
        return None

    async def on_exception(
        self,
        operation: str,
        payload: dict[str, Any],
        exception: BaseException,
        context: RequestContext,
    ) -> None:
        """
        Observe a failed attempt. Raising here replaces the original error.
        """
        return None
