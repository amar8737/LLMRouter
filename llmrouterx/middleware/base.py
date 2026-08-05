from __future__ import annotations

from abc import ABC
from typing import Any


class RequestContext:
    """
    Shared request context available to every middleware.
    """

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        return self.data.get(key, default)


class MiddlewareResult:
    """
    Result returned by middleware.
    """

    def __init__(
        self,
        *,
        payload: dict[str, Any] | None = None,
        response: dict[str, Any] | None = None,
        stop: bool = False,
    ) -> None:
        self.payload = payload
        self.response = response
        self.stop = stop


class BaseMiddleware(ABC):
    """
    Base middleware class.

    Lifecycle:

        before_request()

                ↓

            provider

                ↓

        after_response()

                ↓

        on_exception()
    """

    async def before_request(
        self,
        operation: str,
        payload: dict[str, Any],
        context: RequestContext,
    ) -> MiddlewareResult:
        return MiddlewareResult(payload=payload)

    async def after_response(
        self,
        operation: str,
        payload: dict[str, Any],
        response: dict[str, Any],
        context: RequestContext,
    ) -> MiddlewareResult:
        return MiddlewareResult(response=response)

    async def on_exception(
        self,
        operation: str,
        payload: dict[str, Any],
        exception: Exception,
        context: RequestContext,
    ) -> None:
        return None