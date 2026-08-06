from __future__ import annotations

from collections.abc import Iterator
from threading import RLock

from ..providers.provider_router import ProviderRouter


class ProviderRegistry:
    """
    Thread-safe registry for ProviderRouter instances.

    Responsibilities
    ----------------
    - Register providers
    - Unregister providers
    - Lookup providers
    - Enumerate healthy providers
    - Enable future hot-reload/plugins
    """

    def __init__(self) -> None:
        self._providers: dict[str, ProviderRouter] = {}
        self._lock = RLock()

    # -----------------------------------------------------
    # Registration
    # -----------------------------------------------------

    def register(
        self,
        provider: ProviderRouter,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Register a provider.
        """

        with self._lock:

            if (
                provider.name in self._providers
                and not overwrite
            ):
                raise ValueError(
                    f"Provider '{provider.name}' already exists."
                )

            self._providers[provider.name] = provider

    def unregister(
        self,
        name: str,
    ) -> ProviderRouter:
        """
        Remove a provider.
        """

        with self._lock:

            try:
                return self._providers.pop(name)

            except KeyError as exc:
                raise KeyError(
                    f"Unknown provider '{name}'."
                ) from exc

    # -----------------------------------------------------
    # Lookup
    # -----------------------------------------------------

    def get(
        self,
        name: str,
    ) -> ProviderRouter:

        try:
            return self._providers[name]

        except KeyError as exc:
            raise KeyError(
                f"Unknown provider '{name}'."
            ) from exc

    def exists(
        self,
        name: str,
    ) -> bool:

        return name in self._providers

    # -----------------------------------------------------
    # Enumeration
    # -----------------------------------------------------

    def all(self) -> list[ProviderRouter]:

        with self._lock:
            return list(self._providers.values())

    async def healthy(self) -> list[ProviderRouter]:
        """
        Return all healthy providers.
        """

        providers = self.all()

        healthy: list[ProviderRouter] = []

        for provider in providers:

            if await provider.is_healthy():
                healthy.append(provider)

        return healthy

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------

    def clear(self) -> None:

        with self._lock:
            self._providers.clear()

    def __len__(self) -> int:
        return len(self._providers)

    def __contains__(
        self,
        name: str,
    ) -> bool:

        return self.exists(name)

    def __iter__(
        self,
    ) -> Iterator[ProviderRouter]:

        return iter(self.all())

    def __repr__(self) -> str:

        return (
            f"ProviderRegistry("
            f"providers={list(self._providers.keys())})"
        )