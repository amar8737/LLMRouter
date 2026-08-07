"""LLMRouter package

Expose the public router and package version.
"""

from .router.llmrouter import LLMRouter

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("llmrouterx")
    except PackageNotFoundError:
        __version__ = "0.0.0"
except ImportError:  # pragma: no cover - pre-3.8 importlib.metadata shim
    __version__ = "0.0.0"

__all__ = ["LLMRouter", "__version__"]
