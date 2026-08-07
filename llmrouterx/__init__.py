"""LLMRouter package

Expose the public router and package version.
"""

from .router.llmrouter import LLMRouter

try:
    from importlib.metadata import PackageNotFoundError, version
    try:
        __version__ = version("llmrouterx")
    except PackageNotFoundError:
        __version__ = "0.1.8"
except ImportError:
    __version__ = "0.1.8"

__all__ = ["LLMRouter", "__version__"]
