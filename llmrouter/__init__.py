"""LLMRouter package

Expose the public router and package version.
"""

from .router.llmrouter import LLMRouter

__version__ = "0.1.5"

__all__ = ["LLMRouter", "__version__"]
