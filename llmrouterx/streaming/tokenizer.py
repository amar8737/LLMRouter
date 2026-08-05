from __future__ import annotations

from collections.abc import Callable, Iterable

Token = str
Tokenizer = Callable[[str], Iterable[Token]]


def whitespace_tokenizer(text: str) -> list[str]:
    """
    Split text on whitespace.

    Example
    -------
    >>> whitespace_tokenizer("Hello world!")
    ["Hello", "world!"]
    """
    return text.split()


def character_tokenizer(text: str) -> list[str]:
    """
    Split text into individual characters.

    Useful for demos and testing.
    """
    return list(text)


class TokenizerManager:
    """
    Registry for tokenizer implementations.
    """

    def __init__(self) -> None:
        self._tokenizers: dict[str, Tokenizer] = {
            "whitespace": whitespace_tokenizer,
            "character": character_tokenizer,
        }

    def register(
        self,
        name: str,
        tokenizer: Tokenizer,
    ) -> None:
        if name in self._tokenizers:
            raise ValueError(
                f"Tokenizer '{name}' already exists."
            )

        self._tokenizers[name] = tokenizer

    def unregister(self, name: str) -> None:
        self._tokenizers.pop(name, None)

    def get(self, name: str = "whitespace") -> Tokenizer:
        try:
            return self._tokenizers[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown tokenizer '{name}'."
            ) from exc

    def tokenize(
        self,
        text: str,
        *,
        tokenizer: str = "whitespace",
    ) -> list[str]:
        return list(
            self.get(tokenizer)(text)
        )


default_tokenizer = TokenizerManager()