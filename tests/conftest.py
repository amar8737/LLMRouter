import pytest


class AsyncSDKError(Exception):
    """Mimics an SDK exception with HTTP status metadata."""

    def __init__(self, status_code, message, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}


class OpenAICompatibleClient:
    """Mimics the OpenAI SDK object shape.

    Exposes ``chat.completions.create``, ``embeddings.create``,
    ``responses.create`` and ``models.list`` so adapters can be exercised
    without the real SDK installed.
    """

    def __init__(self):
        self.calls = []
        self.handler = None
        self.chat = type("Chat", (), {"completions": self._namespace(self._chat)})()
        self.responses = self._namespace(self._responses)
        self.embeddings = self._namespace(self._embeddings)
        self.rerank = self._namespace(self._rerank)
        self.models = type("Models", (), {"list": self._models})()

    def set_handler(self, handler):
        self.handler = handler

    def _namespace(self, fn):
        return type("Namespace", (), {"create": fn})()

    async def _chat(self, **kwargs):
        self.calls.append(("chat", kwargs))
        if self.handler:
            return await self.handler("chat", kwargs)
        return _chat_response("default")

    async def _responses(self, *args, **kwargs):
        self.calls.append(("responses", kwargs))
        if self.handler:
            return await self.handler("responses", kwargs)
        return _chat_response("responses")

    async def _embeddings(self, **kwargs):
        self.calls.append(("embeddings", kwargs))
        if self.handler:
            return await self.handler("embeddings", kwargs)
        return _embedding_response([0.1, 0.2, 0.3])

    async def _rerank(self, **kwargs):
        self.calls.append(("rerank", kwargs))
        if self.handler:
            return await self.handler("rerank", kwargs)
        return _rerank_response(
            [
                type("Result", (), {"index": 0, "relevance_score": 0.42})(),
                type("Result", (), {"index": 1, "relevance_score": 0.91})(),
            ]
        )

    async def _models(self, **kwargs):
        self.calls.append(("models", kwargs))
        if self.handler:
            return await self.handler("models", kwargs)
        return []


def _chat_response(text):
    return type(
        "Response",
        (),
        {
            "choices": [
                type(
                    "Choice",
                    (),
                    {"message": type("Message", (), {"content": text})()},
                )()
            ]
        },
    )()


def _embedding_response(vector):
    return type(
        "Response",
        (),
        {"data": [type("Embedding", (), {"embedding": vector})()]},
    )()


def _rerank_response(results):
    return type("Response", (), {"results": results})()


class AnthropicClient:
    """Mimics the Anthropic SDK ``client.messages`` surface."""

    def __init__(self):
        self.messages = type("Messages", (), {"create": self._create})()
        self.models = type("Models", (), {"list": self._models})()
        self.handler = None

    def set_handler(self, handler):
        self.handler = handler

    async def _create(self, **kwargs):
        if self.handler:
            return await self.handler(**kwargs)
        return _anthropic_text_response("hello anthropic")

    async def _models(self, **kwargs):
        if self.handler:
            return await self.handler(**kwargs)
        return []


def _anthropic_text_response(text):
    return type(
        "Response",
        (),
        {
            "content": [
                type(
                    "Block",
                    (),
                    {"type": "text", "text": text},
                )()
            ]
        },
    )()


@pytest.fixture
def openai_client():
    return OpenAICompatibleClient()


@pytest.fixture
def anthropic_client():
    return AnthropicClient()
