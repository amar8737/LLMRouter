from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from typing import Any


class JsonFormatter(logging.Formatter):
    """
    Emit log records as single-line JSON objects.

    Extra fields are attached via the private ``_llmrouterx_extra`` attribute
    on the log record, for example the router sets
    ``record._llmrouterx_extra = {"request_id": rid, "provider": "openai"}
    `` before logging.  Do not use the standard ``extra`` keyword argument:
    it is ignored by this formatter.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        extra = getattr(record, "_llmrouterx_extra", None)
        if isinstance(extra, Mapping):
            payload.update(extra)

        return json.dumps(payload, default=str)


def setup_logging(
    *,
    level: int | str = logging.INFO,
    fmt: str = "json",
    stream=sys.stderr,
) -> None:
    """
    Configure the ``llmrouterx`` logger.

    Parameters
    ----------
    level
        Log level, e.g. ``logging.INFO`` or ``"DEBUG"``.
    fmt
        ``"json"`` for structured single-line JSON, ``"plain"`` otherwise.
    stream
        Output stream, defaults to stderr.
    """
    handler = logging.StreamHandler(stream)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    logger = logging.getLogger("llmrouterx")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
