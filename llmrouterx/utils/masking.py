from __future__ import annotations


def mask_api_key(api_key: str | None) -> str:
    """
    Render an API key safe for logs: keep the last 4 characters only.

    Keys that are already short (<= 4 chars) are returned unchanged.
    """
    if not api_key:
        return "<unset>"

    text = str(api_key)

    return text if len(text) <= 4 else f"...{text[-4:]}"
