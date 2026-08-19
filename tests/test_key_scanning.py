"""Tests for key-name scanning (``api_key_env`` prefixes and regexes)."""

import pytest

from llmrouterx.config.config import RouterConfig
from llmrouterx.config.secrets import (
    KeyResolutionError,
    expand_env_key_names,
    resolve_key,
    resolve_keys,
)


def test_expand_base_only(monkeypatch):
    monkeypatch.setenv("OPEN_AI_KEY", "sk-a")
    monkeypatch.delenv("OPEN_AI_KEY_1", raising=False)
    assert expand_env_key_names("OPEN_AI_KEY") == ["OPEN_AI_KEY"]


def test_expand_scans_numbered_variants(monkeypatch):
    monkeypatch.setenv("OPEN_AI_KEY", "sk-a")
    monkeypatch.setenv("OPEN_AI_KEY_1", "sk-b")
    monkeypatch.setenv("OPEN_AI_KEY_2", "sk-c")
    monkeypatch.setenv("OPEN_AI_KEY_3", "sk-d")
    assert expand_env_key_names("OPEN_AI_KEY") == [
        "OPEN_AI_KEY",
        "OPEN_AI_KEY_1",
        "OPEN_AI_KEY_2",
        "OPEN_AI_KEY_3",
    ]


def test_expand_stops_at_first_gap(monkeypatch):
    monkeypatch.setenv("OPEN_AI_KEY", "sk-a")
    monkeypatch.setenv("OPEN_AI_KEY_1", "sk-b")
    monkeypatch.setenv("OPEN_AI_KEY_2", "sk-c")
    monkeypatch.setenv("OPEN_AI_KEY_4", "sk-d")
    assert expand_env_key_names("OPEN_AI_KEY") == [
        "OPEN_AI_KEY",
        "OPEN_AI_KEY_1",
        "OPEN_AI_KEY_2",
    ]


def test_expand_numbered_only_when_base_missing(monkeypatch):
    monkeypatch.delenv("OPEN_AI_KEY", raising=False)
    monkeypatch.setenv("OPEN_AI_KEY_1", "sk-b")
    monkeypatch.setenv("OPEN_AI_KEY_2", "sk-c")
    assert expand_env_key_names("OPEN_AI_KEY") == ["OPEN_AI_KEY", "OPEN_AI_KEY_1", "OPEN_AI_KEY_2"]


def test_expand_nothing_set_raises(monkeypatch):
    monkeypatch.delenv("OPEN_AI_KEY", raising=False)
    monkeypatch.delenv("OPEN_AI_KEY_1", raising=False)
    with pytest.raises(KeyResolutionError, match="not set"):
        expand_env_key_names("OPEN_AI_KEY")


def test_expand_scan_disabled(monkeypatch):
    monkeypatch.setenv("OPEN_AI_KEY", "sk-a")
    monkeypatch.setenv("OPEN_AI_KEY_1", "sk-b")
    assert expand_env_key_names("OPEN_AI_KEY", scan=False) == ["OPEN_AI_KEY"]


def test_expand_regex(monkeypatch):
    monkeypatch.setenv("JINA_KEY", "jina-1")
    monkeypatch.setenv("JINA_KEY_2", "jina-2")
    monkeypatch.setenv("JINA_KEY_10", "jina-10")
    monkeypatch.setenv("UNRELATED", "x")
    names = expand_env_key_names("whatever", scan=False, regex=r"JINA_KEY(_\d+)?")
    assert names == ["JINA_KEY", "JINA_KEY_2", "JINA_KEY_10"]


def test_expand_regex_no_match_raises(monkeypatch):
    monkeypatch.delenv("NOPE_1", raising=False)
    with pytest.raises(KeyResolutionError, match="matched api_key_env_regex"):
        expand_env_key_names("NOPE", scan=False, regex=r"NOPE_\d+")


def test_resolve_keys_literal(monkeypatch):
    assert resolve_keys({"api_key": "sk-1"}) == ["sk-1"]


def test_resolve_keys_env_scan(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-a")
    monkeypatch.setenv("GROQ_API_KEY_1", "gsk-b")
    keys = resolve_keys({"api_key_env": "GROQ_API_KEY"})
    assert keys == ["gsk-a", "gsk-b"]


def test_resolve_keys_env_scan_disabled(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-a")
    monkeypatch.setenv("GROQ_API_KEY_1", "gsk-b")
    keys = resolve_keys({"api_key_env": "GROQ_API_KEY", "api_key_env_scan": False})
    assert keys == ["gsk-a"]


def test_resolve_keys_env_regex(monkeypatch):
    monkeypatch.setenv("VOYAGE_KEY", "pa-1")
    monkeypatch.setenv("VOYAGE_KEY_1", "pa-2")
    keys = resolve_keys({"api_key_env": "VOYAGE_KEY", "api_key_env_regex": r"VOYAGE_KEY(_\d+)?"})
    assert keys == ["pa-1", "pa-2"]


def test_resolve_key_single(monkeypatch):
    monkeypatch.setenv("OPEN_AI_KEY", "sk-1")
    assert resolve_key({"api_key_env": "OPEN_AI_KEY"}) == "sk-1"


def test_config_from_dict_expands_clients(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-a")
    monkeypatch.setenv("OPENAI_API_KEY_1", "sk-b")
    key_file = tmp_path / "groq-key"
    key_file.write_text("gsk-file")

    cfg = RouterConfig.from_dict(
        {
            "providers": [
                {
                    "name": "openai",
                    "clients": [{"client": "openai", "api_key_env": "OPENAI_API_KEY"}],
                },
                {
                    "name": "groq",
                    "clients": [{"client": "groq", "api_key_file": str(key_file)}],
                },
            ],
            "max_retries": 2,
        }
    )

    openai_clients = cfg.providers[0]["clients"]
    assert [c["api_key"] for c in openai_clients] == ["sk-a", "sk-b"]
    assert cfg.providers[1]["clients"][0]["api_key"] == "gsk-file"


def test_config_from_dict_missing_key_raises(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    monkeypatch.delenv("MISSING_KEY_1", raising=False)
    with pytest.raises(KeyResolutionError):
        RouterConfig.from_dict(
            {"providers": [{"name": "openai", "clients": [{"api_key_env": "MISSING_KEY"}]}]}
        )
