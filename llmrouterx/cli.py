"""Command-line interface for LLMRouter.

Provides config management commands for LLMRouter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from llmrouterx.config.config import RouterConfig


def _parse_provider_spec(spec: str) -> tuple[str, str, str | None]:
    """Parse a provider specification.

    Formats:
    - "provider:key" -> (provider, key, None)
    - "provider:key@model" -> (provider, key, model)

    Returns (provider_name, key_spec, default_model).
    """
    if ":" not in spec:
        raise ValueError(
            f"Provider spec {spec!r} must be formatted as 'provider:key' or 'provider:key@model'"
        )

    provider_part, key_part = spec.split(":", 1)

    model = None
    if "@" in key_part:
        key_part, model = key_part.split("@", 1)

    return provider_part, key_part, model


def _build_router_config_from_providers(
    provider_specs: list[str],
) -> RouterConfig:
    """Build a RouterConfig from --provider specs."""
    providers = []
    for spec in provider_specs:
        provider_name, key_spec, default_model = _parse_provider_spec(spec)
        providers.append(
            {
                "provider": provider_name,
                "key": key_spec if _looks_like_literal_key(key_spec) else None,
                "key_env": None if _looks_like_literal_key(key_spec) else key_spec,
                "model": default_model,
            }
        )
    return RouterConfig.from_providers(providers)


def _build_router_config_from_fallback(
    fallback_specs: list[str],
) -> RouterConfig:
    """Build a RouterConfig from --fallback specs (cascade)."""
    providers = []
    for spec in fallback_specs:
        provider_name, key_spec, default_model = _parse_provider_spec(spec)
        providers.append(
            {
                "provider": provider_name,
                "key": key_spec if _looks_like_literal_key(key_spec) else None,
                "key_env": None if _looks_like_literal_key(key_spec) else key_spec,
                "model": default_model,
            }
        )
    return RouterConfig.from_providers(providers)


def _looks_like_literal_key(key_spec: str) -> bool:
    """Heuristic: does this look like a literal API key vs env var name?"""
    # Common prefixes for literal keys
    literal_prefixes = ("sk-", "gsk-", "sk-ant-", "xai-", "pplx-", "api-", "key-")
    return any(key_spec.startswith(p) for p in literal_prefixes) or len(key_spec) > 32


def _resolve_keys(client_cfg: dict[str, Any]) -> list[str]:
    """Resolve API keys from a client config (copied from config.secrets)."""
    # Simplified version - just return the key if present
    if client_cfg.get("api_key"):
        return [client_cfg["api_key"]]
    # In real implementation, this would resolve env vars and files
    return []


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmrouterx",
        description="LLMRouter CLI - Configuration utilities",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ============================================================
    # init command
    # ============================================================
    init = subparsers.add_parser("init", help="Generate a starter router config file")
    init.add_argument(
        "--format",
        type=str,
        default="yaml",
        choices=["yaml", "json"],
        help="Output format",
    )
    init.add_argument(
        "-o",
        "--output",
        type=str,
        default="router.yaml",
        help="Output file path",
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file",
    )

    # ============================================================
    # config validate command
    # ============================================================
    config_parser = subparsers.add_parser("config", help="Configuration utilities")
    config_subparsers = config_parser.add_subparsers(dest="config_command", help="Config commands")

    validate = config_subparsers.add_parser("validate", help="Validate a router config file")
    validate.add_argument("file", type=str, help="Path to config file (JSON or YAML)")
    validate.add_argument(
        "--format",
        type=str,
        default="auto",
        choices=["auto", "json", "yaml"],
        help="Force file format (default: auto-detect from extension)",
    )
    validate.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error on warnings (not just errors)",
    )

    return parser


def _build_config_from_args(args: argparse.Namespace) -> RouterConfig | None:
    """Build RouterConfig from CLI arguments (--provider, --fallback)."""
    if args.provider and args.fallback:
        print("Error: --provider and --fallback cannot be used together", file=sys.stderr)
        sys.exit(2)

    if args.provider:
        return _build_router_config_from_providers(args.provider)

    if args.fallback:
        return _build_router_config_from_fallback(args.fallback)

    return None


def _run_init(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        print(f"Error: {output_path} already exists. Use --force to overwrite.", file=sys.stderr)
        sys.exit(2)

    config = _get_starter_config()

    if args.format == "yaml":
        try:
            import yaml
        except ImportError:
            print(
                "Error: PyYAML is required for YAML output. Install with: pip install pyyaml",
                file=sys.stderr,
            )
            sys.exit(1)
        content = yaml.dump(config, sort_keys=False, default_flow_style=False)
    else:
        content = json.dumps(config, indent=2)

    output_path.write_text(content)
    print(f"Generated {output_path}")


def _get_starter_config() -> dict[str, Any]:
    """Generate a comprehensive starter config with comments."""
    return {
        "providers": [
            {
                "name": "openai",
                "clients": [
                    {
                        "client": "openai",
                        "api_key_env": "OPENAI_API_KEY",
                        "default_model": "gpt-4o",
                        "embedding_model": "text-embedding-3-small",
                        # Optional per-client settings:
                        # "weight": 1,                    # For WeightedScheduler
                        # "priority": 100,                # For PriorityScheduler
                        # "failure_threshold": 5,         # Circuit breaker threshold
                        # "cooldown_seconds": 30,         # Circuit breaker reset timeout
                    },
                    # Add more keys for the same provider (rotated by scheduler):
                    # { "client": "openai",
                    #   "api_key_env": "OPENAI_API_KEY_2",
                    #   "default_model": "gpt-4o" },
                ],
                # Optional provider-level scheduler:
                # "scheduler": "least_busy",  # or "round_robin", "random", "weighted", "priority"
            },
            {
                "name": "groq",
                "clients": [
                    {
                        "client": "groq",
                        "api_key_env": "GROQ_API_KEY",
                        "default_model": "llama-3.3-70b-versatile",
                    }
                ],
            },
            {
                "name": "anthropic",
                "clients": [
                    {
                        "client": "anthropic",
                        "api_key_env": "ANTHROPIC_API_KEY",
                        "default_model": "claude-3-5-sonnet-20241022",
                    }
                ],
            },
        ],
        # Global settings:
        "timeout": 60.0,  # Per-request timeout (seconds)
        "max_retries": 3,  # Max retries per request
        "max_concurrent_per_key": 100,  # Max concurrent requests per API key
        "max_concurrent_requests": None,  # Global max concurrent (None = unlimited)
        "total_timeout": None,  # Total timeout across retries (None = unlimited)
        "enable_circuit_breaker": True,  # Enable circuit breaker
        "circuit_breaker_threshold": 5,  # Failures before opening circuit
        "circuit_breaker_reset_timeout": 30.0,  # Seconds before half-open
    }


def _run_config_validate(args: argparse.Namespace) -> None:
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(2)

    # Determine format
    fmt = args.format
    if fmt == "auto":
        suffix = file_path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            fmt = "yaml"
        elif suffix == ".json":
            fmt = "json"
        else:
            print(
                f"Error: Cannot auto-detect format for {file_path}. Use --format.",
                file=sys.stderr,
            )
            sys.exit(2)

    # Load config
    try:
        if fmt == "yaml":
            import yaml

            content = yaml.safe_load(file_path.read_text())
        else:
            content = json.loads(file_path.read_text())
    except Exception as e:
        print(f"Error parsing config file: {e}", file=sys.stderr)
        sys.exit(2)

    # Validate structure (without resolving keys)
    try:
        config = RouterConfig._from_dict_no_resolve(content)
        config.validate()
    except Exception as e:
        print(f"Validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print(f"✓ Config file {file_path} is valid")
    print(f"  Providers: {len(config.providers)}")
    for p in config.providers:
        client_count = len(p.get("clients", []))
        scheduler = p.get("scheduler", "default")
        print(f"  - {p['name']}: {client_count} client(s), scheduler={scheduler}")

    sys.exit(0)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        _run_init(args)
    elif args.command == "config":
        if args.config_command == "validate":
            _run_config_validate(args)
        else:
            parser.print_help()
            sys.exit(2)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
