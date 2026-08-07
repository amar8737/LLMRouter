"""Command-line interface for LLMRouter.

The primary command is ``llmrouterx serve`` which launches the standalone
OpenAI-compatible HTTP gateway backed by Uvicorn.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any


def _use_uvloop() -> bool:
    """Install uvloop as the event-loop policy where available.

    Returns True when uvloop is in use, False when falling back to the
    standard asyncio loop (e.g. platform lacks uvloop support).
    """
    try:
        import uvloop  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        return True
    except Exception:  # pragma: no cover - uvloop unsupported at runtime
        return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmrouterx",
        description="LLMRouter CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    serve = subparsers.add_parser("serve", help="Start the LLMRouter Gateway server")
    serve.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    serve.add_argument("--port", type=int, default=8000, help="Port to listen on")
    serve.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    serve.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON router config file (see RouterConfig.from_file)",
    )
    serve.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")
    serve.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging verbosity",
    )
    serve.add_argument(
        "--admin-token",
        type=str,
        default=None,
        help="Bearer token required on /dashboard and /metrics (also LLMROUTER_ADMIN_TOKEN)",
    )
    serve.add_argument(
        "--api-key",
        type=str,
        action="append",
        default=None,
        help="Bearer key accepted on /v1/* (repeatable; also LLMROUTER_API_KEYS, comma-separated)",
    )
    serve.add_argument(
        "--no-docs",
        action="store_true",
        help="Disable interactive OpenAPI docs (also LLMROUTER_DOCS=0)",
    )

    return parser


def _run_serve(args: argparse.Namespace) -> None:
    if args.workers and args.workers < 1:
        print("--workers must be >= 1.", file=sys.stderr)
        sys.exit(2)

    if args.reload and args.workers and args.workers > 1:
        print("--reload cannot be combined with --workers > 1.", file=sys.stderr)
        sys.exit(2)

    try:
        import uvicorn
    except ImportError:
        print("Error: FastAPI or Uvicorn is missing.", file=sys.stderr)
        print("Please install server extras: pip install llmrouterx[server]", file=sys.stderr)
        sys.exit(1)

    if _use_uvloop():
        logging.getLogger("llmrouterx.cli").info("Using uvloop event loop.")
    else:
        logging.getLogger("llmrouterx.cli").info("Using the standard asyncio event loop.")

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    # Multi-worker mode needs the app factory as an import string, so it cannot
    # be combined with an inline config file. Everything else builds the app in
    # this process and passes the instance directly to Uvicorn.
    if args.config:
        if args.workers and args.workers > 1:
            print(
                "--workers > 1 requires factory mode and cannot be combined with --config.",
                file=sys.stderr,
            )
            sys.exit(2)

        from llmrouterx.server.app import create_app

        app: Any = create_app(
            config_path=args.config,
            admin_token=args.admin_token,
            api_keys=args.api_key,
            docs_enabled=not args.no_docs,
        )
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
        return

    target = "llmrouterx.server.app:create_app"
    uvicorn.run(
        target,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        factory=True,
        log_level=args.log_level,
    )


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "serve":
        parser.print_help()
        return

    _run_serve(args)


if __name__ == "__main__":
    main()
