# AGENTS.md

## Environment
- Python 3.10+ (use a venv)
- Package uses setuptools; install editable with extras: `pip install -e .[dev,server]`

## Common commands
- Lint: `ruff check llmrouterx tests`
- Format: `ruff format llmrouterx tests`
- Type check: `mypy llmrouterx`
- Tests: `pytest -q` (uses pytest-asyncio; tests live in `tests/`)

## Build checks (run before PRs)
1. `ruff check . && ruff format --check .`
2. `mypy llmrouterx`
3. `pytest -q`

## Notes
- `ClientNode` expects a provider **adapter** (see AdapterFactory), not a raw SDK client.
- Adapters must implement `chat`, `stream`, `chat_completion`, and `embeddings`.
- `LLMRouter.metrics` exposes `get()`, `snapshot()`, `incr()`, `timing()`, `timing_stats()`, `track_tokens()`.
