# LLMRouter (MVP)

Minimal scaffold for the LLMRouter project described in `requirements.md`.

Quick start:

Run the included smoke test:

```bash
PYTHONPATH=. python3 tests/run_tests.py
```

This repository contains a minimal async `LLMRouter` implementation and a stub provider for local testing.

Testing
-------

Smoke test (quick run):

```bash
PYTHONPATH=. python3 tests/run_tests.py
```

Unit tests (requires `pytest`):

Install dev dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run tests:

```bash
PYTHONPATH=. pytest -q
```

Tests added cover scheduler implementations (`tests/test_schedulers.py`).

Local install (editable)
------------------------

It's recommended to use a virtual environment to install the package locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

If your system Python is managed by the OS (PEP 668), create a venv first as shown above before installing.

# LLMRouter