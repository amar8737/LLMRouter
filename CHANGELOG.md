# Changelog

All notable changes to this project will be documented in this file.

## [0.1.5] - 2026-08-05

- Fix: Reset client failure counters on successful requests so health recovers.
- Fix: Propagate `asyncio.CancelledError` through routing and provider layers.
- Feature: Add cooperative cancellation utilities (`utils/cancellation.py`).
- Feature: Support external cancellation for sync stream wrappers via `stop_event`.
- Feature: Add `background_task` integration for async streaming producers.
- Docs: Add streaming examples and `stop_event` usage in `docs/STREAMING.md`.
- CI: Ensure `pytest-asyncio` is installed so async tests run in CI.
- Packaging: Require Python >=3.10 and prepare OIDC/twine publish flow.

## Unreleased

## [0.1.6] - 2026-08-05

- Chore: Bump package version for next release.

No additional changes yet.
# Changelog

All notable changes to this project will be documented in this file.

## v0.1.0 - Initial release (MVP)

- Minimal LLMRouter scaffold
- Public `LLMRouter` interface
- `ProviderRouter`, `CompositeRouter`, `ClientNode`
- Schedulers: RoundRobin, LeastBusy, Random, Weighted, Priority
- Exponential retry policy
- Simple metrics collector
- Unit tests for schedulers
- Packaging and CI-ready workflow
