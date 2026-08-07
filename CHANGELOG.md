# Changelog

All notable changes to this project will be documented in this file.

## [0.1.9] - 2026-08-07

### Reliability
- Add `on_retry` middleware hook: middleware can now veto individual retries.
- Add `StreamError` (raised when no provider/client can produce a streamed response).
- Add `total_timeout` to `LLMRouter`, `RouterConfig`, and `RouterFactory`
  (`LLMROUTER_TOTAL_TIMEOUT` env var): a hard deadline covering retries and failover.
- Add a default timeout (5s) to `CompositeRouter.health()`.
- Add opt-in transient-failure counting via `ClientNode(count_transient_failures=True)`.
- Deduplicate backoff between the scheduler retry loop and the router-level retry policy.
- Populate `context.provider` / `context.api_key` (and `last_provider` / `last_api_key`)
  so middleware and metrics can attribute every request.
- `LLMRouter.close()` now waits for in-flight tasks before cancelling leftovers.
- Fix `on_exception` middleware only running for streams; it now fires for every
  retryable and non-retryable failure.

### Observability
- Add backward-compatible `labels` support to `MetricsCollector.incr()`/`timing()`;
  the router emits per-provider `requests.*` and `latency.*` metrics.
- Add structured logging helpers (`utils/logging.py`): `JsonFormatter` and `setup_logging()`.

### Docs
- Document the thread-per-stream limitation of `SyncStreamEngine`.
- Update `README.md`, `docs/QUICKSTART.md`, `docs/STREAMING.md`, and
  `README_ENHANCED.md` for the new APIs and corrected package name (`llmrouterx`).

### Dev / CI
- Add `mypy` and `pytest-cov` dev dependencies plus config; fix type errors so
  `mypy` passes cleanly on `llmrouterx`.
- Add coverage and type-check steps to the GitHub Actions workflow.
- Add ~150 regression tests across retry, circuit breaker, metrics, config,
  streaming, adapters, middleware, and composite routing (195 tests total).

## [0.1.6] - 2026-08-05

- Chore: bump package version for next release.

## [0.1.5] - 2026-08-05

- Fix: Reset client failure counters on successful requests so health recovers.
- Fix: Propagate `asyncio.CancelledError` through routing and provider layers.
- Feature: Add cooperative cancellation utilities (`utils/cancellation.py`).
- Feature: Support external cancellation for sync stream wrappers via `stop_event`.
- Feature: Add `background_task` integration for async streaming producers.
- Docs: Add streaming examples and `stop_event` usage in `docs/STREAMING.md`.
- CI: Ensure `pytest-asyncio` is installed so async tests run in CI.
- Packaging: Require Python >=3.10 and prepare OIDC/twine publish flow.

## [0.1.0] - Initial release (MVP)

- Minimal LLMRouter scaffold
- Public `LLMRouter` interface
- `ProviderRouter`, `CompositeRouter`, `ClientNode`
- Schedulers: RoundRobin, LeastBusy, Random, Weighted, Priority
- Exponential retry policy
- Simple metrics collector
- Unit tests for schedulers
- Packaging and CI-ready workflow
