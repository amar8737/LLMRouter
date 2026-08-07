import time

import pytest

from llmrouterx.retry.circuit_breaker import CircuitBreaker, CircuitState


@pytest.fixture
def breaker():
    return CircuitBreaker(failure_threshold=3, reset_timeout=10.0, half_open_max_calls=1)


def test_initial_state_closed(breaker):
    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_opens_after_threshold(breaker):
    for _ in range(3):
        breaker.record_failure()

    assert breaker.state == CircuitState.OPEN
    assert breaker.allow_request() is False
    assert "failures=3" in repr(breaker)


def test_does_not_open_below_threshold(breaker):
    for _ in range(2):
        breaker.record_failure()

    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_record_success_closes_breaker(breaker):
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    breaker.record_success()

    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_transitions_to_half_open_after_timeout(breaker, monkeypatch):
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    current = time.monotonic()
    monkeypatch.setattr("llmrouterx.retry.circuit_breaker.time.monotonic", lambda: current + 10.5)

    assert breaker.state == CircuitState.HALF_OPEN


def test_half_open_allows_one_trial_request(breaker, monkeypatch):
    for _ in range(3):
        breaker.record_failure()

    current = time.monotonic()
    monkeypatch.setattr("llmrouterx.retry.circuit_breaker.time.monotonic", lambda: current + 11.0)

    assert breaker.allow_request() is True
    assert breaker.allow_request() is False


def test_half_open_success_closes_circuit(breaker, monkeypatch):
    for _ in range(3):
        breaker.record_failure()

    current = time.monotonic()
    monkeypatch.setattr("llmrouterx.retry.circuit_breaker.time.monotonic", lambda: current + 11.0)

    assert breaker.allow_request() is True
    breaker.record_success()

    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_half_open_failure_reopens_circuit(breaker, monkeypatch):
    for _ in range(3):
        breaker.record_failure()

    current = time.monotonic()
    monkeypatch.setattr("llmrouterx.retry.circuit_breaker.time.monotonic", lambda: current + 11.0)

    assert breaker.allow_request() is True
    breaker.record_failure()

    assert breaker.state == CircuitState.OPEN
    assert breaker.allow_request() is False


def test_reset_returns_to_closed(breaker):
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    breaker.reset()

    assert breaker.state == CircuitState.CLOSED
    assert breaker.allow_request() is True


def test_half_open_max_calls_limits_concurrent_probes(monkeypatch):
    b = CircuitBreaker(
        failure_threshold=2,
        reset_timeout=5.0,
        half_open_max_calls=2,
    )
    b.record_failure()
    b.record_failure()

    current = time.monotonic()
    monkeypatch.setattr("llmrouterx.retry.circuit_breaker.time.monotonic", lambda: current + 6.0)

    assert b.allow_request() is True
    assert b.allow_request() is True
    assert b.allow_request() is False


def test_single_failure_not_enough_to_open():
    b = CircuitBreaker(failure_threshold=5, reset_timeout=30.0)
    b.record_failure()
    assert b.state == CircuitState.CLOSED
