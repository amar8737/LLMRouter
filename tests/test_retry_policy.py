import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from llmrouterx.retry.exponential import ExponentialRetry, HTTPError


@pytest.fixture
def retry():
    return ExponentialRetry(max_retries=3, base=1.0, factor=2.0, max_backoff=60.0, jitter=False)


def test_should_retry_retryable_http_statuses(retry):
    for status in (429, 500, 502, 503, 504):
        assert retry.should_retry(HTTPError(status, "boom"), attempt=1) is True


def test_should_retry_rejects_non_retryable_status(retry):
    assert retry.should_retry(HTTPError(400, "bad"), attempt=1) is False
    assert retry.should_retry(HTTPError(404, "nf"), attempt=1) is False


def test_should_retry_connection_and_timeout(retry):
    assert retry.should_retry(ConnectionError("no route"), attempt=1) is True
    assert retry.should_retry(asyncio.TimeoutError(), attempt=1) is True


def test_should_retry_rejects_plain_exceptions(retry):
    assert retry.should_retry(ValueError("nope"), attempt=1) is False


def test_should_retry_respects_max_retries(retry):
    assert retry.should_retry(HTTPError(500, "x"), attempt=3) is False


def test_backoff_exponential_without_headers(retry):
    assert retry.get_backoff(HTTPError(500, "x"), attempt=1) == 1.0
    assert retry.get_backoff(HTTPError(500, "x"), attempt=2) == 2.0
    assert retry.get_backoff(HTTPError(500, "x"), attempt=3) == 4.0


def test_backoff_respects_max_backoff():
    r = ExponentialRetry(max_retries=5, base=1.0, factor=2.0, max_backoff=10.0, jitter=False)
    assert r.get_backoff(HTTPError(500, "x"), attempt=5) == 10.0
    assert r.get_backoff(HTTPError(500, "x"), attempt=10) == 10.0


def test_backoff_uses_retry_after_seconds():
    r = ExponentialRetry(max_retries=3, base=1.0, factor=2.0, max_backoff=60.0, jitter=False)
    exc = HTTPError(429, "slow", headers={"Retry-After": "2.5"})
    assert r.get_backoff(exc, attempt=1) == 2.5


def test_backoff_uses_retry_after_http_date():
    r = ExponentialRetry(max_retries=3, base=1.0, factor=2.0, max_backoff=60.0, jitter=False)
    future = datetime.now(timezone.utc) + timedelta(seconds=3)
    exc = HTTPError(
        429, "slow", headers={"Retry-After": future.strftime("%a, %d %b %Y %H:%M:%S GMT")}
    )
    backoff = r.get_backoff(exc, attempt=1)
    assert 2.0 <= backoff <= 4.0


def test_backoff_ignores_invalid_retry_after():
    r = ExponentialRetry(max_retries=3, base=1.0, factor=2.0, max_backoff=60.0, jitter=False)
    exc = HTTPError(429, "slow", headers={"Retry-After": "garbage"})
    assert r.get_backoff(exc, attempt=1) >= 1.0


def test_retry_after_capped_by_max_backoff():
    r = ExponentialRetry(max_retries=3, base=1.0, factor=2.0, max_backoff=5.0, jitter=False)
    exc = HTTPError(429, "slow", headers={"Retry-After": "100"})
    assert r.get_backoff(exc, attempt=1) == 5.0


def test_rate_limit_min_backoff_applies_when_no_header():
    r = ExponentialRetry(
        max_retries=3,
        base=0.1,
        factor=1.0,
        max_backoff=60.0,
        jitter=False,
        rate_limit_min_backoff=30.0,
    )
    exc = HTTPError(429, "limited")
    assert r.get_backoff(exc, attempt=1) == 30.0


def test_jitter_scales_backoff():
    r = ExponentialRetry(max_retries=3, base=1.0, factor=2.0, max_backoff=60.0, jitter=True)
    for attempt in range(1, 4):
        value = r.get_backoff(HTTPError(500, "x"), attempt)
        expected = 1.0 * (2.0 ** (attempt - 1))
        assert 0.5 * expected <= value <= 1.5 * expected


@pytest.mark.asyncio
async def test_wait_sleeps_for_backoff():
    r = ExponentialRetry(max_retries=3, base=0.05, factor=1.0, max_backoff=1.0, jitter=False)
    started = asyncio.get_event_loop().time()
    await r.wait(HTTPError(500, "x"), attempt=1)
    elapsed = asyncio.get_event_loop().time() - started
    assert elapsed >= 0.04


def test_wait_default_in_base_retry_does_not_sleep():
    from llmrouterx.retry.base import BaseRetry

    class NoDelay(BaseRetry):
        def should_retry(self, exc, attempt):
            return True

    policy = NoDelay()
    assert policy.get_backoff(ValueError("x"), attempt=1) == 0.0


def test_http_error_repr_includes_status():
    exc = HTTPError(500, "server exploded")
    assert str(exc) == "500: server exploded"
    assert exc.status_code == 500
    assert exc.headers == {}
