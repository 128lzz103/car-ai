"""重试策略、Retry-After 和错误分类单测。"""

from datetime import datetime, timezone

from minicoder.retry import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    RetryPolicy,
    error_from_response,
    parse_retry_after,
)


class Response:
    def __init__(self, status_code, *, text="", headers=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}


def test_exponential_delay_without_jitter():
    policy = RetryPolicy(base_delay=1, max_delay=4, jitter=0)
    assert [policy.delay(i, random_value=0.5) for i in range(5)] == [1, 2, 4, 4, 4]


def test_jitter_uses_injected_random_value():
    policy = RetryPolicy(base_delay=5, max_delay=10, jitter=0.2)
    assert policy.delay(0, random_value=0) == 4
    assert policy.delay(0, random_value=1) == 6
    assert policy.delay(2, random_value=1) == 10


def test_retry_after_seconds_and_cap():
    policy = RetryPolicy(max_retry_after=60)
    assert parse_retry_after("5") == 5
    assert policy.delay(0, retry_after=120, random_value=0.5) == 60


def test_retry_after_http_date():
    now = datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc)
    assert parse_retry_after("Sun, 02 Aug 2026 12:00:05 GMT", now=now) == 5


def test_error_classification():
    auth = error_from_response(Response(401, text="bad key"))
    limited = error_from_response(Response(429, headers={"Retry-After": "3"}))
    assert isinstance(auth, ProviderAuthenticationError)
    assert auth.retryable is False
    assert isinstance(limited, ProviderRateLimitError)
    assert limited.retryable is True
    assert limited.retry_after == 3
