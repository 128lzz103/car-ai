"""Provider 重试策略、错误分类和 Retry-After 解析。"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 8.0
    jitter: float = 0.2
    max_retry_after: float = 60.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries 不能为负数")
        if self.base_delay < 0 or self.max_delay < 0 or self.max_retry_after < 0:
            raise ValueError("重试等待时间不能为负数")
        if not 0 <= self.jitter <= 1:
            raise ValueError("jitter 必须位于 0 到 1 之间")

    @property
    def max_attempts(self) -> int:
        return self.max_retries + 1

    def delay(
        self,
        retry_index: int,
        *,
        retry_after: float | None = None,
        random_value: float | None = None,
    ) -> float:
        if retry_after is not None:
            return min(self.max_retry_after, max(0.0, retry_after))
        base = min(self.max_delay, self.base_delay * (2**retry_index))
        value = random.random() if random_value is None else random_value
        factor = 1 + self.jitter * (2 * value - 1)
        return min(self.max_delay, max(0.0, base * factor))


@dataclass(frozen=True)
class RetryEvent:
    attempt: int
    max_attempts: int
    delay: float
    reason: str
    status_code: int | None = None


class ProviderError(RuntimeError):
    """所有可预期 Provider 错误的基类。"""


class ProviderHTTPError(ProviderError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        response_body: str = "",
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        self.status_code = status_code
        self.response_body = response_body
        self.retryable = retryable
        self.retry_after = retry_after
        detail = f"HTTP {status_code}: {message}"
        if response_body:
            detail += f" ({response_body})"
        super().__init__(detail)


class ProviderAuthenticationError(ProviderHTTPError):
    pass


class ProviderPermissionError(ProviderHTTPError):
    pass


class ProviderInvalidRequestError(ProviderHTTPError):
    pass


class ProviderRateLimitError(ProviderHTTPError):
    pass


class ProviderServerError(ProviderHTTPError):
    pass


class ProviderRetryExhaustedError(ProviderError):
    def __init__(self, attempts: int, last_error: BaseException) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Provider 请求重试耗尽({attempts} 次尝试): {last_error}")


class StreamInterruptedError(ProviderError):
    def __init__(self, partial_text: str, cause: BaseException) -> None:
        self.partial_text = partial_text
        self.cause = cause
        super().__init__(
            f"流式输出已中断，已输出 {len(partial_text)} 个字符；为避免重复内容未自动重试: {cause}"
        )


class IncompleteStreamError(ProviderError):
    """连接结束但 Provider 没有发送协议结束标志。"""


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        target = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    return max(0.0, (target - current).total_seconds())


def error_from_response(response: Any) -> ProviderHTTPError:
    status = int(response.status_code)
    try:
        raw_body = response.text
    except httpx.ResponseNotRead:
        try:
            raw_body = response.read().decode(errors="replace")
        except (OSError, httpx.HTTPError):
            raw_body = ""
    body = str(raw_body or "").strip().replace("\n", " ")[:300]
    retry_after = parse_retry_after(getattr(response, "headers", {}).get("Retry-After"))
    common = {
        "status_code": status,
        "response_body": body,
        "retryable": status in RETRYABLE_STATUS_CODES or status >= 500,
        "retry_after": retry_after,
    }
    if status == 401:
        return ProviderAuthenticationError("认证失败，请检查 API Key", **common)
    if status == 403:
        return ProviderPermissionError("当前凭据没有访问权限", **common)
    if status == 404:
        return ProviderInvalidRequestError("端点或模型不存在", **common)
    if status in {400, 422}:
        return ProviderInvalidRequestError("请求参数或工具 Schema 无效", **common)
    if status == 429:
        return ProviderRateLimitError("请求受到限流", **common)
    if status in RETRYABLE_STATUS_CODES or status >= 500:
        return ProviderServerError("Provider 暂时不可用", **common)
    return ProviderHTTPError("Provider 请求失败", **common)


RetryCallback = Callable[[RetryEvent], None]
Sleep = Callable[[float], None]
RandomValue = Callable[[], float]


class RetryController:
    """计算等待并发送重试事件；是否重试由 Provider 决定。"""

    def __init__(
        self,
        policy: RetryPolicy | None = None,
        *,
        sleep: Sleep = time.sleep,
        random_value: RandomValue = random.random,
    ) -> None:
        self.policy = policy or RetryPolicy()
        self.sleep = sleep
        self.random_value = random_value
        self.on_retry: RetryCallback | None = None

    def wait(
        self,
        failed_attempt_index: int,
        reason: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> RetryEvent:
        delay = self.policy.delay(
            failed_attempt_index,
            retry_after=retry_after,
            random_value=None if retry_after is not None else self.random_value(),
        )
        event = RetryEvent(
            attempt=failed_attempt_index + 2,
            max_attempts=self.policy.max_attempts,
            delay=delay,
            reason=reason,
            status_code=status_code,
        )
        if self.on_retry:
            self.on_retry(event)
        self.sleep(delay)
        return event


def is_retryable_exception(error: BaseException) -> bool:
    if isinstance(error, ProviderHTTPError):
        return error.retryable
    return isinstance(
        error,
        (httpx.TimeoutException, httpx.TransportError, IncompleteStreamError),
    )
