"""车辆 API 的稳定错误分类。"""

from __future__ import annotations


class VehicleError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "VEHICLE_ERROR",
        status_code: int | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
        request_id: str = "",
    ) -> None:
        self.code = code
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after = retry_after
        self.request_id = request_id
        super().__init__(message)


class VehicleAuthenticationError(VehicleError):
    pass


class VehiclePermissionError(VehicleError):
    pass


class VehicleValidationError(VehicleError, ValueError):
    pass


class VehicleNotFoundError(VehicleError):
    pass


class VehicleConflictError(VehicleError):
    pass


class VehicleRateLimitError(VehicleError):
    pass


class VehicleServiceError(VehicleError):
    pass


class VehicleTimeoutError(VehicleError):
    pass


class VehicleRetryExhaustedError(VehicleError):
    def __init__(self, attempts: int, last_error: VehicleError) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"车辆 API 请求重试耗尽（{attempts} 次尝试）：{last_error}",
            code="VEHICLE_RETRY_EXHAUSTED",
            status_code=last_error.status_code,
            retryable=False,
            request_id=last_error.request_id,
        )
