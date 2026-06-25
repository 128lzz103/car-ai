"""类型化同步 Vehicle API 客户端。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import httpx

from ..retry import RetryController, RetryPolicy, parse_retry_after
from .errors import (
    VehicleAuthenticationError,
    VehicleConflictError,
    VehicleError,
    VehicleNotFoundError,
    VehiclePermissionError,
    VehicleRateLimitError,
    VehicleRetryExhaustedError,
    VehicleServiceError,
    VehicleTimeoutError,
    VehicleValidationError,
)
from .models import (
    ChargingStation,
    RouteEstimate,
    VehicleStatus,
    normalize_location_name,
    normalize_vehicle_id,
    validate_coordinates,
    validate_radius,
    validate_temperature,
)


class VehicleClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: float = 5.0,
        retry_policy: RetryPolicy | None = None,
        retry_controller: RetryController | None = None,
        client: httpx.Client | None = None,
        allow_remote: bool = False,
        idempotency_key_factory: Callable[[], str] | None = None,
    ) -> None:
        self.base_url = _validate_base_url(base_url, allow_remote=allow_remote)
        self.token = token.strip()
        if not self.token:
            raise VehicleValidationError("车辆 API Token 不能为空", code="MISSING_API_TOKEN")
        if timeout <= 0:
            raise VehicleValidationError("车辆 API timeout 必须大于 0", code="INVALID_TIMEOUT")
        self.timeout = timeout
        self.retry = retry_controller or RetryController(retry_policy or RetryPolicy(max_retries=2))
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        if idempotency_key_factory is None:
            from uuid import uuid4

            def new_idempotency_key() -> str:
                return uuid4().hex

            idempotency_key_factory = new_idempotency_key
        self._idempotency_key_factory = idempotency_key_factory

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> VehicleClient:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def get_vehicle_status(self, vehicle_id: str) -> VehicleStatus:
        vehicle_id = normalize_vehicle_id(vehicle_id)
        data = self._request("GET", f"/v1/vehicles/{vehicle_id}/status", retryable=True)
        vehicle = _object_field(data, "vehicle")
        vehicle["observed_at"] = str(data.get("observed_at", ""))
        return VehicleStatus.from_dict(vehicle)

    def estimate_route(
        self,
        *,
        destination: str,
        origin: str | None = None,
        vehicle_id: str | None = None,
        departure_time: str | None = None,
    ) -> RouteEstimate:
        payload: dict[str, Any] = {
            "destination": normalize_location_name(destination, field="destination", max_length=120)
        }
        if origin:
            payload["origin"] = normalize_location_name(origin, field="origin")
        if vehicle_id:
            payload["vehicle_id"] = normalize_vehicle_id(vehicle_id)
        if departure_time:
            payload["departure_time"] = departure_time
        data = self._request("POST", "/v1/routes/estimate", json=payload, retryable=True)
        return RouteEstimate.from_dict(_object_field(data, "route"))

    def get_charging_stations(
        self,
        *,
        location: str | None = None,
        corridor: list[str] | None = None,
        vehicle_id: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        radius_km: float = 20.0,
    ) -> list[ChargingStation]:
        params: list[tuple[str, str]] = []
        if location:
            params.append(("location", normalize_location_name(location)))
        for item in corridor or []:
            params.append(("corridor", normalize_location_name(item, field="corridor")))
        if vehicle_id:
            params.append(("vehicle_id", normalize_vehicle_id(vehicle_id)))
        if latitude is not None or longitude is not None:
            if latitude is None or longitude is None:
                raise VehicleValidationError(
                    "latitude 与 longitude 必须同时提供", code="INVALID_COORDINATES"
                )
            latitude, longitude = validate_coordinates(latitude, longitude)
            params.extend(
                [
                    ("latitude", str(latitude)),
                    ("longitude", str(longitude)),
                    ("radius_km", str(validate_radius(radius_km))),
                ]
            )
        data = self._request(
            "GET",
            "/v1/charging-stations",
            params=params,
            retryable=True,
        )
        stations = data.get("stations") if isinstance(data, dict) else None
        if not isinstance(stations, list):
            raise VehicleValidationError("充电站响应缺少 stations", code="INVALID_RESPONSE")
        return [ChargingStation.from_dict(item) for item in stations]

    def control_climate(
        self,
        vehicle_id: str,
        *,
        action: str,
        target_temperature: float | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        vehicle_id = normalize_vehicle_id(vehicle_id)
        if action not in {"turn_on", "turn_off", "set_temperature"}:
            raise VehicleValidationError("未知空调操作", code="INVALID_CLIMATE_ACTION")
        payload: dict[str, Any] = {"action": action}
        if action == "set_temperature":
            payload["target_temperature"] = validate_temperature(target_temperature)
        key = idempotency_key or self._idempotency_key_factory()
        if not key or len(key) > 128:
            raise VehicleValidationError(
                "Idempotency-Key 必须为 1～128 字符", code="INVALID_IDEMPOTENCY_KEY"
            )
        data = self._request(
            "PUT",
            f"/v1/vehicles/{vehicle_id}/climate",
            json=payload,
            headers={"Idempotency-Key": key},
            retryable=True,
        )
        return _object_field(data, "result")

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: list[tuple[str, str]] | None = None,
        headers: dict[str, str] | None = None,
        retryable: bool,
    ) -> dict[str, Any]:
        request_headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            **(headers or {}),
        }
        for attempt_index in range(self.retry.policy.max_attempts):
            try:
                response = self._client.request(
                    method,
                    self.base_url + path,
                    json=json,
                    params=params,
                    headers=request_headers,
                )
            except httpx.TimeoutException as error:
                failure: VehicleError = VehicleTimeoutError(
                    f"车辆 API 请求超时：{error}",
                    code="VEHICLE_TIMEOUT",
                    retryable=True,
                )
            except httpx.TransportError as error:
                failure = VehicleServiceError(
                    f"车辆 API 网络错误：{error}",
                    code="VEHICLE_TRANSPORT_ERROR",
                    retryable=True,
                )
            else:
                if response.status_code < 400:
                    try:
                        payload = response.json()
                    except ValueError as error:
                        raise VehicleServiceError(
                            "车辆 API 返回了无效 JSON",
                            code="INVALID_RESPONSE",
                            status_code=response.status_code,
                        ) from error
                    if not isinstance(payload, dict):
                        raise VehicleServiceError(
                            "车辆 API 响应根节点必须是对象",
                            code="INVALID_RESPONSE",
                            status_code=response.status_code,
                        )
                    return payload
                failure = _error_from_response(response)

            if not retryable or not failure.retryable:
                raise failure
            if attempt_index >= self.retry.policy.max_retries:
                raise VehicleRetryExhaustedError(attempt_index + 1, failure) from failure
            self.retry.wait(
                attempt_index,
                str(failure),
                status_code=failure.status_code,
                retry_after=failure.retry_after,
            )
        raise AssertionError("unreachable")


def _validate_base_url(value: str, *, allow_remote: bool) -> str:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise VehicleValidationError("车辆 API URL 必须是 http(s) URL", code="INVALID_API_URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise VehicleValidationError(
            "车辆 API URL 不得包含凭据、查询或 fragment", code="INVALID_API_URL"
        )
    if not allow_remote and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise VehiclePermissionError(
            "Mock Vehicle API 默认只允许 localhost；远程服务需显式授权",
            code="REMOTE_API_DENIED",
        )
    return value.strip().rstrip("/")


def _object_field(value: dict[str, Any], name: str) -> dict[str, Any]:
    item = value.get(name)
    if not isinstance(item, dict):
        raise VehicleValidationError(f"响应缺少对象字段 {name}", code="INVALID_RESPONSE")
    return dict(item)


def _error_from_response(response: httpx.Response) -> VehicleError:
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    raw_error = payload.get("error") if isinstance(payload, dict) else None
    item = raw_error if isinstance(raw_error, dict) else {}
    status = response.status_code
    code = str(item.get("code") or f"HTTP_{status}")
    message = str(item.get("message") or f"车辆 API 请求失败：HTTP {status}")
    request_id = str(item.get("request_id") or response.headers.get("X-Request-ID", ""))
    retry_after = parse_retry_after(response.headers.get("Retry-After"))
    retryable = bool(item.get("retryable", status == 429 or status >= 500))
    common = {
        "code": code,
        "status_code": status,
        "retryable": retryable,
        "retry_after": retry_after,
        "request_id": request_id,
    }
    if status == 401:
        return VehicleAuthenticationError(message, **common)
    if status == 403:
        return VehiclePermissionError(message, **common)
    if status == 404:
        return VehicleNotFoundError(message, **common)
    if status in {400, 422}:
        return VehicleValidationError(message, **common)
    if status == 409:
        return VehicleConflictError(message, **common)
    if status == 429:
        return VehicleRateLimitError(message, **common)
    return VehicleServiceError(message, **common)
