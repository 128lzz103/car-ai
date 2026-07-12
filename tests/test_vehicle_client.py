"""VehicleClient 错误分类、重试、类型转换和权限策略测试。"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from minicoder.retry import RetryController, RetryPolicy
from minicoder.security import PermissionMode
from minicoder.vehicle.client import VehicleClient
from minicoder.vehicle.errors import (
    VehicleAuthenticationError,
    VehiclePermissionError,
    VehicleRetryExhaustedError,
    VehicleValidationError,
)
from minicoder.vehicle.mock_server import create_app
from minicoder.vehicle.policy import VehicleActionPolicy


def _client(app=None, **kwargs):
    transport = TestClient(app or create_app(token="token", test_mode=True))
    return VehicleClient(
        "http://localhost",
        "token",
        client=transport,
        retry_controller=kwargs.pop(
            "retry_controller",
            RetryController(RetryPolicy(max_retries=1, jitter=0), sleep=lambda _delay: None),
        ),
        idempotency_key_factory=lambda: "generated-key",
        **kwargs,
    )


def test_client_converts_vehicle_route_station_and_climate_models():
    client = _client()

    status = client.get_vehicle_status("a102")
    route = client.estimate_route(vehicle_id="A102", destination="上海虹桥站")
    stations = client.get_charging_stations(corridor=list(route.corridor))
    climate = client.control_climate(
        "A102",
        action="set_temperature",
        target_temperature=24,
    )

    assert status.vehicle_id == "A102"
    assert route.distance_km == 295
    assert any(station.station_id == "CS-WX-001" for station in stations)
    assert climate["climate"]["target_temperature"] == 24


def test_client_retries_rate_limit_and_respects_retry_after():
    sleeps = []
    retry = RetryController(RetryPolicy(max_retries=1, jitter=0), sleep=sleeps.append)
    app = create_app(token="token", test_mode=True)
    with TestClient(app) as setup:
        setup.post(
            "/__test__/faults",
            headers={"Authorization": "Bearer token"},
            json={"mode": "rate_limit", "remaining": 1, "retry_after": 0.25},
        )
    client = _client(app, retry_controller=retry)

    assert client.get_vehicle_status("A102").vehicle_id == "A102"
    assert sleeps == [0.25]


def test_client_classifies_authentication_and_retry_exhaustion():
    unauthorized = VehicleClient(
        "http://localhost",
        "wrong",
        client=TestClient(create_app(token="right")),
    )
    with pytest.raises(VehicleAuthenticationError):
        unauthorized.get_vehicle_status("A102")

    def fail(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            request=_request,
            json={"error": {"code": "DOWN", "message": "down", "retryable": True}},
        )

    retry = RetryController(RetryPolicy(max_retries=1, jitter=0), sleep=lambda _delay: None)
    unavailable = VehicleClient(
        "http://localhost",
        "token",
        client=httpx.Client(transport=httpx.MockTransport(fail)),
        retry_controller=retry,
    )
    with pytest.raises(VehicleRetryExhaustedError) as error:
        unavailable.get_vehicle_status("A102")
    assert error.value.attempts == 2


def test_client_rejects_remote_url_and_invalid_control():
    with pytest.raises(VehiclePermissionError):
        VehicleClient("https://vehicle.example.com", "token")
    with pytest.raises(VehicleValidationError):
        VehicleClient("not-a-url", "token")

    client = _client()
    with pytest.raises(VehicleValidationError):
        client.control_climate("A102", action="set_temperature", target_temperature=40)


def test_vehicle_action_policy_separates_queries_and_controls():
    requests = []
    policy = VehicleActionPolicy(
        mode=PermissionMode.ASK,
        confirm=lambda request: requests.append(request) or True,
        allowed_vehicle_ids=frozenset({"A102"}),
    )

    assert policy.authorize_query("A102").allowed is True
    assert policy.authorize_query("A103").allowed is False
    assert policy.authorize_control("A102", action="turn_on", description="打开空调").allowed
    assert requests[0].vehicle_id == "A102"

    readonly = VehicleActionPolicy(mode=PermissionMode.ALLOW, read_only=True)
    assert not readonly.authorize_control("A102", action="turn_on", description="打开").allowed
