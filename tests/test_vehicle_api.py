"""FastAPI 教学型 Mock Vehicle API 的契约与故障隔离测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from minicoder.vehicle.mock_server import create_app

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _client(*, test_mode=False):
    return TestClient(create_app(token=TOKEN, test_mode=test_mode))


def test_health_and_authentication_boundaries():
    with _client() as client:
        assert client.get("/v1/health").json()["status"] == "ok"
        unauthorized = client.get("/v1/vehicles/A102/status")
        authorized = client.get("/v1/vehicles/A102/status", headers=AUTH)

    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["code"] == "AUTHENTICATION_FAILED"
    assert authorized.status_code == 200
    assert authorized.json()["vehicle"]["estimated_range_km"] == 142
    assert authorized.headers["X-Request-ID"]


def test_status_route_station_and_location_validation():
    with _client() as client:
        status = client.get("/v1/vehicles/a102/status", headers=AUTH)
        route = client.post(
            "/v1/routes/estimate",
            headers=AUTH,
            json={"vehicle_id": "A102", "destination": "上海虹桥站"},
        )
        stations = client.get(
            "/v1/charging-stations",
            headers=AUTH,
            params=[("corridor", "无锡"), ("corridor", "苏州")],
        )
        bad_vehicle = client.get("/v1/vehicles/bad%20id/status", headers=AUTH)
        bad_route = client.post(
            "/v1/routes/estimate",
            headers=AUTH,
            json={"destination": "南京南站"},
        )
        bad_coordinates = client.get(
            "/v1/charging-stations",
            headers=AUTH,
            params={"latitude": 91, "longitude": 0},
        )

    assert status.json()["vehicle"]["vehicle_id"] == "A102"
    assert route.json()["route"]["distance_km"] == 295
    assert len(stations.json()["stations"]) == 2
    assert bad_vehicle.status_code == 400
    assert bad_route.status_code == 422
    assert bad_coordinates.status_code == 400


def test_climate_control_requires_idempotency_and_updates_state():
    with _client() as client:
        missing_key = client.put(
            "/v1/vehicles/A102/climate",
            headers=AUTH,
            json={"action": "turn_on"},
        )
        headers = {**AUTH, "Idempotency-Key": "climate-001"}
        first = client.put(
            "/v1/vehicles/A102/climate",
            headers=headers,
            json={"action": "set_temperature", "target_temperature": 23},
        )
        replay = client.put(
            "/v1/vehicles/A102/climate",
            headers=headers,
            json={"action": "set_temperature", "target_temperature": 23},
        )
        status = client.get("/v1/vehicles/A102/status", headers=AUTH)

    assert missing_key.status_code == 400
    assert first.json()["result"]["idempotent_replay"] is False
    assert replay.json()["result"]["idempotent_replay"] is True
    assert status.json()["vehicle"]["climate"]["target_temperature"] == 23


def test_fault_routes_exist_only_in_test_mode():
    with _client(test_mode=False) as production:
        assert production.post("/__test__/reset", headers=AUTH).status_code == 404

    with _client(test_mode=True) as client:
        configured = client.post(
            "/__test__/faults",
            headers=AUTH,
            json={"mode": "rate_limit", "remaining": 1, "retry_after": 0},
        )
        limited = client.get("/v1/vehicles/A102/status", headers=AUTH)
        recovered = client.get("/v1/vehicles/A102/status", headers=AUTH)
        reset = client.post("/__test__/reset", headers=AUTH)

    assert configured.status_code == 200
    assert limited.status_code == 429
    assert limited.json()["error"]["retryable"] is True
    assert recovered.status_code == 200
    assert reset.status_code == 200


def test_test_mode_faults_cover_offline_stale_and_service_error():
    with _client(test_mode=True) as client:
        for mode, expected in (("vehicle_offline", 409), ("server_error", 503)):
            client.post("/__test__/faults", headers=AUTH, json={"mode": mode})
            assert client.get("/v1/vehicles/A102/status", headers=AUTH).status_code == expected

        client.post("/__test__/faults", headers=AUTH, json={"mode": "stale_data"})
        stale = client.get("/v1/vehicles/A102/status", headers=AUTH)

    assert stale.status_code == 200
    assert stale.json()["stale"] is True
