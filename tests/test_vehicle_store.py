"""车辆模型、输入验证和线程安全状态存储测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from minicoder.vehicle.errors import (
    VehicleConflictError,
    VehicleNotFoundError,
    VehicleValidationError,
)
from minicoder.vehicle.models import (
    normalize_location_name,
    normalize_vehicle_id,
    validate_coordinates,
    validate_radius,
    validate_temperature,
)
from minicoder.vehicle.store import FaultSpec, VehicleStateStore


def test_shared_vehicle_and_location_validation():
    assert normalize_vehicle_id(" a102 ") == "A102"
    assert normalize_location_name(" 南京南站 ") == "南京南站"
    assert validate_coordinates(31.9, 118.8) == (31.9, 118.8)
    assert validate_radius(20) == 20
    assert validate_temperature("22") == 22

    for value in ("", "bad id", "../A102", "A" * 33):
        with pytest.raises(VehicleValidationError):
            normalize_vehicle_id(value)
    with pytest.raises(VehicleValidationError):
        normalize_location_name("bad\x00location")
    with pytest.raises(VehicleValidationError):
        validate_coordinates(91, 0)
    with pytest.raises(VehicleValidationError):
        validate_radius(0)
    with pytest.raises(VehicleValidationError):
        validate_temperature(31)


def test_store_reads_vehicle_route_and_stations():
    store = VehicleStateStore()

    vehicle = store.get_vehicle("a102")
    route = store.estimate_route(vehicle_id="A102", destination="上海虹桥站")
    stations = store.find_charging_stations(corridor=route["corridor"])
    nearby = store.find_charging_stations(
        latitude=31.57,
        longitude=120.42,
        radius_km=5,
    )

    assert vehicle["vehicle_id"] == "A102"
    assert route["distance_km"] == 295
    assert route["estimated_energy_kwh"] == 47.79
    assert any(item["station_id"] == "CS-WX-001" for item in stations)
    assert [item["station_id"] for item in nearby] == ["CS-WX-001"]


def test_store_reports_unknown_and_invalid_queries():
    store = VehicleStateStore()
    with pytest.raises(VehicleNotFoundError):
        store.get_vehicle("A999")
    with pytest.raises(VehicleNotFoundError):
        store.estimate_route(vehicle_id="A102", destination="火星")
    with pytest.raises(VehicleValidationError):
        store.estimate_route(destination="南京南站")
    with pytest.raises(VehicleValidationError):
        store.find_charging_stations()
    with pytest.raises(VehicleValidationError):
        store.find_charging_stations(latitude=31.0)


def test_climate_update_is_idempotent_and_thread_safe():
    store = VehicleStateStore()

    def update(_index: int):
        return store.update_climate(
            "A102",
            action="set_temperature",
            target_temperature=23,
            idempotency_key="same-request",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(update, range(20)))

    assert sum(not item["idempotent_replay"] for item in results) == 1
    assert store.get_vehicle("A102")["climate"] == {
        "enabled": True,
        "target_temperature": 23,
    }
    with pytest.raises(VehicleConflictError):
        store.update_climate(
            "A104",
            action="turn_on",
            target_temperature=None,
            idempotency_key="offline",
        )


def test_fault_state_is_locked_and_consumed():
    store = VehicleStateStore()
    store.set_fault(FaultSpec("rate_limit", remaining=2))
    assert store.consume_fault().mode == "rate_limit"
    assert store.consume_fault().mode == "rate_limit"
    assert store.consume_fault() is None
    with pytest.raises(VehicleValidationError):
        store.set_fault(FaultSpec("unknown"))
    with pytest.raises(VehicleValidationError):
        store.set_fault(FaultSpec("timeout", remaining=0))
