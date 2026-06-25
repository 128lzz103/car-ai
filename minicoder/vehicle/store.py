"""教学型 Mock API 的线程安全内存状态存储。"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from importlib.resources import files
from threading import RLock
from typing import Any

from .errors import VehicleConflictError, VehicleNotFoundError, VehicleValidationError
from .models import (
    normalize_location_name,
    normalize_vehicle_id,
    validate_coordinates,
    validate_radius,
    validate_temperature,
)


@dataclass(frozen=True)
class FaultSpec:
    mode: str
    remaining: int = 1
    delay_seconds: float = 0.05
    retry_after: float = 0.01


class VehicleStateStore:
    """RLock 只保证单进程线程安全，不提供多实例一致性。"""

    def __init__(
        self,
        *,
        vehicles: dict[str, Any] | None = None,
        routes: list[dict[str, Any]] | None = None,
        stations: list[dict[str, Any]] | None = None,
    ) -> None:
        self._lock = RLock()
        self._initial_vehicles = copy.deepcopy(vehicles or _load_json("vehicles.json"))
        self._routes = copy.deepcopy(routes or _load_json("routes.json"))
        self._stations = copy.deepcopy(stations or _load_json("charging_stations.json"))
        self._vehicles: dict[str, Any] = {}
        self._idempotency_results: dict[tuple[str, str], dict[str, Any]] = {}
        self._fault: FaultSpec | None = None
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._vehicles = {
                normalize_vehicle_id(vehicle_id): copy.deepcopy(value)
                for vehicle_id, value in self._initial_vehicles.items()
            }
            self._idempotency_results.clear()
            self._fault = None

    def get_vehicle(self, vehicle_id: str) -> dict[str, Any]:
        normalized = normalize_vehicle_id(vehicle_id)
        with self._lock:
            vehicle = self._vehicles.get(normalized)
            if vehicle is None:
                raise VehicleNotFoundError(
                    f"车辆 {normalized} 不存在",
                    code="VEHICLE_NOT_FOUND",
                    status_code=404,
                )
            result = copy.deepcopy(vehicle)
        result["vehicle_id"] = normalized
        return result

    def update_climate(
        self,
        vehicle_id: str,
        *,
        action: str,
        target_temperature: float | None,
        idempotency_key: str,
    ) -> dict[str, Any]:
        normalized = normalize_vehicle_id(vehicle_id)
        if not idempotency_key or len(idempotency_key) > 128:
            raise VehicleValidationError(
                "Idempotency-Key 必须为 1～128 字符",
                code="INVALID_IDEMPOTENCY_KEY",
            )
        if action not in {"turn_on", "turn_off", "set_temperature"}:
            raise VehicleValidationError("未知空调操作", code="INVALID_CLIMATE_ACTION")
        temperature = (
            validate_temperature(target_temperature) if action == "set_temperature" else None
        )

        with self._lock:
            cache_key = (normalized, idempotency_key)
            if cache_key in self._idempotency_results:
                replay = copy.deepcopy(self._idempotency_results[cache_key])
                replay["idempotent_replay"] = True
                return replay
            vehicle = self._vehicles.get(normalized)
            if vehicle is None:
                raise VehicleNotFoundError(
                    f"车辆 {normalized} 不存在",
                    code="VEHICLE_NOT_FOUND",
                    status_code=404,
                )
            if not vehicle.get("online", False):
                raise VehicleConflictError(
                    f"车辆 {normalized} 当前离线",
                    code="VEHICLE_OFFLINE",
                    status_code=409,
                )
            climate = vehicle["climate"]
            if action == "turn_on":
                climate["enabled"] = True
            elif action == "turn_off":
                climate["enabled"] = False
            else:
                climate["enabled"] = True
                climate["target_temperature"] = temperature
            result = {
                "vehicle_id": normalized,
                "action": action,
                "climate": copy.deepcopy(climate),
                "idempotent_replay": False,
            }
            self._idempotency_results[cache_key] = copy.deepcopy(result)
            return result

    def estimate_route(
        self,
        *,
        destination: str,
        origin: str | None = None,
        vehicle_id: str | None = None,
    ) -> dict[str, Any]:
        destination = normalize_location_name(destination, field="destination", max_length=120)
        vehicle: dict[str, Any] | None = None
        if vehicle_id:
            vehicle = self.get_vehicle(vehicle_id)
        if origin is None and vehicle:
            origin = vehicle["location"]["name"]
        if origin is None:
            raise VehicleValidationError(
                "origin 与 vehicle_id 至少提供一个",
                code="MISSING_ROUTE_ORIGIN",
            )
        origin = normalize_location_name(origin, field="origin")
        with self._lock:
            match = next(
                (
                    copy.deepcopy(route)
                    for route in self._routes
                    if route["origin"].casefold() == origin.casefold()
                    and route["destination"].casefold() == destination.casefold()
                ),
                None,
            )
        if match is None:
            raise VehicleNotFoundError(
                f"未找到从 {origin} 到 {destination} 的模拟路线",
                code="ROUTE_NOT_FOUND",
                status_code=404,
            )
        consumption = float(vehicle["consumption_kwh_per_100km"]) if vehicle else 17.0
        match["estimated_energy_kwh"] = round(
            float(match["distance_km"]) * consumption / 100,
            2,
        )
        return match

    def find_charging_stations(
        self,
        *,
        location: str | None = None,
        corridor: list[str] | None = None,
        vehicle_id: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        radius_km: float = 20.0,
    ) -> list[dict[str, Any]]:
        terms = [normalize_location_name(item, field="corridor") for item in corridor or []]
        if location:
            terms.append(normalize_location_name(location))
        if vehicle_id:
            terms.append(self.get_vehicle(vehicle_id)["location"]["name"])
        coordinates: tuple[float, float] | None = None
        if latitude is not None or longitude is not None:
            if latitude is None or longitude is None:
                raise VehicleValidationError(
                    "latitude 与 longitude 必须同时提供",
                    code="INVALID_COORDINATES",
                )
            coordinates = validate_coordinates(latitude, longitude)
            radius_km = validate_radius(radius_km)
        if not terms and coordinates is None:
            raise VehicleValidationError(
                "location、corridor 与 vehicle_id 至少提供一个",
                code="MISSING_STATION_LOCATION",
            )
        normalized_terms = [term.casefold() for term in terms]
        with self._lock:
            return [
                copy.deepcopy(station)
                for station in self._stations
                if (
                    any(
                        term in station["city"].casefold()
                        or term in station["location"]["name"].casefold()
                        for term in normalized_terms
                    )
                    or (
                        coordinates is not None
                        and _distance_km(
                            coordinates,
                            (
                                float(station["location"]["latitude"]),
                                float(station["location"]["longitude"]),
                            ),
                        )
                        <= radius_km
                    )
                )
            ]

    def set_fault(self, fault: FaultSpec) -> None:
        if fault.mode not in {
            "rate_limit",
            "timeout",
            "server_error",
            "stale_data",
            "vehicle_offline",
        }:
            raise VehicleValidationError("未知故障模式", code="INVALID_FAULT_MODE")
        if not 1 <= fault.remaining <= 100:
            raise VehicleValidationError("remaining 必须位于 1～100", code="INVALID_FAULT_COUNT")
        with self._lock:
            self._fault = fault

    def consume_fault(self) -> FaultSpec | None:
        with self._lock:
            fault = self._fault
            if fault is None:
                return None
            remaining = fault.remaining - 1
            self._fault = (
                FaultSpec(fault.mode, remaining, fault.delay_seconds, fault.retry_after)
                if remaining > 0
                else None
            )
            return fault


def _load_json(name: str) -> Any:
    path = files("minicoder.vehicle.data").joinpath(name)
    return json.loads(path.read_text(encoding="utf-8"))


def _distance_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*first, *second))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(value))
