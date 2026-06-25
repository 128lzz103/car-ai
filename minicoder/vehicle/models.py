"""车辆 API 的核心模型与共享输入校验。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .errors import VehicleValidationError

_VEHICLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def normalize_vehicle_id(value: str) -> str:
    if not isinstance(value, str):
        raise VehicleValidationError("vehicle_id 必须是字符串", code="INVALID_VEHICLE_ID")
    normalized = value.strip()
    if not _VEHICLE_ID_RE.fullmatch(normalized):
        raise VehicleValidationError(
            "vehicle_id 必须为 1～32 位字母、数字、_ 或 -",
            code="INVALID_VEHICLE_ID",
        )
    return normalized.upper()


def normalize_location_name(value: str, *, field: str = "location", max_length: int = 100) -> str:
    if not isinstance(value, str):
        raise VehicleValidationError(f"{field} 必须是字符串", code="INVALID_LOCATION")
    normalized = value.strip()
    if not normalized or len(normalized) > max_length or _CONTROL_RE.search(normalized):
        raise VehicleValidationError(
            f"{field} 长度必须为 1～{max_length} 且不能包含控制字符",
            code="INVALID_LOCATION",
        )
    return normalized


def validate_coordinates(latitude: float, longitude: float) -> tuple[float, float]:
    if isinstance(latitude, bool) or isinstance(longitude, bool):
        raise VehicleValidationError("坐标必须是数字", code="INVALID_COORDINATES")
    try:
        lat, lon = float(latitude), float(longitude)
    except (TypeError, ValueError) as error:
        raise VehicleValidationError("坐标必须是数字", code="INVALID_COORDINATES") from error
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise VehicleValidationError("坐标超出有效范围", code="INVALID_COORDINATES")
    return lat, lon


def validate_radius(value: float) -> float:
    if isinstance(value, bool):
        raise VehicleValidationError("radius_km 必须是数字", code="INVALID_RADIUS")
    try:
        radius = float(value)
    except (TypeError, ValueError) as error:
        raise VehicleValidationError("radius_km 必须是数字", code="INVALID_RADIUS") from error
    if not 0.1 <= radius <= 100:
        raise VehicleValidationError("radius_km 必须位于 0.1～100", code="INVALID_RADIUS")
    return radius


def validate_temperature(value: float) -> float:
    if isinstance(value, bool):
        raise VehicleValidationError("温度必须是数字", code="INVALID_TEMPERATURE")
    try:
        temperature = float(value)
    except (TypeError, ValueError) as error:
        raise VehicleValidationError("温度必须是数字", code="INVALID_TEMPERATURE") from error
    if not 16 <= temperature <= 30:
        raise VehicleValidationError("温度必须位于 16～30℃", code="INVALID_TEMPERATURE")
    return temperature


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Location:
        name = normalize_location_name(value.get("name", ""))
        latitude, longitude = validate_coordinates(value.get("latitude"), value.get("longitude"))
        return cls(name, latitude, longitude)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClimateState:
    enabled: bool
    target_temperature: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ClimateState:
        enabled = value.get("enabled")
        if not isinstance(enabled, bool):
            raise VehicleValidationError("climate.enabled 必须是布尔值", code="INVALID_RESPONSE")
        return cls(enabled, validate_temperature(value.get("target_temperature")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VehicleStatus:
    vehicle_id: str
    model: str
    battery_soc: float
    estimated_range_km: float
    odometer_km: float
    charging_state: str
    online: bool
    climate: ClimateState
    location: Location
    battery_capacity_kwh: float
    consumption_kwh_per_100km: float
    observed_at: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> VehicleStatus:
        try:
            battery_soc = float(value["battery_soc"])
            estimated_range = float(value["estimated_range_km"])
            odometer = float(value["odometer_km"])
            capacity = float(value["battery_capacity_kwh"])
            consumption = float(value["consumption_kwh_per_100km"])
        except (KeyError, TypeError, ValueError) as error:
            raise VehicleValidationError("车辆状态响应字段无效", code="INVALID_RESPONSE") from error
        if not 0 <= battery_soc <= 100 or min(estimated_range, odometer, capacity, consumption) < 0:
            raise VehicleValidationError("车辆状态数值超出范围", code="INVALID_RESPONSE")
        online = value.get("online")
        if not isinstance(online, bool):
            raise VehicleValidationError("车辆在线状态无效", code="INVALID_RESPONSE")
        return cls(
            vehicle_id=normalize_vehicle_id(value.get("vehicle_id", "")),
            model=str(value.get("model", "")),
            battery_soc=battery_soc,
            estimated_range_km=estimated_range,
            odometer_km=odometer,
            charging_state=str(value.get("charging_state", "unknown")),
            online=online,
            climate=ClimateState.from_dict(value.get("climate") or {}),
            location=Location.from_dict(value.get("location") or {}),
            battery_capacity_kwh=capacity,
            consumption_kwh_per_100km=consumption,
            observed_at=str(value.get("observed_at", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteEstimate:
    route_id: str
    origin: str
    destination: str
    distance_km: float
    duration_minutes: int
    estimated_energy_kwh: float
    corridor: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RouteEstimate:
        try:
            return cls(
                route_id=str(value["route_id"]),
                origin=normalize_location_name(value["origin"], field="origin"),
                destination=normalize_location_name(
                    value["destination"], field="destination", max_length=120
                ),
                distance_km=float(value["distance_km"]),
                duration_minutes=int(value["duration_minutes"]),
                estimated_energy_kwh=float(value["estimated_energy_kwh"]),
                corridor=tuple(str(item) for item in value.get("corridor") or []),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise VehicleValidationError("路线响应字段无效", code="INVALID_RESPONSE") from error

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["corridor"] = list(self.corridor)
        return value


@dataclass(frozen=True)
class ChargingStation:
    station_id: str
    name: str
    location: Location
    available_piles: int
    power_kw: float
    connector_types: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ChargingStation:
        try:
            return cls(
                station_id=str(value["station_id"]),
                name=normalize_location_name(value["name"], field="station_name"),
                location=Location.from_dict(value["location"]),
                available_piles=int(value["available_piles"]),
                power_kw=float(value["power_kw"]),
                connector_types=tuple(str(item) for item in value.get("connector_types") or []),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise VehicleValidationError("充电站响应字段无效", code="INVALID_RESPONSE") from error

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["connector_types"] = list(self.connector_types)
        return value
