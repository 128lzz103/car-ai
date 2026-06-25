"""车辆查询与控制操作的独立权限策略。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..security import PermissionMode, RiskLevel
from .models import normalize_vehicle_id


@dataclass(frozen=True)
class VehiclePermissionRequest:
    vehicle_id: str
    action: str
    description: str
    risk: RiskLevel


@dataclass(frozen=True)
class VehiclePermissionDecision:
    allowed: bool
    reason: str


ConfirmVehicleAction = Callable[[VehiclePermissionRequest], bool]


@dataclass
class VehicleActionPolicy:
    mode: PermissionMode = PermissionMode.ASK
    confirm: ConfirmVehicleAction | None = None
    read_only: bool = False
    allowed_vehicle_ids: frozenset[str] | None = None

    def __post_init__(self) -> None:
        self.mode = PermissionMode.parse(self.mode)
        if self.allowed_vehicle_ids is not None:
            self.allowed_vehicle_ids = frozenset(
                normalize_vehicle_id(item) for item in self.allowed_vehicle_ids
            )

    def authorize_query(self, vehicle_id: str | None = None) -> VehiclePermissionDecision:
        if vehicle_id and not self._vehicle_allowed(vehicle_id):
            return VehiclePermissionDecision(False, "车辆 ID 不在授权范围内")
        return VehiclePermissionDecision(True, "只读车辆查询已授权")

    def authorize_control(
        self,
        vehicle_id: str,
        *,
        action: str,
        description: str,
    ) -> VehiclePermissionDecision:
        normalized = normalize_vehicle_id(vehicle_id)
        if not self._vehicle_allowed(normalized):
            return VehiclePermissionDecision(False, "车辆 ID 不在授权范围内")
        if self.read_only:
            return VehiclePermissionDecision(False, "只读模式禁止车辆控制")
        if self.mode is PermissionMode.ALLOW:
            return VehiclePermissionDecision(True, "allow 模式已授权车辆控制")
        if self.mode is PermissionMode.DENY:
            return VehiclePermissionDecision(False, "deny 模式禁止车辆控制")
        if self.confirm is None:
            return VehiclePermissionDecision(False, "ask 模式当前无法交互确认")
        request = VehiclePermissionRequest(
            vehicle_id=normalized,
            action=action,
            description=description,
            risk=RiskLevel.HIGH,
        )
        if self.confirm(request):
            return VehiclePermissionDecision(True, "用户已授权本次车辆控制")
        return VehiclePermissionDecision(False, "用户拒绝车辆控制")

    def _vehicle_allowed(self, vehicle_id: str) -> bool:
        normalized = normalize_vehicle_id(vehicle_id)
        return self.allowed_vehicle_ids is None or normalized in self.allowed_vehicle_ids
