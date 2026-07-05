"""工作区边界与命令权限策略。

文件工具只能访问工作区内的路径。Shell 无法通过路径检查实现真正沙箱，
因此使用 allow / ask / deny 三种显式权限模式和只读开关进行门控。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class PermissionMode(str, Enum):
    """写操作权限模式。旧名称在 parse() 中兼容迁移。"""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"
    TRUSTED = "allow"
    INTERACTIVE = "ask"
    STRICT = "deny"

    @classmethod
    def parse(cls, value: str | PermissionMode) -> PermissionMode:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        legacy = {"trusted": "allow", "interactive": "ask", "strict": "deny"}
        try:
            return cls(legacy.get(normalized, normalized))
        except ValueError as e:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(f"未知权限模式: {value!r},支持 {choices}") from e


class WorkspaceViolation(ValueError):
    """路径试图越过工作区安全边界。"""


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class PermissionRequest:
    """交给 CLI 或其他前端确认的一次敏感操作。"""

    kind: str
    description: str
    command: str
    workspace: Path
    risk: RiskLevel = RiskLevel.MEDIUM


@dataclass(frozen=True)
class PermissionDecision:
    action: PermissionMode
    reason: str

    @property
    def allowed(self) -> bool:
        return self.action is PermissionMode.ALLOW


ConfirmPermission = Callable[[PermissionRequest], bool]


@dataclass
class WorkspacePolicy:
    """统一管理工作区路径和 Shell 执行权限。"""

    root: Path
    mode: PermissionMode = PermissionMode.ASK
    confirm: ConfirmPermission | None = None
    read_only: bool = False

    def __post_init__(self) -> None:
        self.root = self.root.expanduser().resolve()
        self.mode = PermissionMode.parse(self.mode)
        if not self.root.is_dir():
            raise ValueError(f"工作区目录不存在: {self.root}")

    def resolve_path(self, raw_path: str, *, operation: str = "访问") -> Path:
        """解析路径并确保最终位置仍位于工作区内。"""
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise WorkspaceViolation(f"拒绝{operation}:路径不能为空")

        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve(strict=False)
        except OSError as e:
            raise WorkspaceViolation(f"拒绝{operation}:无法解析路径 {raw_path!r}: {e}") from e

        if resolved != self.root and self.root not in resolved.parents:
            raise WorkspaceViolation(f"拒绝{operation}:路径超出工作区 {self.root}: {resolved}")
        return resolved

    def validate_glob(self, pattern: str) -> str:
        """拒绝可直接跨越搜索根目录的 glob 模式。"""
        if not isinstance(pattern, str) or not pattern.strip():
            raise WorkspaceViolation("拒绝搜索:glob 模式不能为空")
        normalized = pattern.replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        if Path(pattern).is_absolute() or ".." in parts:
            raise WorkspaceViolation(f"拒绝搜索:glob 模式不得包含绝对路径或 '..': {pattern}")
        return pattern

    def authorize(
        self,
        *,
        kind: str,
        description: str,
        command: str,
        risk: RiskLevel,
    ) -> PermissionDecision:
        """统一处理文件写入和 Shell 的 allow / ask / deny 决策。"""
        if self.read_only:
            return PermissionDecision(PermissionMode.DENY, "只读模式禁止写操作和 Shell")
        if self.mode is PermissionMode.ALLOW:
            return PermissionDecision(PermissionMode.ALLOW, "allow 模式已授权")
        if self.mode is PermissionMode.DENY:
            return PermissionDecision(PermissionMode.DENY, "deny 模式禁止执行写操作")
        if self.confirm is None:
            return PermissionDecision(PermissionMode.DENY, "ask 模式当前无法交互确认")

        request = PermissionRequest(
            kind=kind,
            description=description,
            command=command,
            workspace=self.root,
            risk=risk,
        )
        if self.confirm(request):
            return PermissionDecision(PermissionMode.ALLOW, "用户已授权本次执行")
        return PermissionDecision(PermissionMode.DENY, "用户拒绝执行")

    def authorize_write(self, path: Path, *, operation: str) -> PermissionDecision:
        return self.authorize(
            kind="file_write",
            description=f"{operation}将修改工作区文件",
            command=str(path),
            risk=RiskLevel.MEDIUM,
        )

    def authorize_shell(self, command: str, risk: RiskLevel) -> PermissionDecision:
        return self.authorize(
            kind="shell",
            description="Shell 命令可能读取、修改或删除工作区及系统文件",
            command=command,
            risk=risk,
        )
