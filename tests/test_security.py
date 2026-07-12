"""安全策略单测:工作区边界、glob 穿越和 Shell 权限模式。"""

from __future__ import annotations

from pathlib import Path

import pytest

from minicoder.security import PermissionMode, WorkspacePolicy, WorkspaceViolation
from minicoder.tools.bash import BashTool, classify_shell_risk
from minicoder.tools.edit import EditFileTool
from minicoder.tools.glob_tool import GlobTool
from minicoder.tools.write import WriteFileTool


def test_relative_path_resolves_inside_workspace(tmp_path: Path):
    workspace = WorkspacePolicy(tmp_path)
    assert workspace.resolve_path("src/app.py") == (tmp_path / "src" / "app.py").resolve()


def test_parent_traversal_blocked(tmp_path: Path):
    workspace = WorkspacePolicy(tmp_path)
    with pytest.raises(WorkspaceViolation, match="超出工作区"):
        workspace.resolve_path("../outside.txt", operation="写入文件")


def test_absolute_outside_path_blocked(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()
    workspace = WorkspacePolicy(root)
    with pytest.raises(WorkspaceViolation, match="超出工作区"):
        workspace.resolve_path(str(tmp_path / "outside.txt"))


def test_write_tool_cannot_escape_workspace(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    out = WriteFileTool(WorkspacePolicy(root)).run({"path": str(outside), "content": "blocked"})
    assert "拒绝写入文件" in out
    assert not outside.exists()


def test_symlink_escape_blocked_when_supported(tmp_path: Path):
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("当前系统不允许创建符号链接")

    with pytest.raises(WorkspaceViolation, match="超出工作区"):
        WorkspacePolicy(root).resolve_path("link/secret.txt")


def test_glob_parent_traversal_blocked(tmp_path: Path):
    out = GlobTool(WorkspacePolicy(tmp_path)).run({"pattern": "../*.py"})
    assert "glob 模式不得包含" in out


def test_deny_mode_denies_shell(tmp_path: Path):
    workspace = WorkspacePolicy(tmp_path, mode=PermissionMode.DENY)
    out = BashTool(workspace).run({"command": "echo blocked"})
    assert "deny 模式禁止" in out


def test_ask_mode_requires_confirmer(tmp_path: Path):
    workspace = WorkspacePolicy(tmp_path, mode=PermissionMode.ASK)
    out = BashTool(workspace).run({"command": "echo blocked"})
    assert "无法交互确认" in out


def test_ask_mode_uses_confirmation(tmp_path: Path):
    seen = []

    def confirm(request):
        seen.append(request)
        return True

    workspace = WorkspacePolicy(
        tmp_path,
        mode=PermissionMode.ASK,
        confirm=confirm,
    )
    out = BashTool(workspace).run({"command": "echo allowed"})
    assert "allowed" in out
    assert seen[0].command == "echo allowed"
    assert seen[0].workspace == tmp_path.resolve()
    assert seen[0].risk.value == "medium"


def test_ask_mode_honors_rejection(tmp_path: Path):
    workspace = WorkspacePolicy(
        tmp_path,
        mode=PermissionMode.ASK,
        confirm=lambda _request: False,
    )
    out = BashTool(workspace).run({"command": "echo blocked"})
    assert "用户拒绝" in out


def test_allow_mode_still_blocks_dangerous_command(tmp_path: Path):
    workspace = WorkspacePolicy(tmp_path, mode=PermissionMode.ALLOW)
    out = BashTool(workspace).run({"command": "RM -RF /"})
    assert "检测到危险操作" in out


def test_unknown_permission_mode_rejected():
    with pytest.raises(ValueError, match="未知权限模式"):
        PermissionMode.parse("unknown")


def test_legacy_permission_names_migrate():
    assert PermissionMode.parse("trusted") is PermissionMode.ALLOW
    assert PermissionMode.parse("interactive") is PermissionMode.ASK
    assert PermissionMode.parse("strict") is PermissionMode.DENY


def test_file_writes_use_same_allow_ask_deny_policy(tmp_path: Path):
    target = tmp_path / "file.txt"
    denied = WriteFileTool(WorkspacePolicy(tmp_path, mode=PermissionMode.DENY))
    assert "deny 模式禁止" in denied.run({"path": "file.txt", "content": "no"})
    assert not target.exists()

    seen = []
    asking = WriteFileTool(
        WorkspacePolicy(
            tmp_path, mode=PermissionMode.ASK, confirm=lambda request: seen.append(request) or True
        )
    )
    assert "已创建" in asking.run({"path": "file.txt", "content": "yes"})
    assert seen[0].kind == "file_write"


def test_read_only_blocks_write_edit_and_shell(tmp_path: Path):
    target = tmp_path / "file.txt"
    target.write_text("old", encoding="utf-8")
    workspace = WorkspacePolicy(tmp_path, mode=PermissionMode.ALLOW, read_only=True)

    assert "只读模式" in WriteFileTool(workspace).run({"path": "new.txt", "content": "x"})
    assert "只读模式" in EditFileTool(workspace).run(
        {"path": "file.txt", "old_string": "old", "new_string": "new"}
    )
    assert "只读模式" in BashTool(workspace).run({"command": "echo no"})
    assert target.read_text(encoding="utf-8") == "old"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("git status", "low"),
        ("echo hello", "medium"),
        ("pip install package", "high"),
        ("rm -rf /", "critical"),
    ],
)
def test_shell_risk_classification(command: str, expected: str):
    assert classify_shell_risk(command).value == expected
