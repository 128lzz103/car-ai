"""工具注册表 —— 装配可用工具集。

对标 Claude Code 的 tools.ts(docs/04):集中组装工具池。极简版只保留 7 个核心工具。
agent 工具需要 provider/config 才能派生子 agent,因此在装配时注入。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..security import PermissionMode, WorkspacePolicy
from .agent import AgentTool
from .base import Tool
from .bash import BashTool
from .edit import EditFileTool
from .glob_tool import GlobTool
from .grep import GrepTool
from .read import ReadFileTool
from .vehicle import RegisteredActionTool, build_action_tools, build_vehicle_tools
from .write import WriteFileTool

__all__ = [
    "Tool",
    "BashTool",
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "GlobTool",
    "GrepTool",
    "AgentTool",
    "RegisteredActionTool",
    "build_action_tools",
    "build_tools",
    "build_vehicle_tools",
]


def build_tools(
    provider: Any,
    config: Any,
    include_agent: bool = True,
    workspace: WorkspacePolicy | None = None,
    vehicle_client: Any | None = None,
    vehicle_policy: Any | None = None,
    action_registry: Any | None = None,
) -> list[Tool]:
    """装配工具列表。

    include_agent=False 用于子 agent —— 移除 agent 工具以禁止递归 spawn。
    主/子 agent 必须共享 workspace,防止子 agent 获得更宽松的权限。
    """
    workspace = workspace or WorkspacePolicy(
        Path.cwd(),
        mode=PermissionMode.parse(getattr(config, "permission_mode", "ask")),
        read_only=bool(getattr(config, "read_only", False)),
    )
    tools: list[Tool] = [
        BashTool(workspace),
        ReadFileTool(workspace),
        WriteFileTool(workspace),
        EditFileTool(workspace),
        GlobTool(workspace),
        GrepTool(workspace),
    ]
    if include_agent:
        tools.append(AgentTool(provider=provider, config=config, workspace=workspace))
    if action_registry is not None:
        tools.extend(build_action_tools(action_registry))
    elif vehicle_client is not None and vehicle_policy is not None:
        tools.extend(build_vehicle_tools(vehicle_client, vehicle_policy))
    return tools
