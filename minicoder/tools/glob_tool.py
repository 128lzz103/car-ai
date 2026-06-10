"""glob —— 按文件名模式查找文件(只读,并发安全)。"""

from __future__ import annotations

from typing import Any

from ..security import WorkspaceViolation
from .base import WorkspaceTool

MAX_RESULTS = 200
_IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}


class GlobTool(WorkspaceTool):
    name = "glob"
    description = "按 glob 模式查找文件,例如 '**/*.py' 或 'src/**/*.ts'。返回匹配的路径列表。"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "glob 模式,如 **/*.py"},
            "path": {"type": "string", "description": "搜索根目录(默认当前目录)"},
        },
        "required": ["pattern"],
    }

    def is_read_only(self) -> bool:
        return True

    def run(self, args: dict[str, Any]) -> str:
        try:
            root = self.workspace.resolve_path(args.get("path") or ".", operation="搜索目录")
            pattern = self.workspace.validate_glob(args["pattern"])
        except WorkspaceViolation as e:
            return f"错误:{e}"
        if not root.is_dir():
            return f"错误:目录不存在: {root}"

        matches: list[str] = []
        for p in root.glob(pattern):
            if any(part in _IGNORE_DIRS for part in p.parts):
                continue
            try:
                safe_path = self.workspace.resolve_path(str(p), operation="读取搜索结果")
            except WorkspaceViolation:
                continue
            if safe_path.is_file():
                matches.append(str(safe_path))
                if len(matches) >= MAX_RESULTS:
                    break
        if not matches:
            return "(无匹配文件)"
        matches.sort()
        suffix = f"\n... (已截断到 {MAX_RESULTS} 条)" if len(matches) >= MAX_RESULTS else ""
        return "\n".join(matches) + suffix
