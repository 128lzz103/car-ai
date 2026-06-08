"""write_file —— 创建或覆盖文件(写操作,串行)。"""

from __future__ import annotations

from typing import Any

from ..security import WorkspaceViolation
from .base import WorkspaceTool


class WriteFileTool(WorkspaceTool):
    name = "write_file"
    description = "创建新文件或覆盖已有文件。修改已有文件优先用 edit_file。"
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径"},
            "content": {"type": "string", "description": "完整文件内容"},
        },
        "required": ["path", "content"],
    }

    def run(self, args: dict[str, Any]) -> str:
        try:
            path = self.workspace.resolve_path(args["path"], operation="写入文件")
        except WorkspaceViolation as e:
            return f"错误:{e}"
        decision = self.workspace.authorize_write(path, operation="写入文件")
        if not decision.allowed:
            return f"已拒绝写入文件:{decision.reason}。"
        content = args["content"]
        existed = path.is_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        action = "覆盖" if existed else "创建"
        return f"已{action}文件 {path}({len(content)} 字符)"
