"""原子会话持久化、Schema 迁移与备份恢复。"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from .context import ContextProtocolError, validate_tool_protocol
from .providers import ToolCall

SESSION_DIR = Path.home() / ".minicoder" / "sessions"
SESSION_SCHEMA_VERSION = 2
MAX_SESSION_BYTES = 50 * 1024 * 1024
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


class SessionError(Exception):
    """会话持久化错误基类。"""


class SessionWriteError(SessionError):
    pass


class SessionCorruptedError(SessionError):
    pass


class SessionValidationError(SessionError, ValueError):
    pass


class UnsupportedSessionVersionError(SessionValidationError):
    pass


def _tool_call_value(tool_call: Any, key: str, default: Any = None) -> Any:
    if isinstance(tool_call, dict):
        return tool_call.get(key, default)
    return getattr(tool_call, key, default)


def _encode_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    encoded: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise SessionValidationError(f"transcript[{index}] 必须是对象")
        item = dict(message)
        if "tool_calls" in message:
            raw_calls = message["tool_calls"]
            if raw_calls is None:
                raw_calls = []
            if not isinstance(raw_calls, list):
                raise SessionValidationError(f"transcript[{index}].tool_calls 必须是列表")
            calls = []
            for call_index, tool_call in enumerate(raw_calls):
                tool_id = _tool_call_value(tool_call, "id")
                name = _tool_call_value(tool_call, "name")
                arguments = _tool_call_value(tool_call, "arguments", {})
                if not isinstance(tool_id, str) or not tool_id:
                    raise SessionValidationError(
                        f"transcript[{index}].tool_calls[{call_index}].id 无效"
                    )
                if not isinstance(name, str) or not name:
                    raise SessionValidationError(
                        f"transcript[{index}].tool_calls[{call_index}].name 无效"
                    )
                if not isinstance(arguments, dict):
                    raise SessionValidationError(
                        f"transcript[{index}].tool_calls[{call_index}].arguments 必须是对象"
                    )
                calls.append({"id": tool_id, "name": name, "arguments": arguments})
            item["tool_calls"] = calls
        encoded.append(item)
    return encoded


def _decode_messages(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decoded: list[dict[str, Any]] = []
    for index, message in enumerate(raw):
        if not isinstance(message, dict):
            raise SessionValidationError(f"transcript[{index}] 必须是对象")
        item = dict(message)
        if "tool_calls" in message:
            raw_calls = message["tool_calls"]
            if raw_calls is None:
                raw_calls = []
            if not isinstance(raw_calls, list):
                raise SessionValidationError(f"transcript[{index}].tool_calls 必须是列表")
            calls = []
            for call_index, tool_call in enumerate(raw_calls):
                if not isinstance(tool_call, dict):
                    raise SessionValidationError(
                        f"transcript[{index}].tool_calls[{call_index}] 必须是对象"
                    )
                tool_id = tool_call.get("id")
                name = tool_call.get("name")
                arguments = tool_call.get("arguments", {})
                if not isinstance(tool_id, str) or not tool_id.strip():
                    raise SessionValidationError(
                        f"transcript[{index}].tool_calls[{call_index}].id 无效"
                    )
                if not isinstance(name, str) or not name.strip():
                    raise SessionValidationError(
                        f"transcript[{index}].tool_calls[{call_index}].name 无效"
                    )
                if not isinstance(arguments, dict):
                    raise SessionValidationError(
                        f"transcript[{index}].tool_calls[{call_index}].arguments 必须是对象"
                    )
                calls.append(ToolCall(id=tool_id, name=name, arguments=arguments))
            item["tool_calls"] = calls
        decoded.append(item)
    return decoded


def _validate_transcript(messages: list[dict[str, Any]]) -> None:
    if not isinstance(messages, list):
        raise SessionValidationError("transcript 必须是列表")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise SessionValidationError(f"transcript[{index}] 必须是对象")
        role = message.get("role")
        if role not in {"user", "assistant", "tool"}:
            raise SessionValidationError(f"transcript[{index}].role 无效: {role!r}")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise SessionValidationError(f"transcript[{index}].content 必须是字符串")
        if role != "assistant" and message.get("tool_calls"):
            raise SessionValidationError(f"transcript[{index}] 非 assistant 消息不能含 tool_calls")
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise SessionValidationError(f"transcript[{index}].tool_call_id 必须是非空字符串")
    try:
        validate_tool_protocol(messages)
    except ContextProtocolError as error:
        raise SessionValidationError(f"Transcript 工具协议无效: {error}") from error


class SessionStore:
    def __init__(self, root: Path, *, max_bytes: int = MAX_SESSION_BYTES) -> None:
        self.root = root.expanduser().resolve()
        self.max_bytes = max_bytes

    def safe_path(self, name: str) -> Path:
        name = name.strip()
        if not name or not _SAFE_NAME.fullmatch(name) or ".." in name:
            raise SessionValidationError(
                f"非法会话名: {name!r}(只允许字母、数字、. _ -,不含 .. 或路径分隔符)"
            )
        path = (self.root / f"{name}.json").resolve()
        if self.root not in path.parents:
            raise SessionValidationError(f"检测到路径穿越: {name!r}")
        return path

    @staticmethod
    def backup_path(path: Path) -> Path:
        return path.with_name(path.name + ".bak")

    def _serialize(
        self,
        transcript: list[dict[str, Any]],
        meta: dict[str, Any] | None,
    ) -> bytes:
        if meta is not None and not isinstance(meta, dict):
            raise SessionValidationError("meta 必须是对象")
        encoded = _encode_messages(transcript)
        decoded = _decode_messages(encoded)
        _validate_transcript(decoded)
        payload = {
            "schema_version": SESSION_SCHEMA_VERSION,
            "saved_at": time.time(),
            "meta": meta or {},
            "transcript": encoded,
        }
        try:
            data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        except (TypeError, ValueError) as error:
            raise SessionValidationError(f"会话包含不可序列化数据: {error}") from error
        if len(data) > self.max_bytes:
            raise SessionValidationError(f"会话过大: {len(data)} 字节 > {self.max_bytes} 字节")
        return data

    def _fsync_directory(self) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            descriptor = os.open(self.root, flags)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    def _atomic_write(self, path: Path, data: bytes) -> None:
        descriptor = -1
        temp_path: Path | None = None
        try:
            descriptor, raw_temp_path = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
            )
            temp_path = Path(raw_temp_path)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            temp_path = None
            self._fsync_directory()
        except OSError as error:
            raise SessionWriteError(f"原子写入失败 {path}: {error}") from error
        finally:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _read_payload(self, path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], float, int]:
        try:
            size = path.stat().st_size
            if size > self.max_bytes:
                raise SessionCorruptedError(f"会话文件过大: {size} 字节 > {self.max_bytes} 字节")
            raw = json.loads(path.read_text(encoding="utf-8"))
        except SessionError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SessionCorruptedError(f"无法读取会话 {path}: {error}") from error

        if not isinstance(raw, dict):
            raise SessionValidationError("会话根节点必须是对象")
        version = raw.get("schema_version", raw.get("version", 1))
        if not isinstance(version, int):
            raise SessionValidationError("schema_version 必须是整数")
        if version > SESSION_SCHEMA_VERSION or version < 1:
            raise UnsupportedSessionVersionError(
                f"不支持的会话版本 {version};当前支持 1..{SESSION_SCHEMA_VERSION}"
            )

        if version == 1:
            transcript = raw.get("messages")
        else:
            transcript = raw.get("transcript")
        if not isinstance(transcript, list):
            raise SessionValidationError("会话缺少有效 transcript")
        meta = raw.get("meta", {})
        if not isinstance(meta, dict):
            raise SessionValidationError("meta 必须是对象")
        saved_at = raw.get("saved_at", 0)
        if isinstance(saved_at, bool) or not isinstance(saved_at, (int, float)):
            raise SessionValidationError("saved_at 必须是数字")

        messages = _decode_messages(transcript)
        _validate_transcript(messages)
        loaded_meta = dict(meta)
        if version == 1:
            loaded_meta["_session_migrated_from"] = 1
        return messages, loaded_meta, float(saved_at), version

    def save(
        self,
        name: str,
        transcript: list[dict[str, Any]],
        meta: dict[str, Any] | None = None,
    ) -> Path:
        path = self.safe_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._serialize(transcript, meta)
        backup = self.backup_path(path)

        try:
            if path.is_file():
                try:
                    self._read_payload(path)
                except UnsupportedSessionVersionError as error:
                    raise SessionWriteError(f"拒绝覆盖更高版本的会话 {path}: {error}") from error
                except SessionError:
                    pass
                else:
                    self._atomic_write(backup, path.read_bytes())
            self._atomic_write(path, data)
        except SessionWriteError:
            raise
        except OSError as error:
            raise SessionWriteError(f"原子保存会话失败 {path}: {error}") from error
        return path

    def load(
        self,
        name: str,
        *,
        recover: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        path = self.safe_path(name)
        backup = self.backup_path(path)
        if not path.is_file():
            if not recover or not backup.is_file():
                raise FileNotFoundError(f"会话不存在: {name}")
            primary_error: BaseException = FileNotFoundError(f"主会话不存在: {path}")
        else:
            try:
                messages, meta, _saved_at, _version = self._read_payload(path)
                return messages, meta
            except UnsupportedSessionVersionError:
                raise
            except SessionError as error:
                if not recover:
                    raise
                primary_error = error

        if not backup.is_file():
            raise primary_error
        try:
            messages, meta, _saved_at, _version = self._read_payload(backup)
        except SessionError as backup_error:
            raise SessionCorruptedError(
                f"主会话和备份均不可用: primary={primary_error}; backup={backup_error}"
            ) from backup_error
        recovered_meta = dict(meta)
        recovered_meta["_session_recovered_from_backup"] = True
        recovered_meta["_session_primary_error"] = str(primary_error)
        return messages, recovered_meta

    def list(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        names = {path.stem for path in self.root.glob("*.json")}
        backup_suffix = ".json.bak"
        names.update(
            path.name[: -len(backup_suffix)] for path in self.root.glob(f"*{backup_suffix}")
        )

        entries: list[dict[str, Any]] = []
        for name in names:
            try:
                path = self.safe_path(name)
            except SessionValidationError:
                continue
            backup = self.backup_path(path)
            status = "ok"
            error_text = ""
            messages: list[dict[str, Any]] = []
            saved_at = self._safe_mtime(path, backup)

            if path.is_file():
                try:
                    messages, _meta, saved_at, _version = self._read_payload(path)
                except UnsupportedSessionVersionError as error:
                    status = "unsupported"
                    error_text = str(error)
                except SessionError as error:
                    status = "corrupted"
                    error_text = str(error)
                    if backup.is_file():
                        try:
                            messages, _meta, saved_at, _version = self._read_payload(backup)
                        except UnsupportedSessionVersionError as backup_error:
                            status = "unsupported"
                            error_text = f"{error}; backup={backup_error}"
                        except SessionError:
                            pass
                        else:
                            status = "recoverable"
            elif backup.is_file():
                status = "recoverable"
                error_text = f"主会话不存在: {path}"
                try:
                    messages, _meta, saved_at, _version = self._read_payload(backup)
                except UnsupportedSessionVersionError as error:
                    status = "unsupported"
                    error_text = str(error)
                except SessionError as error:
                    status = "corrupted"
                    error_text = str(error)
            entries.append(
                {
                    "name": name,
                    "saved_at": saved_at,
                    "messages": len(messages),
                    "status": status,
                    "error": error_text,
                }
            )
        entries.sort(key=lambda item: item["saved_at"], reverse=True)
        return entries

    @staticmethod
    def _safe_mtime(*paths: Path) -> float:
        for path in paths:
            try:
                return path.stat().st_mtime
            except OSError:
                continue
        return 0.0


def _store() -> SessionStore:
    return SessionStore(SESSION_DIR)


def _safe_path(name: str) -> Path:
    return _store().safe_path(name)


def save_session(
    name: str,
    messages: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> Path:
    return _store().save(name, messages, meta)


def load_session(
    name: str,
    *,
    recover: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _store().load(name, recover=recover)


def list_sessions() -> list[dict[str, Any]]:
    return _store().list()


def _markdown_export(
    messages: list[dict[str, Any]],
    meta: dict[str, Any],
) -> str:
    lines = ["# minicoder Session", ""]
    if meta:
        lines.extend(["## Metadata", ""])
        for key, value in sorted(meta.items()):
            rendered = (
                json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
            )
            lines.append(f"- **{key}**: {rendered}")
        lines.append("")

    role_titles = {"user": "User", "assistant": "Assistant", "tool": "Tool Result"}
    for message in messages:
        role = str(message.get("role") or "unknown")
        title = role_titles.get(role, role.title())
        if role == "tool":
            name = message.get("name") or "tool"
            tool_call_id = message.get("tool_call_id") or "unknown"
            title += f": {name} ({tool_call_id})"
        lines.extend([f"## {title}", "", str(message.get("content") or ""), ""])
        for tool_call in message.get("tool_calls") or []:
            name = _tool_call_value(tool_call, "name", "tool")
            tool_id = _tool_call_value(tool_call, "id", "unknown")
            arguments = _tool_call_value(tool_call, "arguments", {})
            lines.extend(
                [
                    f"### Tool Call: {name} ({tool_id})",
                    "",
                    "```json",
                    json.dumps(arguments, ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def export_session(
    destination: str | Path,
    messages: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
    *,
    format: str | None = None,
) -> Path:
    """将完整 Transcript 原子导出为可恢复 JSON 或可读 Markdown。"""
    path = Path(destination).expanduser().resolve()
    export_format = (format or path.suffix.lstrip(".") or "json").strip().lower()
    aliases = {"md": "markdown", "json": "json", "markdown": "markdown"}
    if export_format not in aliases:
        raise SessionValidationError(f"不支持的导出格式: {export_format!r}")
    export_format = aliases[export_format]

    store = SessionStore(path.parent)
    if export_format == "json":
        data = store._serialize(messages, meta)
    else:
        encoded = _encode_messages(messages)
        decoded = _decode_messages(encoded)
        _validate_transcript(decoded)
        try:
            data = _markdown_export(encoded, meta or {}).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise SessionValidationError(f"会话包含不可导出的数据: {error}") from error
        if len(data) > store.max_bytes:
            raise SessionValidationError(f"导出内容过大: {len(data)} 字节 > {store.max_bytes} 字节")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        store._atomic_write(path, data)
    except SessionError:
        raise
    except OSError as error:
        raise SessionWriteError(f"导出会话失败 {path}: {error}") from error
    return path
