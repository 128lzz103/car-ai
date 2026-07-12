"""原子会话持久化、版本迁移、协议验证与备份恢复测试。"""

from __future__ import annotations

import json

import pytest

from minicoder import session as session_mod
from minicoder.providers import ToolCall


@pytest.fixture(autouse=True)
def _tmp_session_dir(tmp_path, monkeypatch):
    # 把会话目录重定向到临时目录,避免污染 ~/.minicoder
    monkeypatch.setattr(session_mod, "SESSION_DIR", tmp_path / "sessions")


def test_save_load_roundtrip():
    messages = [
        {"role": "user", "content": "做点事"},
        {
            "role": "assistant",
            "content": "好的",
            "tool_calls": [ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "文件内容"},
    ]
    session_mod.save_session("test1", messages, {"model": "gpt-4o"})
    loaded, meta = session_mod.load_session("test1")

    assert meta["model"] == "gpt-4o"
    assert loaded[0]["content"] == "做点事"
    tc = loaded[1]["tool_calls"][0]
    assert isinstance(tc, ToolCall)
    assert tc.id == "c1" and tc.name == "read_file" and tc.arguments == {"path": "a.py"}
    assert loaded[2]["tool_call_id"] == "c1"

    raw = json.loads((session_mod.SESSION_DIR / "test1.json").read_text(encoding="utf-8"))
    assert raw["schema_version"] == session_mod.SESSION_SCHEMA_VERSION
    assert raw["transcript"][0]["content"] == "做点事"


def test_v1_session_is_migrated_in_memory():
    path = session_mod.SESSION_DIR / "legacy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "saved_at": 123,
                "meta": {"model": "legacy-model"},
                "messages": [{"role": "user", "content": "旧会话"}],
            }
        ),
        encoding="utf-8",
    )

    messages, meta = session_mod.load_session("legacy")

    assert messages == [{"role": "user", "content": "旧会话"}]
    assert meta["model"] == "legacy-model"
    assert meta["_session_migrated_from"] == 1


def test_list_sessions():
    session_mod.save_session("s1", [{"role": "user", "content": "a"}])
    session_mod.save_session("s2", [{"role": "user", "content": "b"}])
    names = {s["name"] for s in session_mod.list_sessions()}
    assert names == {"s1", "s2"}


def test_path_traversal_blocked():
    with pytest.raises(ValueError):
        session_mod.save_session("../../etc/passwd", [])
    with pytest.raises(ValueError):
        session_mod.save_session("a/b", [])
    with pytest.raises(ValueError):
        session_mod.load_session("..")


def test_load_missing():
    with pytest.raises(FileNotFoundError):
        session_mod.load_session("does-not-exist")


def test_replace_failure_preserves_primary_and_cleans_temp_files(monkeypatch):
    session_mod.save_session("atomic", [{"role": "user", "content": "old"}])
    path = session_mod.SESSION_DIR / "atomic.json"
    old_bytes = path.read_bytes()
    real_replace = session_mod.os.replace

    def fail_primary_replace(source, destination):
        if destination == path:
            raise OSError("simulated replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(session_mod.os, "replace", fail_primary_replace)
    with pytest.raises(session_mod.SessionWriteError, match="原子写入失败"):
        session_mod.save_session("atomic", [{"role": "user", "content": "new"}])

    assert path.read_bytes() == old_bytes
    assert not list(session_mod.SESSION_DIR.glob("*.tmp"))
    loaded, _meta = session_mod.load_session("atomic")
    assert loaded[0]["content"] == "old"


def test_backup_contains_previous_valid_version():
    session_mod.save_session("history", [{"role": "user", "content": "v1"}])
    session_mod.save_session("history", [{"role": "user", "content": "v2"}])

    backup = session_mod.SESSION_DIR / "history.json.bak"
    raw = json.loads(backup.read_text(encoding="utf-8"))
    assert raw["transcript"][0]["content"] == "v1"


def test_corrupt_primary_recovers_previous_backup():
    session_mod.save_session("recover", [{"role": "user", "content": "v1"}])
    session_mod.save_session("recover", [{"role": "user", "content": "v2"}])
    (session_mod.SESSION_DIR / "recover.json").write_text("{broken", encoding="utf-8")

    messages, meta = session_mod.load_session("recover")

    assert messages[0]["content"] == "v1"
    assert meta["_session_recovered_from_backup"] is True
    assert "无法读取会话" in meta["_session_primary_error"]


def test_both_primary_and_backup_corrupt_fails():
    session_mod.save_session("broken", [{"role": "user", "content": "v1"}])
    session_mod.save_session("broken", [{"role": "user", "content": "v2"}])
    (session_mod.SESSION_DIR / "broken.json").write_text("bad", encoding="utf-8")
    (session_mod.SESSION_DIR / "broken.json.bak").write_text("bad", encoding="utf-8")

    with pytest.raises(session_mod.SessionCorruptedError, match="主会话和备份均不可用"):
        session_mod.load_session("broken")


def test_future_schema_is_rejected_without_falling_back():
    path = session_mod.SESSION_DIR / "future.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"schema_version": 999, "transcript": [], "meta": {}, "saved_at": 1}),
        encoding="utf-8",
    )

    with pytest.raises(session_mod.UnsupportedSessionVersionError, match="不支持的会话版本"):
        session_mod.load_session("future")


def test_invalid_tool_protocol_is_not_saved():
    incomplete = [
        {"role": "user", "content": "read"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [ToolCall(id="c1", name="read_file", arguments={})],
        },
    ]

    with pytest.raises(session_mod.SessionValidationError, match="工具协议无效"):
        session_mod.save_session("invalid", incomplete)
    assert not (session_mod.SESSION_DIR / "invalid.json").exists()


def test_invalid_serialized_tool_call_is_rejected():
    path = session_mod.SESSION_DIR / "invalid-call.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "saved_at": 1,
                "meta": {},
                "transcript": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "", "name": "read_file", "arguments": []}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(session_mod.SessionValidationError, match="id 无效"):
        session_mod.load_session("invalid-call", recover=False)


def test_size_limit_applies_to_save_and_load(tmp_path):
    store = session_mod.SessionStore(tmp_path / "limited", max_bytes=32)
    with pytest.raises(session_mod.SessionValidationError, match="会话过大"):
        store.save("large", [{"role": "user", "content": "content"}])

    path = store.safe_path("external")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * 33)
    with pytest.raises(session_mod.SessionCorruptedError, match="会话文件过大"):
        store.load("external", recover=False)


def test_list_reports_health_statuses():
    session_mod.save_session("ok", [{"role": "user", "content": "ok"}])

    session_mod.save_session("recoverable", [{"role": "user", "content": "old"}])
    session_mod.save_session("recoverable", [{"role": "user", "content": "new"}])
    (session_mod.SESSION_DIR / "recoverable.json").write_text("bad", encoding="utf-8")

    (session_mod.SESSION_DIR / "corrupted.json").write_text("bad", encoding="utf-8")
    (session_mod.SESSION_DIR / "backup-only.json.bak").write_text("bad", encoding="utf-8")
    (session_mod.SESSION_DIR / "unsupported.json").write_text(
        json.dumps({"schema_version": 999, "transcript": [], "saved_at": 1}),
        encoding="utf-8",
    )

    statuses = {entry["name"]: entry["status"] for entry in session_mod.list_sessions()}

    assert statuses == {
        "ok": "ok",
        "recoverable": "recoverable",
        "corrupted": "corrupted",
        "backup-only": "corrupted",
        "unsupported": "unsupported",
    }


def test_export_json_is_atomic_and_load_compatible(tmp_path):
    destination = tmp_path / "export.json"
    messages = [{"role": "user", "content": "portable"}]

    exported = session_mod.export_session(
        destination,
        messages,
        {"model": "test-model"},
        format="json",
    )

    raw = json.loads(exported.read_text(encoding="utf-8"))
    assert raw["schema_version"] == session_mod.SESSION_SCHEMA_VERSION
    assert raw["transcript"] == messages
    store = session_mod.SessionStore(tmp_path)
    loaded, meta = store.load("export")
    assert loaded == messages
    assert meta["model"] == "test-model"
    assert not list(tmp_path.glob("*.tmp"))


def test_export_markdown_renders_messages_and_tool_calls(tmp_path):
    messages = [
        {"role": "user", "content": "read it"},
        {
            "role": "assistant",
            "content": "working",
            "tool_calls": [ToolCall(id="1", name="read_file", arguments={"path": "a.py"})],
        },
        {"role": "tool", "tool_call_id": "1", "name": "read_file", "content": "data"},
    ]

    path = session_mod.export_session(
        tmp_path / "export.md",
        messages,
        {"provider": "openai"},
        format="markdown",
    )
    rendered = path.read_text(encoding="utf-8")

    assert "# minicoder Session" in rendered
    assert "## User" in rendered
    assert "### Tool Call: read_file (1)" in rendered
    assert '"path": "a.py"' in rendered
    assert "## Tool Result: read_file (1)" in rendered


def test_export_rejects_unknown_format(tmp_path):
    with pytest.raises(session_mod.SessionValidationError, match="不支持的导出格式"):
        session_mod.export_session(tmp_path / "session.txt", [], format="text")
