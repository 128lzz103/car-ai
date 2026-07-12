"""交互式 REPL 会话自动保存与恢复提示测试。"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from minicoder import cli as cli_mod
from minicoder.agent import Agent
from minicoder.providers import Reply, ToolCall
from minicoder.security import PermissionRequest
from minicoder.session import SessionWriteError


class OneReplyProvider:
    def chat(self, *args, **kwargs):
        return Reply(text="done")


def _config(*, autosave: bool = True):
    return SimpleNamespace(
        provider="openai",
        model="test-model",
        max_rounds=2,
        context_window=1_000,
        output_reserve=100,
        permission_mode="strict",
        read_only=False,
        autosave=autosave,
        autosave_name="test-autosave",
    )


def _repl(config=None):
    config = config or _config()
    agent = Agent(OneReplyProvider(), config, [], "system", stream=False)
    return cli_mod.Repl(agent, config)


def test_repl_autosaves_full_transcript_with_metadata(monkeypatch):
    saves = []

    def fake_save(name, transcript, meta):
        saves.append((name, transcript, meta))
        return Path("saved.json")

    monkeypatch.setattr(cli_mod, "save_session", fake_save)
    repl = _repl()

    assert repl.agent.chat("hello") == "done"

    assert [item[2]["checkpoint_reason"] for item in saves] == [
        "user",
        "assistant_final",
    ]
    name, transcript, meta = saves[-1]
    assert name == "test-autosave"
    assert [message["role"] for message in transcript] == ["user", "assistant"]
    assert meta["model"] == "test-model"
    assert meta["provider"] == "openai"
    assert meta["workspace"] == str(Path.cwd().resolve())


def test_repl_can_disable_autosave():
    repl = _repl(_config(autosave=False))
    assert repl.agent.on_checkpoint is None


def test_manual_save_includes_runtime_metadata(monkeypatch):
    saves = []

    def fake_save(name, transcript, meta):
        saves.append((name, transcript, meta))
        return Path("manual.json")

    monkeypatch.setattr(cli_mod, "save_session", fake_save)
    repl = _repl()

    repl._handle_command("/save named")

    name, _transcript, meta = saves[-1]
    assert name == "named"
    assert meta["checkpoint_reason"] == "manual"
    assert {"model", "provider", "workspace"} <= meta.keys()


def test_session_list_shows_health_and_load_warns_on_backup(monkeypatch, capsys):
    repl = _repl()
    monkeypatch.setattr(
        cli_mod,
        "list_sessions",
        lambda: [
            {
                "name": "damaged",
                "messages": 2,
                "saved_at": 1,
                "status": "recoverable",
                "error": "bad primary",
            }
        ],
    )
    monkeypatch.setattr(
        cli_mod,
        "load_session",
        lambda _name: (
            [{"role": "user", "content": "restored"}],
            {"_session_recovered_from_backup": True},
        ),
    )

    repl._list_sessions()
    repl._load_session("damaged")
    output = capsys.readouterr().out

    assert "[recoverable]" in output
    assert "已从上一份有效备份恢复" in output
    assert repl.agent.transcript[0]["content"] == "restored"


def test_repl_exports_current_session_inside_workspace(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    repl = _repl(_config(autosave=False))
    repl.agent.messages = [{"role": "user", "content": "export me"}]

    repl._handle_command('/export markdown "exports/session.md"')

    exported = tmp_path / "exports" / "session.md"
    assert exported.is_file()
    assert "export me" in exported.read_text(encoding="utf-8")
    assert "已导出" in capsys.readouterr().out

    repl._handle_command("/export text")
    repl._handle_command("/export json ../outside.json")
    output = capsys.readouterr().out
    assert "用法: /export" in output
    assert "超出工作区" in output


def test_repl_run_dispatches_commands_and_turns(monkeypatch, capsys):
    repl = _repl(_config(autosave=False))
    lines = iter(["", "/help", "hello", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(lines))

    repl.run()

    output = capsys.readouterr().out
    assert "minicoder v" in output
    assert "可用命令" in output
    assert "done" in output
    assert "再见" in output


def test_repl_run_handles_end_of_input(monkeypatch, capsys):
    repl = _repl(_config(autosave=False))

    def end_input(_prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", end_input)
    repl.run()
    assert "再见" in capsys.readouterr().out


def test_repl_turn_reports_cancel_and_error(monkeypatch, capsys):
    repl = _repl(_config(autosave=False))

    def cancel(_text):
        raise KeyboardInterrupt

    monkeypatch.setattr(repl.agent, "chat", cancel)
    repl._run_turn("cancel")
    assert "已取消当前轮" in capsys.readouterr().out

    def fail(_text):
        raise RuntimeError("provider down")

    monkeypatch.setattr(repl.agent, "chat", fail)
    repl._run_turn("fail")
    assert "RuntimeError: provider down" in capsys.readouterr().out


def test_repl_commands_cover_model_tokens_compact_and_errors(monkeypatch, capsys):
    repl = _repl(_config(autosave=False))
    compact_calls = []
    monkeypatch.setattr(repl.agent, "compact", lambda *, force: compact_calls.append(force))
    monkeypatch.setattr(cli_mod, "_show_git_diff", lambda: print("diff output"))

    for command in (
        "/model",
        "/model changed",
        "/tokens",
        "/compact",
        "/diff",
        "/intent",
        "/unknown",
    ):
        assert repl._handle_command(command) is False

    monkeypatch.setattr(
        cli_mod,
        "save_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(SessionWriteError("disk full")),
    )
    repl._handle_command("/save broken")
    monkeypatch.setattr(cli_mod, "list_sessions", lambda: [])
    repl._handle_command("/sessions")
    repl._handle_command("/load")
    monkeypatch.setattr(
        cli_mod,
        "load_session",
        lambda _name: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    repl._handle_command("/load missing")

    output = capsys.readouterr().out
    assert "当前模型" in output
    assert "已切换模型: changed" in output
    assert "当前上下文" in output
    assert compact_calls == [True]
    assert "diff output" in output
    assert "未知命令" in output
    assert "当前没有结构化意图结果" in output
    assert "disk full" in output
    assert "暂无已保存会话" in output
    assert "用法: /load" in output
    assert "missing" in output


def test_intent_command_includes_execution_report(capsys):
    repl = _repl(_config(autosave=False))
    repl.agent.interpreter = SimpleNamespace(
        last_interpretation=SimpleNamespace(to_dict=lambda: {"intent": "trip_charge_planning"})
    )
    repl.agent.last_execution = SimpleNamespace(
        to_dict=lambda: {"status": "completed", "steps": []}
    )

    repl._show_intent()

    output = capsys.readouterr().out
    assert '"interpretation"' in output
    assert '"execution"' in output
    assert '"completed"' in output


def test_repl_tool_rendering_and_summary(capsys):
    repl = _repl(_config(autosave=False))
    call = ToolCall(id="1", name="read_file", arguments={"path": "x" * 100})
    repl._print_tool_start(call)
    repl._print_tool_result(call, "line\n" + "x" * 600)

    output = capsys.readouterr().out
    assert "read_file" in output
    assert "…" in output
    assert cli_mod._tool_summary(ToolCall("2", "noop", {})) == ""
    assert "$0.0013" in cli_mod._fmt_cost(100, 100, "unknown")


def test_git_diff_and_permission_helpers(monkeypatch, tmp_path, capsys):
    completed = SimpleNamespace(stdout=" file.py | 1 +")
    monkeypatch.setattr(cli_mod.subprocess, "run", lambda *_args, **_kwargs: completed)
    cli_mod._show_git_diff()
    assert "file.py" in capsys.readouterr().out

    def fail_git(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("git", 5)

    monkeypatch.setattr(cli_mod.subprocess, "run", fail_git)
    cli_mod._show_git_diff()
    assert "无法获取" in capsys.readouterr().out

    request = PermissionRequest("shell", "run command", "echo ok", tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "YES")
    assert cli_mod._confirm_permission(request) is True

    def cancel(_prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", cancel)
    assert cli_mod._confirm_permission(request) is False


def test_build_agent_wires_interactive_workspace(monkeypatch, tmp_path):
    config = _config(autosave=False)
    config.permission_mode = "ask"
    provider = OneReplyProvider()
    captured = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_mod, "get_provider", lambda _config: provider)
    monkeypatch.setattr(cli_mod, "build_system_prompt", lambda cwd: f"prompt:{cwd}")

    def fake_tools(_provider, _config, *, workspace, **_kwargs):
        captured["workspace"] = workspace
        return []

    monkeypatch.setattr(cli_mod, "build_tools", fake_tools)
    agent = cli_mod._build_agent(config, stream=True, interactive=True)

    assert agent.provider is provider
    assert agent.stream is True
    assert agent.system_prompt == f"prompt:{tmp_path}"
    assert captured["workspace"].confirm is cli_mod._confirm_permission


def test_automotive_agent_executes_local_knowledge_plan(monkeypatch, tmp_path):
    config = _config(autosave=False)
    config.application_profile = "automotive"
    config.intent_model = "intent-model"
    config.timezone = "Asia/Shanghai"
    config.intent_rule_fast_path = True
    config.vehicle_api_url = "http://127.0.0.1:8765"
    config.vehicle_api_token = "token"
    config.vehicle_api_timeout = 1
    config.vehicle_api_allow_remote = False
    config.vehicle_permission_mode = "allow"
    config.vehicle_allowed_ids = ()
    config.knowledge_dir = ""
    config.knowledge_index = ".minicoder/test-knowledge.db"
    config.knowledge_top_k = 3
    config.knowledge_auto_rebuild = True
    config.planner_mode = "rules"
    config.planner_model = "planner-model"
    config.context_window = 10_000
    provider = OneReplyProvider()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_mod, "get_provider", lambda _config: provider)
    monkeypatch.setattr(cli_mod, "build_system_prompt", lambda _cwd: "system")

    agent = cli_mod._build_agent(config, stream=False, interactive=False)
    response = agent.chat("动力电池怎么保养")

    assert response == "done"
    assert agent.last_execution is not None
    assert agent.last_execution.status == "completed"
    knowledge = agent.last_execution.outputs["vehicle_knowledge"]
    assert knowledge["results"][0]["document_id"] == "battery-care"
    assert knowledge["citations"]
    assert any(tool.name == "search_vehicle_knowledge" for tool in agent.tools)


class MainConfig(SimpleNamespace):
    def require_api_key(self):
        if not self.api_key:
            raise RuntimeError("missing key")


def _main_config(*, api_key="key"):
    return MainConfig(
        provider="openai",
        model="model",
        api_key=api_key,
        base_url="https://example.test/v1",
        max_rounds=2,
        permission_mode="strict",
        context_window=1_000,
        output_reserve=100,
        autosave=False,
        autosave_name="autosave",
    )


def test_main_handles_config_and_api_key_errors(monkeypatch, capsys):
    monkeypatch.setattr(
        cli_mod.Config,
        "load",
        lambda: (_ for _ in ()).throw(ValueError("bad config")),
    )
    assert cli_mod.main([]) == 2
    assert "配置错误:bad config" in capsys.readouterr().err

    monkeypatch.setattr(cli_mod.Config, "load", lambda: _main_config(api_key=""))
    assert cli_mod.main([]) == 2
    assert "missing key" in capsys.readouterr().err


def test_main_headless_overrides_and_failure(monkeypatch, capsys):
    config = _main_config()
    monkeypatch.delenv("MINICODER_PROVIDER", raising=False)
    monkeypatch.setattr(cli_mod.Config, "load", lambda: config)
    fake_agent = SimpleNamespace(chat=lambda prompt: f"answer:{prompt}")
    builds = []

    def fake_build(received, stream, *, interactive):
        builds.append((received, stream, interactive))
        return fake_agent

    monkeypatch.setattr(cli_mod, "_build_agent", fake_build)
    assert (
        cli_mod.main(
            [
                "-p",
                "hello",
                "--provider",
                "anthropic",
                "--model",
                "new-model",
                "--permission-mode",
                "allow",
                "--read-only",
            ]
        )
        == 0
    )
    assert config.model == "new-model"
    assert config.permission_mode == "allow"
    assert config.read_only is True
    assert builds[-1][1:] == (False, False)
    assert "answer:hello" in capsys.readouterr().out

    fake_agent.chat = lambda _prompt: (_ for _ in ()).throw(RuntimeError("failed"))
    assert cli_mod.main(["-p", "hello"]) == 1
    assert "RuntimeError: failed" in capsys.readouterr().err


def test_main_starts_interactive_repl(monkeypatch):
    config = _main_config()
    fake_agent = SimpleNamespace()
    runs = []
    monkeypatch.setattr(cli_mod.Config, "load", lambda: config)
    monkeypatch.setattr(cli_mod, "_build_agent", lambda *_args, **_kwargs: fake_agent)
    monkeypatch.setattr(
        cli_mod,
        "Repl",
        lambda agent, received: SimpleNamespace(run=lambda: runs.append((agent, received))),
    )

    assert cli_mod.main([]) == 0
    assert runs == [(fake_agent, config)]
