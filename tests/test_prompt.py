"""System prompt 的项目记忆、Git 快照与容错测试。"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

from minicoder import prompt as prompt_mod


def test_build_prompt_without_repository(tmp_path):
    prompt = prompt_mod.build_system_prompt(tmp_path)
    assert prompt_mod.SYSTEM_PROMPT in prompt
    assert f"工作目录: {tmp_path}" in prompt
    assert "今天的日期:" in prompt
    assert "Git 状态" not in prompt


def test_project_memory_prefers_agents_and_is_bounded(tmp_path):
    (tmp_path / "AGENTS.md").write_text("A" * 9_000, encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("should not load", encoding="utf-8")

    memory = prompt_mod._project_memory(tmp_path)

    assert memory.startswith("# 项目说明(AGENTS.md)\n")
    assert len(memory.removeprefix("# 项目说明(AGENTS.md)\n")) == 8_000
    assert "should not load" not in memory


def test_git_context_includes_branch_status_and_log(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()

    def fake_git(args, _cwd):
        if args[0] == "rev-parse":
            return "main"
        if args[0] == "status":
            return "M file.py\n" + "x" * 2_100
        return "abc initial"

    monkeypatch.setattr(prompt_mod, "_run_git", fake_git)
    context = prompt_mod._git_context(tmp_path)

    assert "当前分支: main" in context
    assert "改动:\nM file.py" in context
    assert "最近提交:\nabc initial" in context
    assert "x" * 2_001 not in context


def test_run_git_success_nonzero_and_exception(monkeypatch, tmp_path):
    monkeypatch.setattr(
        prompt_mod.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=" main \n"),
    )
    assert prompt_mod._run_git(["status"], tmp_path) == "main"

    monkeypatch.setattr(
        prompt_mod.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="ignored"),
    )
    assert prompt_mod._run_git(["status"], tmp_path) == ""

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("git", 5)

    monkeypatch.setattr(prompt_mod.subprocess, "run", timeout)
    assert prompt_mod._run_git(["status"], tmp_path) == ""
