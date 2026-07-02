"""CLI / REPL —— 交互入口与斜杠命令。

REPL + 斜杠命令 + 一次性无头模式(-p/--print)。
斜杠命令:/model /compact /tokens /diff /save /sessions /load /export /help,以及 quit/exit。
Ctrl+C 取消当前一轮,回到提示符。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from . import __version__
from .agent import Agent
from .config import Config
from .context import input_budget, total_tokens
from .executor import ExecutionReport, PlanExecutor, PlanValidator
from .knowledge.service import build_knowledge_retriever
from .planner import ConstrainedHybridPlanner
from .prompt import build_system_prompt
from .providers import ToolCall, get_provider
from .security import PermissionMode, PermissionRequest, WorkspacePolicy, WorkspaceViolation
from .session import SessionError, export_session, list_sessions, load_session, save_session
from .tools import build_tools
from .tools.bash import classify_shell_risk
from .understanding import AUTOMOTIVE_PROMPT, Interpretation, VehicleIntentInterpreter
from .vehicle.actions import build_vehicle_registry
from .vehicle.client import VehicleClient
from .vehicle.policy import VehicleActionPolicy, VehiclePermissionRequest

# 简单价格表(美元/百万 token),仅估算用,可按需修改
_PRICE = {
    "default": (2.5, 10.0),
}


def _fmt_cost(inp: int, out: int, model: str) -> str:
    price_in, price_out = _PRICE.get(model, _PRICE["default"])
    cost = inp / 1e6 * price_in + out / 1e6 * price_out
    return f"输入 {inp} tok / 输出 {out} tok ≈ ${cost:.4f}"


class Repl:
    def __init__(self, agent: Agent, config: Config) -> None:
        self.agent = agent
        self.config = config
        self.workspace = Path.cwd().resolve()
        self._wire_callbacks()

    def _wire_callbacks(self) -> None:
        self.agent.on_text = lambda t: (sys.stdout.write(t), sys.stdout.flush())
        self.agent.on_notice = lambda m: print(f"\n\033[2m· {m}\033[0m")
        self.agent.on_tool_start = self._print_tool_start
        self.agent.on_tool_result = self._print_tool_result
        if getattr(self.config, "autosave", True):
            self.agent.on_checkpoint = self._autosave
        if self.agent.interpreter is not None:
            self.agent.on_interpretation = self._print_interpretation
        if self.agent.plan_executor is not None:
            self.agent.on_plan_execution = self._print_plan_execution

    def _session_meta(self, reason: str | None = None) -> dict[str, str]:
        meta = {
            "model": self.config.model,
            "provider": self.config.provider,
            "workspace": str(self.workspace),
            "profile": getattr(self.config, "application_profile", "coding"),
        }
        if reason:
            meta["checkpoint_reason"] = reason
        return meta

    def _autosave(self, transcript: list[dict], reason: str) -> None:
        name = getattr(self.config, "autosave_name", "autosave")
        save_session(name, transcript, self._session_meta(reason))

    def _print_tool_start(self, tc: ToolCall) -> None:
        summary = _tool_summary(tc)
        risk = ""
        if tc.name == "bash":
            level = classify_shell_risk(str(tc.arguments.get("command", "")))
            risk = f" \033[33m[risk:{level.value}]\033[0m"
        print(f"\n\033[36m⚙ {tc.name}\033[0m{risk} {summary}")

    def _print_tool_result(self, tc: ToolCall, out: str) -> None:
        preview = out if len(out) <= 500 else out[:500] + f" … (+{len(out) - 500} 字符)"
        indented = "\n".join(f"  {line}" for line in preview.splitlines())
        print(f"\033[2m{indented}\033[0m")

    # ---- 主循环 ----
    def run(self) -> None:
        print(
            f"minicoder v{__version__} · provider={self.config.provider} · "
            f"model={self.config.model} · permissions={self.config.permission_mode} · "
            f"read_only={'on' if self.config.read_only else 'off'} · "
            f"profile={getattr(self.config, 'application_profile', 'coding')}"
        )
        print("输入你的需求,或 /help 看命令,quit 退出。\n")
        while True:
            try:
                line = input("\033[1m›\033[0m ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                return
            if not line:
                continue
            if line.startswith("/") or line in ("quit", "exit"):
                if self._handle_command(line):
                    return
                continue
            self._run_turn(line)

    def _run_turn(self, text: str) -> None:
        try:
            result = self.agent.chat(text)
            if not self.agent.stream:
                print(result)
            print()  # 收尾换行
        except KeyboardInterrupt:
            print("\n\033[33m已取消当前轮。\033[0m")
        except Exception as e:  # noqa: BLE001
            print(f"\n\033[31m错误:{type(e).__name__}: {e}\033[0m")

    # ---- 斜杠命令,返回 True 表示应退出 ----
    def _handle_command(self, line: str) -> bool:
        parts = line.split(maxsplit=1)
        cmd = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "/quit", "/exit"):
            print("再见。")
            return True
        if cmd == "/help":
            _print_help()
        elif cmd == "/model":
            if arg:
                self.config.model = arg
                self.agent.config.model = arg
                print(f"已切换模型: {arg}")
            else:
                print(f"当前模型: {self.config.model}")
        elif cmd == "/tokens":
            ctx = total_tokens(
                self.agent.messages,
                system=self.agent.system_prompt,
                tools=[tool.schema() for tool in self.agent.tools],
                model=self.config.model,
            )
            budget = input_budget(self.config.context_window, self.config.output_reserve)
            print(f"当前上下文 ≈ {ctx} tok ({ctx / budget:.0%} 输入预算)")
            print(
                f"窗口 {self.config.context_window} tok / "
                f"预留输出 {self.config.output_reserve} tok / 输入预算 {budget} tok"
            )
            print(
                "累计 "
                + _fmt_cost(
                    self.agent.total_input_tokens, self.agent.total_output_tokens, self.config.model
                )
            )
        elif cmd == "/compact":
            before = len(self.agent.messages)
            self.agent.compact(force=True)
            print(f"压缩:{before} → {len(self.agent.messages)} 条消息")
        elif cmd == "/diff":
            _show_git_diff()
        elif cmd == "/save":
            name = arg or f"session-{int(time.time())}"
            try:
                path = save_session(name, self.agent.transcript, self._session_meta("manual"))
                print(f"已保存: {path}")
            except SessionError as e:
                print(f"错误:{e}")
        elif cmd == "/sessions":
            self._list_sessions()
        elif cmd == "/load":
            self._load_session(arg)
        elif cmd == "/export":
            self._export_session(arg)
        elif cmd == "/intent":
            self._show_intent()
        else:
            print(f"未知命令: {cmd}(/help 查看可用命令)")
        return False

    def _print_interpretation(self, interpretation: Interpretation) -> None:
        result = interpretation.result
        entities = ", ".join(f"{key}={value}" for key, value in result.entities.items())
        suffix = f" · {entities}" if entities else ""
        print(
            f"\n\033[2m· intent={result.intent.value} · status={result.status.value} "
            f"· source={interpretation.source}{suffix}\033[0m"
        )

    def _show_intent(self) -> None:
        interpreter = self.agent.interpreter
        interpretation = interpreter.last_interpretation if interpreter else None
        if interpretation is None:
            print("(当前没有结构化意图结果；请启用 automotive profile 并发送请求)")
            return
        payload = {
            "interpretation": interpretation.to_dict(),
            "execution": (
                self.agent.last_execution.to_dict()
                if self.agent.last_execution is not None
                else None
            ),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    def _print_plan_execution(self, report: ExecutionReport) -> None:
        completed = sum(step.status == "completed" for step in report.steps)
        print(
            f"\033[2m· vehicle plan={report.status} · completed={completed}/{len(report.steps)}\033[0m"
        )

    def _list_sessions(self) -> None:
        sessions = list_sessions()
        if not sessions:
            print("(暂无已保存会话)")
            return
        for s in sessions:
            when = (
                time.strftime("%Y-%m-%d %H:%M", time.localtime(s["saved_at"]))
                if s["saved_at"]
                else "未知时间"
            )
            print(f"  [{s['status']:<11}] {s['name']:<24} {s['messages']:>4} 条消息  {when}")
        print("用 /load <名称> 恢复。")

    def _load_session(self, name: str) -> None:
        if not name:
            print("用法: /load <会话名>")
            return
        try:
            messages, meta = load_session(name)
        except (FileNotFoundError, SessionError) as e:
            print(f"错误:{e}")
            return
        self.agent.messages = messages
        if meta.get("model"):
            self.config.model = meta["model"]
            self.agent.config.model = meta["model"]
        print(f"已恢复会话 {name}({len(messages)} 条消息)")
        if meta.get("_session_recovered_from_backup"):
            print("警告: 主会话损坏或缺失，已从上一份有效备份恢复。")

    def _export_session(self, argument: str) -> None:
        parts = argument.split(maxsplit=1)
        export_format = parts[0].lower() if parts else ""
        if export_format not in {"json", "md", "markdown"}:
            print("用法: /export <json|markdown> [工作区内路径]")
            return
        extension = "json" if export_format == "json" else "md"
        raw_path = (
            parts[1].strip().strip('"').strip("'")
            if len(parts) > 1
            else f"session-{int(time.time())}.{extension}"
        )
        try:
            destination = WorkspacePolicy(
                self.workspace,
                mode=PermissionMode.ALLOW,
            ).resolve_path(raw_path, operation="导出会话")
            path = export_session(
                destination,
                self.agent.transcript,
                self._session_meta("export"),
                format=export_format,
            )
        except (SessionError, WorkspaceViolation) as error:
            print(f"错误:{error}")
            return
        print(f"已导出: {path}")


def _tool_summary(tc: ToolCall) -> str:
    a = tc.arguments
    for key in ("path", "pattern", "command", "task"):
        if key in a:
            val = str(a[key])
            return val if len(val) <= 80 else val[:80] + "…"
    return ""


def _show_git_diff() -> None:
    try:
        out = subprocess.run(["git", "diff", "--stat"], capture_output=True, text=True, timeout=5)
        print(out.stdout.strip() or "(本次会话无 git 改动)")
    except (OSError, subprocess.SubprocessError):
        print("(无法获取 git diff)")


def _print_help() -> None:
    print(
        """可用命令:
  /model [名称]    查看或切换模型
  /compact         立即压缩上下文
  /tokens          查看 token 用量与成本估算
  /diff            查看 git 改动(git diff --stat)
  /save [名称]     保存当前会话
  /sessions        列出已保存会话
  /load <名称>     恢复某个会话
  /export <格式> [路径]  导出当前会话(json|markdown)
  /intent          查看最近一次结构化意图、初始计划与执行报告
  /help            显示本帮助
  quit / exit      退出(对话中 Ctrl+C 取消当前轮)"""
    )


def _confirm_permission(request: PermissionRequest) -> bool:
    print(f"\n\033[33m权限确认 [risk:{request.risk.value}]: {request.description}\033[0m")
    print(f"工作区: {request.workspace}")
    print(f"命令: {request.command}")
    try:
        answer = input("允许本次执行? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in {"y", "yes"}


def _confirm_vehicle_action(request: VehiclePermissionRequest) -> bool:
    print(f"\n\033[33m车辆控制确认 [risk:{request.risk.value}]: {request.description}\033[0m")
    print(f"车辆: {request.vehicle_id}")
    try:
        answer = input("允许本次车辆控制? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in {"y", "yes"}


def _build_agent(config: Config, stream: bool, *, interactive: bool) -> Agent:
    provider = get_provider(config)
    cwd = Path.cwd()
    mode = PermissionMode.parse(config.permission_mode)
    workspace = WorkspacePolicy(
        cwd,
        mode=mode,
        confirm=_confirm_permission if interactive and mode is PermissionMode.ASK else None,
        read_only=config.read_only,
    )
    system_prompt = build_system_prompt(cwd)
    interpreter = None
    plan_executor = None
    vehicle_client = None
    vehicle_policy = None
    action_registry = None
    if getattr(config, "application_profile", "coding") == "automotive":
        system_prompt += "\n\n" + AUTOMOTIVE_PROMPT
        vehicle_client = VehicleClient(
            getattr(config, "vehicle_api_url", "http://127.0.0.1:8765"),
            getattr(config, "vehicle_api_token", "local-demo-token"),
            timeout=float(getattr(config, "vehicle_api_timeout", 5.0)),
            allow_remote=bool(getattr(config, "vehicle_api_allow_remote", False)),
        )
        allowed_ids = tuple(getattr(config, "vehicle_allowed_ids", ()))
        vehicle_policy = VehicleActionPolicy(
            mode=PermissionMode.parse(getattr(config, "vehicle_permission_mode", "ask")),
            confirm=_confirm_vehicle_action if interactive else None,
            read_only=config.read_only,
            allowed_vehicle_ids=frozenset(allowed_ids) if allowed_ids else None,
        )
        knowledge_retriever = build_knowledge_retriever(config, cwd)
        action_registry = build_vehicle_registry(
            vehicle_client,
            vehicle_policy,
            knowledge_retriever,
            knowledge_top_k=int(getattr(config, "knowledge_top_k", 5)),
        )
        plan_validator = PlanValidator(action_registry, read_only=config.read_only)
        planner = ConstrainedHybridPlanner(
            provider,
            getattr(config, "planner_model", config.model),
            action_registry,
            validate=plan_validator.validate,
            enabled=getattr(config, "planner_mode", "rules") == "hybrid",
        )
        interpreter = VehicleIntentInterpreter(
            provider,
            model=getattr(config, "intent_model", config.model),
            timezone_name=getattr(config, "timezone", "Asia/Shanghai"),
            rule_fast_path=bool(getattr(config, "intent_rule_fast_path", True)),
            planner=planner,
        )
        plan_executor = PlanExecutor(action_registry, validator=plan_validator)
    tools = build_tools(
        provider,
        config,
        workspace=workspace,
        vehicle_client=vehicle_client,
        vehicle_policy=vehicle_policy,
        action_registry=action_registry,
    )
    return Agent(
        provider=provider,
        config=config,
        tools=tools,
        system_prompt=system_prompt,
        stream=stream,
        interpreter=interpreter,
        plan_executor=plan_executor,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minicoder", description="极简终端 AI coding agent")
    parser.add_argument("-p", "--print", dest="prompt", help="无头模式:执行单条指令后退出")
    parser.add_argument("--provider", help="覆盖 provider(openai|anthropic)")
    parser.add_argument("--model", help="覆盖模型")
    parser.add_argument(
        "--profile",
        choices=["coding", "automotive"],
        help="应用 Profile（coding|automotive）",
    )
    parser.add_argument("--analyze-intent", metavar="TEXT", help="输出汽车意图、实体与初始计划")
    parser.add_argument(
        "--permission-mode",
        choices=[mode.value for mode in PermissionMode],
        help="权限模式(allow|ask|deny)",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="只读模式:拒绝文件写入和 Shell",
    )
    parser.add_argument("-v", "--version", action="version", version=f"minicoder {__version__}")
    args = parser.parse_args(argv)

    try:
        config = Config.load()
    except ValueError as e:
        print(f"配置错误:{e}", file=sys.stderr)
        return 2
    if args.provider:
        import os

        os.environ["MINICODER_PROVIDER"] = args.provider
        config = Config.load()
    if args.model:
        config.model = args.model
    if args.profile:
        config.application_profile = args.profile
    if args.permission_mode:
        config.permission_mode = args.permission_mode
    if args.read_only:
        config.read_only = True

    try:
        config.require_api_key()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    if args.analyze_intent:
        config.application_profile = "automotive"
        agent = _build_agent(config, stream=False, interactive=False)
        assert agent.interpreter is not None
        interpretation = agent.interpreter.interpret(args.analyze_intent)
        print(json.dumps(interpretation.to_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.prompt:  # 无头模式
        agent = _build_agent(config, stream=False, interactive=False)
        try:
            print(agent.chat(args.prompt))
        except Exception as e:  # noqa: BLE001
            print(f"错误:{type(e).__name__}: {e}", file=sys.stderr)
            return 1
        return 0

    Repl(_build_agent(config, stream=True, interactive=True), config).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
