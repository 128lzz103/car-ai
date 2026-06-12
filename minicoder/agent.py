"""Agent Loop —— minicoder 的心脏。

对标本仓库 docs/02-core-engine.md:
- while 循环:发消息 → 若有 tool_use 则执行并回填结果 → 再发,直到无 tool_use
- 读写分离并发(partitionToolCalls):连续只读工具并发执行,写工具串行
- 轮数上限防跑飞;Ctrl+C 取消当前轮
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from .context import ContextManager
from .providers import Provider, Reply, ToolCall
from .retry import RetryEvent
from .tools.base import Tool

if TYPE_CHECKING:
    from .executor import ExecutionReport, PlanExecutor
    from .understanding import Interpretation, RequestInterpreter

MAX_CONCURRENCY = 8  # 只读工具组的最大并发(对标 CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY)


class Agent:
    def __init__(
        self,
        provider: Provider,
        config: Any,
        tools: list[Tool],
        system_prompt: str,
        stream: bool = True,
        messages: list[dict[str, Any]] | None = None,
        interpreter: RequestInterpreter | None = None,
        plan_executor: PlanExecutor | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.tools = tools
        self.tools_by_name = {t.name: t for t in tools}
        self.system_prompt = system_prompt
        self.stream = stream
        self.interpreter = interpreter
        self.plan_executor = plan_executor
        self.last_execution: ExecutionReport | None = None
        self.context = ContextManager(messages)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        # UI 回调(CLI 注入):流式文本、通知、工具开始
        self.on_text: Callable[[str], None] | None = None
        self.on_notice: Callable[[str], None] | None = None
        self.on_tool_start: Callable[[ToolCall], None] | None = None
        self.on_tool_result: Callable[[ToolCall, str], None] | None = None
        self.on_checkpoint: Callable[[list[dict[str, Any]], str], None] | None = None
        self.on_interpretation: Callable[[Interpretation], None] | None = None
        self.on_plan_execution: Callable[[ExecutionReport], None] | None = None
        if hasattr(self.provider, "on_retry") and self.provider.on_retry is None:
            self.provider.on_retry = self._on_retry

    # ---- 单次用户输入 → 跑完整个 agent turn,返回最终文本 ----
    def chat(self, user_text: str) -> str:
        self.last_execution = None
        self.context.append({"role": "user", "content": user_text})
        self._checkpoint("user")
        effective_system = self.system_prompt
        if self.interpreter is not None:
            interpretation = self.interpreter.interpret(user_text)
            self.total_input_tokens += interpretation.input_tokens
            self.total_output_tokens += interpretation.output_tokens
            if self.on_interpretation:
                self.on_interpretation(interpretation)
            if interpretation.clarification:
                if self.stream and self.on_text:
                    self.on_text(interpretation.clarification)
                self.context.append(
                    {"role": "assistant", "content": interpretation.clarification, "tool_calls": []}
                )
                self._checkpoint("assistant_clarification")
                return interpretation.clarification
            effective_system += (
                "\n\n# 当前请求的已验证结构化理解\n" + interpretation.prompt_context()
            )
            if self.plan_executor is not None and interpretation.plan.status == "ready":
                execution = self.plan_executor.execute(interpretation.plan)
                self.last_execution = execution
                if self.on_plan_execution:
                    self.on_plan_execution(execution)
                import json

                effective_system += (
                    "\n\n# 已完成的确定性车辆计划执行结果\n"
                    + json.dumps(execution.to_dict(), ensure_ascii=False, separators=(",", ":"))
                    + "\n只能依据这些结果回答，不要重复调用已完成的车辆工具。"
                )
        supports_tools = self._supports("tool_calls")
        effective_stream = self.stream and self._supports("streaming")
        tool_schemas = [t.schema() for t in self.tools] if supports_tools else []
        if self.tools and not supports_tools:
            self._notice("当前 Provider 不支持工具调用，本轮将仅使用文本能力。")
        if self.stream and not effective_stream:
            self._notice("当前 Provider 不支持流式输出，已自动切换为非流式。")

        for _ in range(self.config.max_rounds):
            messages = self.context.prepare(
                self.provider,
                self.config.model,
                context_window=getattr(self.config, "context_window", 128_000),
                output_reserve=getattr(self.config, "output_reserve", 8_192),
                system=effective_system,
                tools=tool_schemas,
                on_notice=self._notice,
            )

            reply = self.provider.chat(
                messages=messages,
                tools=tool_schemas,
                system=effective_system,
                model=self.config.model,
                stream=effective_stream,
                on_text=self.on_text if effective_stream else None,
            )
            self.total_input_tokens += reply.input_tokens
            self.total_output_tokens += reply.output_tokens

            # 非流式:文本没通过 on_text 输出过,补一次
            if not effective_stream and reply.text and self.on_text:
                self.on_text(reply.text)

            self.context.append(self._assistant_msg(reply))

            if not reply.tool_calls:
                self._checkpoint("assistant_final")
                return reply.text

            results = self._run_tools(reply.tool_calls)
            self.context.extend(results)
            self._checkpoint("tool_results")

        self._notice(f"已达最大轮数 {self.config.max_rounds},停止。")
        self._checkpoint("max_rounds")
        return "(已达最大轮数限制)"

    @property
    def messages(self) -> list[dict[str, Any]]:
        """兼容旧接口：返回当前发送给模型的 working context。"""
        return self.context.working_messages

    @messages.setter
    def messages(self, messages: list[dict[str, Any]]) -> None:
        self.context.reset(messages)

    @property
    def transcript(self) -> list[dict[str, Any]]:
        return self.context.transcript

    def compact(self, *, force: bool = False) -> list[dict[str, Any]]:
        return self.context.prepare(
            self.provider,
            self.config.model,
            context_window=getattr(self.config, "context_window", 128_000),
            output_reserve=getattr(self.config, "output_reserve", 8_192),
            system=self.system_prompt,
            tools=[tool.schema() for tool in self.tools],
            on_notice=self._notice,
            force=force,
        )

    @staticmethod
    def _assistant_msg(reply: Reply) -> dict[str, Any]:
        return {"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls}

    def _notice(self, msg: str) -> None:
        if self.on_notice:
            self.on_notice(msg)

    def _supports(self, capability: str) -> bool:
        capabilities = getattr(self.provider, "capabilities", None)
        return bool(getattr(capabilities, capability, True))

    def _checkpoint(self, reason: str) -> None:
        if not self.on_checkpoint:
            return
        try:
            self.on_checkpoint(list(self.transcript), reason)
        except Exception as error:  # noqa: BLE001 -- 持久化失败不能中断 Agent Loop
            self._notice(f"会话检查点保存失败 ({reason}): {type(error).__name__}: {error}")

    def _on_retry(self, event: RetryEvent) -> None:
        self._notice(
            f"Provider 请求失败: {event.reason}; {event.delay:.1f}s 后重试 "
            f"({event.attempt}/{event.max_attempts})"
        )

    # ---- 读写分离执行(docs/02 的 partitionToolCalls) ----
    def _run_tools(self, tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
        """把工具调用分区成批次:连续只读 → 并发;写工具 → 各自串行。
        返回顺序与输入顺序严格一致(tool_result 必须对齐 tool_call)。"""
        results_by_id: dict[str, str] = {}

        for batch in self._partition(tool_calls):
            if len(batch) > 1:  # 只读并发批
                with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
                    for tc, out in zip(batch, pool.map(self._exec_one, batch)):
                        results_by_id[tc.id] = out
            else:  # 单个(写工具或落单的只读)串行
                tc = batch[0]
                results_by_id[tc.id] = self._exec_one(tc)

        return [
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": results_by_id[tc.id],
            }
            for tc in tool_calls
        ]

    def _partition(self, tool_calls: list[ToolCall]) -> list[list[ToolCall]]:
        """连续的并发安全(只读)工具归为一个并发批;其余各自单独成批。"""
        batches: list[list[ToolCall]] = []
        for tc in tool_calls:
            tool = self.tools_by_name.get(tc.name)
            safe = bool(tool and tool.is_concurrency_safe())
            if safe and batches and self._batch_is_safe(batches[-1]):
                batches[-1].append(tc)
            else:
                batches.append([tc])
        return batches

    def _batch_is_safe(self, batch: list[ToolCall]) -> bool:
        tool = self.tools_by_name.get(batch[0].name)
        return bool(tool and tool.is_concurrency_safe())

    def _exec_one(self, tc: ToolCall) -> str:
        if self.on_tool_start:
            self.on_tool_start(tc)
        tool = self.tools_by_name.get(tc.name)
        if tool is None:
            out = f"错误:未知工具 {tc.name!r}"
        else:
            try:
                out = tool.run(tc.arguments)
            except Exception as e:  # noqa: BLE001 —— 工具错误回填给模型,不炸循环
                out = f"错误:工具 {tc.name} 执行失败: {type(e).__name__}: {e}"
        if self.on_tool_result:
            self.on_tool_result(tc, out)
        return out
