"""协议安全的上下文管理。

ContextManager 同时维护完整 transcript 和可压缩 working context。压缩以完整的
工具交换组为单位，绝不拆开 assistant tool_calls 与对应的 tool results。
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .models import get_model_profile

if TYPE_CHECKING:
    from .providers import Provider

DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_OUTPUT_RESERVE = 8_192
DEFAULT_SAFETY_MARGIN = 0.05

TIER1_RATIO = 0.50
TIER2_RATIO = 0.70
TIER3_RATIO = 0.90

TOOL_OUTPUT_MAX_CHARS = 4_000
EMERGENCY_TOOL_OUTPUT_MAX_CHARS = 1_500
KEEP_RECENT_GROUPS = 3


class ContextProtocolError(ValueError):
    """消息序列违反工具调用协议。"""


class ContextOverflowError(RuntimeError):
    """安全压缩后仍无法放入模型输入预算。"""


@dataclass(frozen=True)
class MessageGroup:
    """不可拆分的协议消息组。"""

    kind: str
    messages: tuple[dict[str, Any], ...]


_TOKEN_CHUNKS = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]+"
    r"|[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]|\s+"
)


def _is_cjk_chunk(value: str) -> bool:
    if not value:
        return False
    code = ord(value[0])
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )


def count_tokens(text: str, model: str = "") -> int:
    """按模型档案对 CJK、单词、数字与标点做确定性的本地估算。"""
    if not text:
        return 1
    profile = get_model_profile(model)
    total = 0
    for chunk in _TOKEN_CHUNKS.findall(text):
        if chunk.isspace():
            total += len(chunk) // 8
        elif _is_cjk_chunk(chunk):
            total += math.ceil(len(chunk) / profile.cjk_chars_per_token)
        elif chunk.isdigit():
            total += math.ceil(len(chunk) / 3)
        elif chunk[0].isalpha():
            total += max(1, math.ceil(len(chunk) / profile.chars_per_token))
        else:
            total += len(chunk)
    return max(1, total)


def _tool_call_value(tool_call: Any, key: str, default: Any = None) -> Any:
    if isinstance(tool_call, dict):
        return tool_call.get(key, default)
    return getattr(tool_call, key, default)


def _message_text(message: dict[str, Any]) -> str:
    parts = [str(message.get("role") or ""), str(message.get("content") or "")]
    if message.get("tool_call_id"):
        parts.append(str(message["tool_call_id"]))
    if message.get("name"):
        parts.append(str(message["name"]))
    for tool_call in message.get("tool_calls") or []:
        parts.append(str(_tool_call_value(tool_call, "id", "")))
        parts.append(str(_tool_call_value(tool_call, "name", "")))
        arguments = _tool_call_value(tool_call, "arguments", {})
        parts.append(json.dumps(arguments, ensure_ascii=False, sort_keys=True))
    return " ".join(parts)


def total_tokens(
    messages: list[dict[str, Any]],
    *,
    system: str = "",
    tools: list[dict[str, Any]] | None = None,
    model: str = "",
) -> int:
    """估算消息、system prompt、工具 schema 及消息协议开销。"""
    profile = get_model_profile(model)
    total = count_tokens(system, model) if system else 0
    total += sum(
        profile.message_overhead + count_tokens(_message_text(message), model)
        for message in messages
    )
    if tools:
        total += count_tokens(json.dumps(tools, ensure_ascii=False, sort_keys=True), model)
    return total


def input_budget(
    context_window: int,
    output_reserve: int = DEFAULT_OUTPUT_RESERVE,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
) -> int:
    reserve = max(0, output_reserve)
    margin = max(0, int(context_window * safety_margin))
    return max(1, context_window - reserve - margin)


def _tool_call_ids(message: dict[str, Any]) -> list[str]:
    ids = [str(_tool_call_value(tc, "id", "")) for tc in message.get("tool_calls") or []]
    if any(not tool_id for tool_id in ids):
        raise ContextProtocolError("assistant tool_call 缺少 id")
    if len(ids) != len(set(ids)):
        raise ContextProtocolError("assistant tool_call id 重复")
    return ids


def group_messages(messages: list[dict[str, Any]]) -> list[MessageGroup]:
    """按完整 Turn 分组，并验证 tool call/result 的完整对应关系。"""
    groups: list[MessageGroup] = []
    turn: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if role == "tool":
            raise ContextProtocolError(f"发现孤立 tool result: {message.get('tool_call_id')!r}")
        if role not in {"user", "assistant"}:
            raise ContextProtocolError(f"不支持的消息角色: {role!r}")

        if role == "user" and turn:
            groups.append(MessageGroup("turn", tuple(turn)))
            turn = []
        turn.append(message)

        tool_calls = message.get("tool_calls") or []
        if role != "assistant" or not tool_calls:
            index += 1
            continue

        expected_ids = _tool_call_ids(message)
        index += 1
        results: list[dict[str, Any]] = []
        while index < len(messages) and messages[index].get("role") == "tool":
            results.append(messages[index])
            index += 1
        result_ids = [str(item.get("tool_call_id") or "") for item in results]
        if result_ids != expected_ids:
            missing = [tool_id for tool_id in expected_ids if tool_id not in result_ids]
            unexpected = [tool_id for tool_id in result_ids if tool_id not in expected_ids]
            raise ContextProtocolError(
                "tool result 与 assistant tool_calls 不完整对应"
                f"(expected={expected_ids}, actual={result_ids}, "
                f"missing={missing}, unexpected={unexpected})"
            )
        turn.extend(results)
    if turn:
        groups.append(MessageGroup("turn", tuple(turn)))
    return groups


def validate_tool_protocol(messages: list[dict[str, Any]]) -> None:
    group_messages(messages)


def _flatten(groups: list[MessageGroup]) -> list[dict[str, Any]]:
    return [message for group in groups for message in group.messages]


def _trim_tool_outputs(
    messages: list[dict[str, Any]],
    max_chars: int,
    *,
    protected_ids: set[int] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    protected_ids = protected_ids or set()
    out: list[dict[str, Any]] = []
    trimmed_count = 0
    for message in messages:
        content = str(message.get("content") or "")
        if (
            message.get("role") == "tool"
            and id(message) not in protected_ids
            and len(content) > max_chars
        ):
            half = max_chars // 2
            removed = len(content) - max_chars
            trimmed = (
                f"{content[:half]}\n\n... [已裁剪 {removed} 字符; "
                "如仍需要请重新读取] ...\n\n"
                f"{content[-half:]}"
            )
            out.append({**message, "content": trimmed})
            trimmed_count += 1
        else:
            out.append(message)
    return (out if trimmed_count else messages), trimmed_count


def _summarize(
    provider: Provider,
    messages: list[dict[str, Any]],
    model: str,
    *,
    tight: bool,
) -> str:
    transcript_parts = []
    for message in messages:
        text = str(message.get("content") or "")
        for tool_call in message.get("tool_calls") or []:
            name = _tool_call_value(tool_call, "name", "")
            arguments = _tool_call_value(tool_call, "arguments", {})
            text += f"\n[调用工具 {name}({arguments})]"
        transcript_parts.append(f"{message['role']}: {text}")

    instruction = (
        "把下面的对话压缩成最紧凑的要点，只保留后续任务必需的信息：目标、关键决策、"
        "已修改文件、验证结果、错误原因和未完成事项。"
        if tight
        else "摘要下面较早的对话，保留目标、关键决策、文件改动、验证结果和待办。"
    )
    reply = provider.chat(
        messages=[
            {
                "role": "user",
                "content": f"{instruction}\n\n---\n" + "\n".join(transcript_parts),
            }
        ],
        tools=[],
        system="你是对话摘要器，只输出忠实、紧凑的上下文摘要。",
        model=model,
        stream=False,
    )
    return reply.text.strip() or "(摘要为空)"


def _notice(callback: Callable[[str], None] | None, message: str) -> None:
    if callback:
        callback(message)


def compact_if_needed(
    messages: list[dict[str, Any]],
    provider: Provider,
    model: str,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    on_notice: Callable[[str], None] | None = None,
    *,
    system: str = "",
    tools: list[dict[str, Any]] | None = None,
    output_reserve: int = 0,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    force: bool = False,
) -> list[dict[str, Any]]:
    """返回协议完整、符合输入预算的 working context。"""
    groups = group_messages(messages)
    budget = input_budget(context_window, output_reserve, safety_margin)

    def used(items: list[dict[str, Any]]) -> int:
        return total_tokens(items, system=system, tools=tools, model=model)

    original_tokens = used(messages)
    ratio = original_tokens / budget
    if not force and ratio < TIER1_RATIO:
        return messages

    protected = {id(message) for message in groups[-1].messages} if groups else set()
    working, trimmed = _trim_tool_outputs(
        messages,
        TOOL_OUTPUT_MAX_CHARS,
        protected_ids=protected,
    )
    if not force and used(working) / budget < TIER2_RATIO:
        if trimmed:
            _notice(on_notice, f"上下文压缩:裁剪 {trimmed} 条较旧工具输出(tier1)")
        return working

    current_ratio = used(working) / budget
    tight = force or current_ratio >= TIER3_RATIO
    if tight:
        working, recent_trimmed = _trim_tool_outputs(working, TOOL_OUTPUT_MAX_CHARS)
        trimmed += recent_trimmed

    groups = group_messages(working)
    keep_recent = 1 if tight else KEEP_RECENT_GROUPS
    candidate = working
    summarized = 0
    summary_failed = False

    if len(groups) > keep_recent + 1:
        first_group = groups[0]
        recent_groups = groups[-keep_recent:]
        older_groups = groups[1:-keep_recent]
        older_messages = _flatten(older_groups)
        try:
            summary = _summarize(provider, older_messages, model, tight=tight)
        except Exception as error:  # noqa: BLE001 - 压缩失败不应直接炸掉主任务
            summary_failed = True
            _notice(
                on_notice,
                f"上下文摘要失败，已回退到工具输出裁剪: {type(error).__name__}: {error}",
            )
        else:
            summary_message = {
                "role": "user",
                "content": f"[早前对话摘要]\n{summary}",
                "_context_summary": True,
            }
            candidate = [
                *first_group.messages,
                summary_message,
                *_flatten(recent_groups),
            ]
            summarized = len(older_messages)

    if tight or used(candidate) > budget:
        candidate, emergency_trimmed = _trim_tool_outputs(
            candidate,
            EMERGENCY_TOOL_OUTPUT_MAX_CHARS,
        )
        trimmed += emergency_trimmed

    validate_tool_protocol(candidate)
    final_tokens = used(candidate)
    if final_tokens > budget:
        suffix = "，且摘要服务不可用" if summary_failed else ""
        raise ContextOverflowError(
            f"上下文压缩后仍超出输入预算: {final_tokens} > {budget} token{suffix}。"
            "请缩短当前请求、重新开始会话或提高 MINICODER_CONTEXT_WINDOW。"
        )

    if summarized:
        tier = "tier3 紧急" if tight else "tier2"
        _notice(on_notice, f"上下文压缩:摘要 {summarized} 条历史消息({tier})")
    elif trimmed:
        _notice(on_notice, f"上下文压缩:裁剪 {trimmed} 条工具输出")
    elif force:
        _notice(on_notice, "上下文压缩:当前没有可安全压缩的历史消息")
    return candidate


class ContextManager:
    """维护完整 transcript 与发送给模型的 working context。"""

    def __init__(self, messages: list[dict[str, Any]] | None = None) -> None:
        self.reset(messages or [])

    @property
    def transcript(self) -> list[dict[str, Any]]:
        return self._transcript

    @property
    def working_messages(self) -> list[dict[str, Any]]:
        return self._working

    def reset(self, messages: list[dict[str, Any]]) -> None:
        validate_tool_protocol(messages)
        self._transcript = list(messages)
        self._working = list(messages)

    def append(self, message: dict[str, Any]) -> None:
        self._transcript.append(message)
        self._working.append(message)

    def extend(self, messages: list[dict[str, Any]]) -> None:
        self._transcript.extend(messages)
        self._working.extend(messages)

    def prepare(
        self,
        provider: Provider,
        model: str,
        *,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        output_reserve: int = DEFAULT_OUTPUT_RESERVE,
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        on_notice: Callable[[str], None] | None = None,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        self._working = compact_if_needed(
            self._working,
            provider,
            model,
            context_window=context_window,
            on_notice=on_notice,
            system=system,
            tools=tools,
            output_reserve=output_reserve,
            force=force,
        )
        return self._working
