"""上下文压缩单测:token 估算、tier1 裁剪、tier2/3 摘要(摘要用假 provider)。"""

from __future__ import annotations

import pytest

from minicoder.context import (
    TOOL_OUTPUT_MAX_CHARS,
    ContextManager,
    ContextOverflowError,
    ContextProtocolError,
    compact_if_needed,
    count_tokens,
    group_messages,
    total_tokens,
    validate_tool_protocol,
)
from minicoder.providers import ToolCall


class FakeProvider:
    """摘要用的假 provider,永远返回固定摘要文本。"""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools, system, model=None, stream=True, on_text=None):
        self.calls += 1
        from minicoder.providers import Reply

        return Reply(text="SUMMARY")


def test_count_tokens():
    assert count_tokens("") == 1
    assert count_tokens("a" * 400) == 100
    assert count_tokens("中文测试") == 4
    assert count_tokens("中文测试", "gpt-4o") == 3
    assert count_tokens("123456") == 2


def test_no_compact_when_small():
    msgs = [{"role": "user", "content": "hi"}]
    provider = FakeProvider()
    out = compact_if_needed(msgs, provider, "m", context_window=100_000)
    assert out is msgs
    assert provider.calls == 0


def test_tier1_trims_tool_output():
    # 超长工具输出位于较旧的完整交换组，近期用户消息保持原样。
    big = "x" * (TOOL_OUTPUT_MAX_CHARS * 3)
    msgs = [
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [ToolCall(id="1", name="read_file", arguments={"path": "a"})],
        },
        {"role": "tool", "tool_call_id": "1", "name": "read_file", "content": big},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "next"},
    ]
    provider = FakeProvider()
    # 窗口设为刚好让 ratio 落在 tier1~tier2 之间
    window = int(total_tokens(msgs) / 0.55)
    out = compact_if_needed(msgs, provider, "m", context_window=window)
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert "已裁剪" in tool_msgs[0]["content"]
    assert provider.calls == 0  # tier1 不调 LLM


def test_tier2_summarizes():
    # 很多完整 Turn + 小窗口 → 触发摘要
    msgs = []
    for index in range(20):
        msgs.extend(
            [
                {"role": "user", "content": "original task" if index == 0 else f"next {index}"},
                {"role": "assistant", "content": "step " * 50},
            ]
        )
    provider = FakeProvider()
    window = int(total_tokens(msgs) / 0.75)  # ratio ≈ 0.75,落在 tier2
    out = compact_if_needed(msgs, provider, "m", context_window=window)
    assert provider.calls == 1
    assert out[:2] == msgs[:2]  # 首个完整 Turn 保留
    assert any("摘要" in m.get("content", "") for m in out)
    assert len(out) < len(msgs)


def test_tool_exchange_is_an_atomic_group():
    calls = [ToolCall(id=str(i), name="read_file", arguments={"path": str(i)}) for i in range(7)]
    msgs = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": calls},
        *[
            {"role": "tool", "tool_call_id": str(i), "name": "read_file", "content": "ok"}
            for i in range(7)
        ],
    ]
    groups = group_messages(msgs)
    assert len(groups) == 1
    assert groups[0].kind == "turn"
    assert len(groups[0].messages) == 9
    validate_tool_protocol(msgs)


def test_orphan_tool_result_rejected():
    msgs = [{"role": "tool", "tool_call_id": "1", "content": "orphan"}]
    with pytest.raises(ContextProtocolError, match="孤立"):
        validate_tool_protocol(msgs)


def test_missing_tool_result_rejected():
    msgs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [ToolCall(id="1", name="read_file", arguments={})],
        }
    ]
    with pytest.raises(ContextProtocolError, match="不完整对应"):
        validate_tool_protocol(msgs)


def test_compaction_keeps_parallel_results_with_their_call():
    calls = [ToolCall(id=str(i), name="read_file", arguments={}) for i in range(7)]
    msgs = []
    for index in range(8):
        msgs.extend(
            [
                {"role": "user", "content": "original task" if index == 0 else f"next {index}"},
                {"role": "assistant", "content": "old " * 100},
            ]
        )
    msgs.extend(
        [
            {"role": "user", "content": "read in parallel"},
            {"role": "assistant", "content": "", "tool_calls": calls},
        ]
    )
    msgs += [
        {"role": "tool", "tool_call_id": str(i), "name": "read_file", "content": "ok"}
        for i in range(7)
    ]
    provider = FakeProvider()
    window = int(total_tokens(msgs) / 0.75)
    out = compact_if_needed(msgs, provider, "m", context_window=window)
    validate_tool_protocol(out)
    turn = next(
        group
        for group in group_messages(out)
        if any(message.get("tool_calls") for message in group.messages)
    )
    assert len(turn.messages) == 9


def test_context_manager_preserves_full_transcript():
    msgs = []
    for index in range(8):
        msgs.extend(
            [
                {"role": "user", "content": "original task" if index == 0 else f"next {index}"},
                {"role": "assistant", "content": "old " * 30},
            ]
        )
    manager = ContextManager(msgs)
    original = list(manager.transcript)
    manager.prepare(
        FakeProvider(),
        "m",
        context_window=100_000,
        output_reserve=0,
        force=True,
    )
    assert manager.transcript == original
    assert len(manager.working_messages) < len(manager.transcript)
    assert any(message.get("_context_summary") for message in manager.working_messages)


def test_summary_failure_falls_back_when_still_within_budget():
    class FailingProvider:
        def chat(self, *args, **kwargs):
            raise RuntimeError("summary unavailable")

    msgs = []
    for index in range(6):
        msgs.extend(
            [
                {"role": "user", "content": f"task {index}"},
                {"role": "assistant", "content": "step"},
            ]
        )
    notices = []
    out = compact_if_needed(
        msgs,
        FailingProvider(),
        "m",
        context_window=100_000,
        on_notice=notices.append,
        force=True,
    )
    assert out == msgs
    assert any("摘要失败" in notice for notice in notices)


def test_uncompressible_context_raises_clear_error():
    msgs = [{"role": "user", "content": "中" * 500}]
    with pytest.raises(ContextOverflowError, match="仍超出输入预算"):
        compact_if_needed(
            msgs,
            FakeProvider(),
            "m",
            context_window=100,
            output_reserve=0,
            safety_margin=0,
        )


def test_multiple_tool_rounds_stay_in_one_complete_turn():
    messages = [
        {"role": "user", "content": "do work"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [ToolCall(id="1", name="read_file", arguments={})],
        },
        {"role": "tool", "tool_call_id": "1", "content": "read"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [ToolCall(id="2", name="write_file", arguments={})],
        },
        {"role": "tool", "tool_call_id": "2", "content": "written"},
        {"role": "assistant", "content": "done"},
    ]

    groups = group_messages(messages)

    assert len(groups) == 1
    assert groups[0].kind == "turn"
    assert list(groups[0].messages) == messages
