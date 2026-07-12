"""Provider 协议转换、非流式重试和流式恢复单测。"""

from __future__ import annotations

import json

import httpx
import pytest

from minicoder.providers import AnthropicProvider, OpenAIProvider, ToolCall
from minicoder.retry import (
    ProviderAuthenticationError,
    ProviderRetryExhaustedError,
    RetryController,
    RetryPolicy,
    StreamInterruptedError,
)


def _messages():
    return [
        {"role": "user", "content": "read files"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                ToolCall(id="1", name="read_file", arguments={"path": "a.py"}),
                ToolCall(id="2", name="read_file", arguments={"path": "b.py"}),
            ],
        },
        {"role": "tool", "tool_call_id": "1", "name": "read_file", "content": "a"},
        {"role": "tool", "tool_call_id": "2", "name": "read_file", "content": "b"},
    ]


def test_anthropic_groups_parallel_tool_results():
    provider = AnthropicProvider("key", "https://example.com", "model")
    payload = provider._messages_payload(_messages())
    result_message = payload[-1]
    assert result_message["role"] == "user"
    assert [block["tool_use_id"] for block in result_message["content"]] == ["1", "2"]


def test_openai_keeps_one_message_per_tool_result():
    provider = OpenAIProvider("key", "https://example.com", "model")
    payload = provider._messages_payload(_messages(), "system")
    tool_messages = [message for message in payload if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == ["1", "2"]


def test_provider_capabilities_are_explicit_and_model_aware():
    openai = OpenAIProvider("key", "https://example.com", "gpt-4.1")
    anthropic = AnthropicProvider("key", "https://example.com", "claude-sonnet-4-5")

    assert openai.capabilities.tool_calls is True
    assert openai.capabilities.forced_tool_choice is True
    assert openai.capabilities.streaming_usage is True
    assert openai.capabilities.parallel_tool_calls is True
    assert openai.capabilities_for().context_window == 1_047_576
    assert anthropic.capabilities.tool_calls is True
    assert anthropic.capabilities.forced_tool_choice is True
    assert anthropic.capabilities_for().context_window == 200_000


def test_openai_forced_tool_choice_payload(monkeypatch):
    captured = {}

    def post(*_args, **kwargs):
        captured.update(kwargs["json"])
        return _openai_success()

    monkeypatch.setattr("minicoder.providers.httpx.post", post)
    provider = OpenAIProvider("key", "https://example.com", "model")
    tool = {"name": "emit", "description": "emit", "parameters": {"type": "object"}}

    provider.chat([], [tool], "system", stream=False, tool_choice="emit")

    assert captured["tool_choice"] == {"type": "function", "function": {"name": "emit"}}


def test_anthropic_forced_tool_choice_payload(monkeypatch):
    captured = {}

    def post(*_args, **kwargs):
        captured.update(kwargs["json"])
        return httpx.Response(200, json={"content": [], "usage": {}})

    monkeypatch.setattr("minicoder.providers.httpx.post", post)
    provider = AnthropicProvider("key", "https://example.com", "model")
    tool = {"name": "emit", "description": "emit", "parameters": {"type": "object"}}

    provider.chat([], [tool], "system", stream=False, tool_choice="emit")

    assert captured["tool_choice"] == {"type": "tool", "name": "emit"}


def _controller(*, max_retries=2):
    sleeps = []
    controller = RetryController(
        RetryPolicy(max_retries=max_retries, jitter=0),
        sleep=sleeps.append,
        random_value=lambda: 0.5,
    )
    return controller, sleeps


def _openai_success(text="ok"):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        },
    )


def test_nonstream_retries_429_and_respects_retry_after(monkeypatch):
    controller, sleeps = _controller()
    events = []
    provider = OpenAIProvider("key", "https://example.com", "model", retry_controller=controller)
    provider.on_retry = events.append
    responses = [
        httpx.Response(429, text="limited", headers={"Retry-After": "3"}),
        _openai_success("done"),
    ]
    monkeypatch.setattr("minicoder.providers.httpx.post", lambda *args, **kwargs: responses.pop(0))

    reply = provider.chat([], [], "system", stream=False)
    assert reply.text == "done"
    assert sleeps == [3]
    assert events[0].attempt == 2
    assert events[0].status_code == 429


def test_nonstream_authentication_error_is_not_retried(monkeypatch):
    controller, sleeps = _controller()
    provider = OpenAIProvider("key", "https://example.com", "model", retry_controller=controller)
    calls = 0

    def unauthorized(*args, **kwargs):
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="bad key")

    monkeypatch.setattr("minicoder.providers.httpx.post", unauthorized)
    with pytest.raises(ProviderAuthenticationError):
        provider.chat([], [], "system", stream=False)
    assert calls == 1
    assert sleeps == []


def test_nonstream_timeout_exhaustion(monkeypatch):
    controller, sleeps = _controller(max_retries=2)
    provider = OpenAIProvider("key", "https://example.com", "model", retry_controller=controller)
    calls = 0

    def timeout(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise httpx.ConnectTimeout("timeout")

    monkeypatch.setattr("minicoder.providers.httpx.post", timeout)
    with pytest.raises(ProviderRetryExhaustedError) as error:
        provider.chat([], [], "system", stream=False)
    assert error.value.attempts == 3
    assert calls == 3
    assert sleeps == [1, 2]


class FakeStreamResponse:
    def __init__(self, lines, *, status_code=200, text="", headers=None):
        self.lines = lines
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_lines(self):
        for line in self.lines:
            if isinstance(line, BaseException):
                raise line
            yield line


class FakeClient:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def stream(self, *args, **kwargs):
        return self.response


class UnreadErrorResponse:
    status_code = 429
    headers = {"Retry-After": "2"}

    @property
    def text(self):
        raise httpx.ResponseNotRead()

    def read(self):
        return b"stream limited"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_lines(self):
        return iter(())


def _install_streams(monkeypatch, responses):
    attempts = []

    def factory(*args, **kwargs):
        attempts.append(1)
        return FakeClient(responses.pop(0))

    monkeypatch.setattr("minicoder.providers.httpx.Client", factory)
    return attempts


def _openai_line(delta, finish_reason=None):
    data = {"choices": [{"delta": delta, "finish_reason": finish_reason}]}
    return "data: " + json.dumps(data)


def test_openai_stream_retries_before_visible_output(monkeypatch):
    controller, sleeps = _controller()
    provider = OpenAIProvider("key", "https://example.com", "model", retry_controller=controller)
    responses = [
        FakeStreamResponse([httpx.ReadError("disconnect")]),
        FakeStreamResponse([_openai_line({"content": "ok"}), "data: [DONE]"]),
    ]
    attempts = _install_streams(monkeypatch, responses)
    output = []
    reply = provider.chat([], [], "system", stream=True, on_text=output.append)
    assert reply.text == "ok"
    assert output == ["ok"]
    assert len(attempts) == 2
    assert sleeps == [1]


def test_stream_http_error_body_is_read_before_retry(monkeypatch):
    controller, sleeps = _controller()
    provider = OpenAIProvider("key", "https://example.com", "model", retry_controller=controller)
    responses = [
        UnreadErrorResponse(),
        FakeStreamResponse([_openai_line({"content": "ok"}, "stop")]),
    ]
    attempts = _install_streams(monkeypatch, responses)
    reply = provider.chat([], [], "system", stream=True)
    assert reply.text == "ok"
    assert len(attempts) == 2
    assert sleeps == [2]


def test_finish_reason_stops_before_later_disconnect(monkeypatch):
    controller, sleeps = _controller()
    provider = OpenAIProvider("key", "https://example.com", "model", retry_controller=controller)
    responses = [
        FakeStreamResponse([_openai_line({"content": "complete"}, "stop"), httpx.ReadError("late")])
    ]
    attempts = _install_streams(monkeypatch, responses)
    output = []
    reply = provider.chat([], [], "system", stream=True, on_text=output.append)
    assert reply.text == "complete"
    assert output == ["complete"]
    assert len(attempts) == 1
    assert sleeps == []


def test_openai_stream_does_not_retry_after_visible_output(monkeypatch):
    controller, sleeps = _controller()
    provider = OpenAIProvider("key", "https://example.com", "model", retry_controller=controller)
    responses = [
        FakeStreamResponse([_openai_line({"content": "partial"}), httpx.ReadError("disconnect")])
    ]
    attempts = _install_streams(monkeypatch, responses)
    output = []
    with pytest.raises(StreamInterruptedError) as error:
        provider.chat([], [], "system", stream=True, on_text=output.append)
    assert error.value.partial_text == "partial"
    assert output == ["partial"]
    assert len(attempts) == 1
    assert sleeps == []


def test_partial_tool_json_is_discarded_before_retry(monkeypatch):
    controller, _sleeps = _controller()
    provider = OpenAIProvider("key", "https://example.com", "model", retry_controller=controller)
    partial = {
        "tool_calls": [
            {
                "index": 0,
                "id": "old",
                "function": {"name": "read_file", "arguments": '{"path":'},
            }
        ]
    }
    complete = {
        "tool_calls": [
            {
                "index": 0,
                "id": "new",
                "function": {"name": "read_file", "arguments": '{"path":"b.py"}'},
            }
        ]
    }
    responses = [
        FakeStreamResponse([_openai_line(partial), httpx.ReadError("disconnect")]),
        FakeStreamResponse([_openai_line(complete, "tool_calls")]),
    ]
    _install_streams(monkeypatch, responses)
    reply = provider.chat([], [], "system", stream=True)
    assert len(reply.tool_calls) == 1
    assert reply.tool_calls[0].id == "new"
    assert reply.tool_calls[0].arguments == {"path": "b.py"}


def test_incomplete_stream_retries_then_exhausts(monkeypatch):
    controller, sleeps = _controller(max_retries=1)
    provider = OpenAIProvider("key", "https://example.com", "model", retry_controller=controller)
    responses = [FakeStreamResponse([]), FakeStreamResponse([])]
    attempts = _install_streams(monkeypatch, responses)
    with pytest.raises(ProviderRetryExhaustedError):
        provider.chat([], [], "system", stream=True)
    assert len(attempts) == 2
    assert sleeps == [1]


def test_anthropic_stream_retries_before_output(monkeypatch):
    controller, sleeps = _controller()
    provider = AnthropicProvider("key", "https://example.com", "model", retry_controller=controller)
    event = lambda value: "data: " + json.dumps(value)  # noqa: E731
    responses = [
        FakeStreamResponse([httpx.ReadError("disconnect")]),
        FakeStreamResponse(
            [
                event(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }
                ),
                event(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "ok"},
                    }
                ),
                event({"type": "message_stop"}),
            ]
        ),
    ]
    _install_streams(monkeypatch, responses)
    output = []
    reply = provider.chat([], [], "system", stream=True, on_text=output.append)
    assert reply.text == "ok"
    assert output == ["ok"]
    assert sleeps == [1]
