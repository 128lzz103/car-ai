"""Provider 抽象层 —— 统一 chat 接口,支持 OpenAI 兼容与 Anthropic。

对标 Claude Code 的服务层 API 客户端(docs/13-services.md):
- `client.ts` 多 Provider 支持(Direct API / Bedrock / Vertex / Foundry)——这里抽象一层
  Provider,让 Agent Loop 与具体后端解耦,同时支持 OpenAI 兼容 API 与 Anthropic。
- `withRetry.ts` 指数退避重试 —— 统一策略见 `retry.py`。

## 归一化的内部消息格式(Agent 层使用)
- {"role": "user", "content": str}
- {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
- {"role": "tool", "tool_call_id": str, "name": str, "content": str}

各 Provider 负责把这套格式翻译成自己的 wire 格式,并把响应翻译回统一的 Reply。

## 为什么 llm 层最复杂(见 docs/02)
流式响应会把每个 tool_call 的 arguments 切成碎片,必须按 index 重新拼接;provider 偶尔
返回半个 JSON 或空 usage。429/超时/5xx 由 retry.py 统一分类并退避重试。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import httpx

from .models import get_model_profile
from .retry import (
    IncompleteStreamError,
    ProviderHTTPError,
    ProviderRetryExhaustedError,
    RetryCallback,
    RetryController,
    RetryPolicy,
    StreamInterruptedError,
    error_from_response,
    is_retryable_exception,
)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Reply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class ProviderCapabilities:
    tool_calls: bool = False
    forced_tool_choice: bool = False
    streaming: bool = False
    streaming_usage: bool = False
    parallel_tool_calls: bool = False
    context_window: int | None = None


class Provider(ABC):
    capabilities = ProviderCapabilities()

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        retry_policy: RetryPolicy | None = None,
        retry_controller: RetryController | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.retry = retry_controller or RetryController(retry_policy)

    @property
    def on_retry(self) -> RetryCallback | None:
        return self.retry.on_retry

    @on_retry.setter
    def on_retry(self, callback: RetryCallback | None) -> None:
        self.retry.on_retry = callback

    def capabilities_for(self, model: str | None = None) -> ProviderCapabilities:
        profile = get_model_profile(model or self.model)
        return replace(self.capabilities, context_window=profile.context_window)

    def _request_with_retry(self, request: Callable[[], httpx.Response]) -> httpx.Response:
        policy = self.retry.policy
        for attempt_index in range(policy.max_attempts):
            try:
                response = request()
            except (httpx.TimeoutException, httpx.TransportError) as error:
                failure: BaseException = error
            else:
                if response.status_code < 400:
                    return response
                failure = error_from_response(response)

            if not is_retryable_exception(failure):
                raise failure
            if attempt_index >= policy.max_retries:
                raise ProviderRetryExhaustedError(attempt_index + 1, failure) from failure
            retry_after = failure.retry_after if isinstance(failure, ProviderHTTPError) else None
            status_code = failure.status_code if isinstance(failure, ProviderHTTPError) else None
            self.retry.wait(
                attempt_index,
                str(failure),
                status_code=status_code,
                retry_after=retry_after,
            )
        raise AssertionError("unreachable")

    def _handle_stream_failure(
        self,
        error: BaseException,
        *,
        attempt_index: int,
        partial_text: str,
        output_emitted: bool,
    ) -> None:
        if output_emitted:
            raise StreamInterruptedError(partial_text, error) from error
        if not is_retryable_exception(error):
            raise error
        if attempt_index >= self.retry.policy.max_retries:
            raise ProviderRetryExhaustedError(attempt_index + 1, error) from error
        retry_after = error.retry_after if isinstance(error, ProviderHTTPError) else None
        status_code = error.status_code if isinstance(error, ProviderHTTPError) else None
        self.retry.wait(
            attempt_index,
            str(error),
            status_code=status_code,
            retry_after=retry_after,
        )

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
        model: str | None = None,
        stream: bool = True,
        on_text: Callable[[str], None] | None = None,
        tool_choice: str | None = None,
    ) -> Reply:
        """发起一次补全。stream=True 时,每收到一段文本就调用 on_text 回调。"""
        raise NotImplementedError


# ----------------------------------------------------------------------------
# OpenAI 兼容(OpenAI / DeepSeek / Ollama / Kimi / Qwen ...)
# ----------------------------------------------------------------------------
class OpenAIProvider(Provider):
    capabilities = ProviderCapabilities(
        tool_calls=True,
        forced_tool_choice=True,
        streaming=True,
        streaming_usage=True,
        parallel_tool_calls=True,
    )

    def _tools_payload(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

    def _messages_payload(
        self, messages: list[dict[str, Any]], system: str
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            role = m["role"]
            if role == "assistant" and m.get("tool_calls"):
                out.append(
                    {
                        "role": "assistant",
                        "content": m.get("content") or None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                                },
                            }
                            for tc in m["tool_calls"]
                        ],
                    }
                )
            elif role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m["tool_call_id"],
                        "content": m["content"],
                    }
                )
            else:
                out.append({"role": role, "content": m.get("content", "")})
        return out

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
        model: str | None = None,
        stream: bool = True,
        on_text: Callable[[str], None] | None = None,
        tool_choice: str | None = None,
    ) -> Reply:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": self._messages_payload(messages, system),
            "stream": stream,
        }
        if tools:
            payload["tools"] = self._tools_payload(tools)
        if tool_choice:
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice},
            }
        if stream:
            payload["stream_options"] = {"include_usage": True}

        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/chat/completions"

        if not stream:
            resp = self._request_with_retry(
                lambda: httpx.post(url, json=payload, headers=headers, timeout=300)
            )
            return self._parse_nonstream(resp.json())
        return self._parse_stream(url, payload, headers, on_text)

    def _parse_nonstream(self, data: dict[str, Any]) -> Reply:
        choice = data["choices"][0]["message"]
        reply = Reply(text=choice.get("content") or "")
        for tc in choice.get("tool_calls") or []:
            fn = tc["function"]
            reply.tool_calls.append(
                ToolCall(
                    id=tc["id"], name=fn["name"], arguments=_safe_json(fn.get("arguments", ""))
                )
            )
        usage = data.get("usage") or {}
        reply.input_tokens = usage.get("prompt_tokens", 0) or 0
        reply.output_tokens = usage.get("completion_tokens", 0) or 0
        return reply

    def _parse_stream(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        on_text: Callable[[str], None] | None,
    ) -> Reply:
        for attempt_index in range(self.retry.policy.max_attempts):
            reply = Reply()
            acc: dict[int, dict[str, str]] = {}
            complete = False
            try:
                with httpx.Client(timeout=300) as client:
                    with client.stream("POST", url, json=payload, headers=headers) as resp:
                        if resp.status_code >= 400:
                            raise error_from_response(resp)
                        for line in resp.iter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            data_str = line[len("data:") :].strip()
                            if data_str == "[DONE]":
                                complete = True
                                break
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            if self._consume_chunk(chunk, reply, acc, on_text):
                                complete = True
                                break
                if not complete:
                    raise IncompleteStreamError("OpenAI 流缺少 [DONE] 或 finish_reason")
            except (
                ProviderHTTPError,
                httpx.TimeoutException,
                httpx.TransportError,
                IncompleteStreamError,
            ) as error:
                self._handle_stream_failure(
                    error,
                    attempt_index=attempt_index,
                    partial_text=reply.text,
                    output_emitted=bool(reply.text and on_text),
                )
                continue

            for idx in sorted(acc):
                item = acc[idx]
                reply.tool_calls.append(
                    ToolCall(
                        id=item.get("id") or f"call_{idx}",
                        name=item.get("name", ""),
                        arguments=_safe_json(item.get("args", "")),
                    )
                )
            return reply
        raise AssertionError("unreachable")

    def _consume_chunk(
        self,
        chunk: dict[str, Any],
        reply: Reply,
        acc: dict[int, dict[str, str]],
        on_text: Callable[[str], None] | None,
    ) -> bool:
        usage = chunk.get("usage")
        if usage:
            reply.input_tokens = usage.get("prompt_tokens", 0) or reply.input_tokens
            reply.output_tokens = usage.get("completion_tokens", 0) or reply.output_tokens
        choices = chunk.get("choices") or []
        if not choices:
            return False
        choice = choices[0]
        delta = choice.get("delta") or {}
        if delta.get("content"):
            reply.text += delta["content"]
            if on_text:
                on_text(delta["content"])
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            slot = acc.setdefault(idx, {"id": "", "name": "", "args": ""})
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] += fn["name"]
            if fn.get("arguments"):
                slot["args"] += fn["arguments"]
        return choice.get("finish_reason") is not None


# ----------------------------------------------------------------------------
# Anthropic (Messages API)
# ----------------------------------------------------------------------------
class AnthropicProvider(Provider):
    capabilities = ProviderCapabilities(
        tool_calls=True,
        forced_tool_choice=True,
        streaming=True,
        streaming_usage=True,
        parallel_tool_calls=True,
    )

    _API_VERSION = "2023-06-01"
    _MAX_TOKENS = 8192

    def _tools_payload(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]

    def _messages_payload(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            role = m["role"]
            if role == "assistant" and m.get("tool_calls"):
                blocks: list[dict[str, Any]] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    blocks.append(
                        {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                    )
                out.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": m["content"],
                }
                # Anthropic 要求同一轮的多个 tool_result 放在一个 user content 中。
                previous_is_tool_results = (
                    out
                    and out[-1]["role"] == "user"
                    and isinstance(out[-1]["content"], list)
                    and all(item.get("type") == "tool_result" for item in out[-1]["content"])
                )
                if previous_is_tool_results:
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            else:
                out.append({"role": role, "content": m.get("content", "")})
        return out

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
        model: str | None = None,
        stream: bool = True,
        on_text: Callable[[str], None] | None = None,
        tool_choice: str | None = None,
    ) -> Reply:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "max_tokens": self._MAX_TOKENS,
            "system": system,
            "messages": self._messages_payload(messages),
            "stream": stream,
        }
        if tools:
            payload["tools"] = self._tools_payload(tools)
        if tool_choice:
            payload["tool_choice"] = {"type": "tool", "name": tool_choice}

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self._API_VERSION,
            "content-type": "application/json",
        }
        url = f"{self.base_url}/v1/messages"

        if not stream:
            resp = self._request_with_retry(
                lambda: httpx.post(url, json=payload, headers=headers, timeout=300)
            )
            return self._parse_nonstream(resp.json())
        return self._parse_stream(url, payload, headers, on_text)

    def _parse_nonstream(self, data: dict[str, Any]) -> Reply:
        reply = Reply()
        for block in data.get("content", []):
            if block["type"] == "text":
                reply.text += block["text"]
            elif block["type"] == "tool_use":
                reply.tool_calls.append(
                    ToolCall(id=block["id"], name=block["name"], arguments=block.get("input") or {})
                )
        usage = data.get("usage") or {}
        reply.input_tokens = usage.get("input_tokens", 0) or 0
        reply.output_tokens = usage.get("output_tokens", 0) or 0
        return reply

    def _parse_stream(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        on_text: Callable[[str], None] | None,
    ) -> Reply:
        for attempt_index in range(self.retry.policy.max_attempts):
            reply = Reply()
            blocks: dict[int, dict[str, Any]] = {}
            complete = False
            try:
                with httpx.Client(timeout=300) as client:
                    with client.stream("POST", url, json=payload, headers=headers) as resp:
                        if resp.status_code >= 400:
                            raise error_from_response(resp)
                        for line in resp.iter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            try:
                                event = json.loads(line[len("data:") :].strip())
                            except json.JSONDecodeError:
                                continue
                            if self._consume_event(event, reply, blocks, on_text):
                                complete = True
                                break
                if not complete:
                    raise IncompleteStreamError("Anthropic 流缺少 message_stop")
            except (
                ProviderHTTPError,
                httpx.TimeoutException,
                httpx.TransportError,
                IncompleteStreamError,
            ) as error:
                self._handle_stream_failure(
                    error,
                    attempt_index=attempt_index,
                    partial_text=reply.text,
                    output_emitted=bool(reply.text and on_text),
                )
                continue

            for idx in sorted(blocks):
                block = blocks[idx]
                if block.get("type") == "tool_use":
                    reply.tool_calls.append(
                        ToolCall(
                            id=block["id"],
                            name=block["name"],
                            arguments=_safe_json(block.get("json", "")),
                        )
                    )
            return reply
        raise AssertionError("unreachable")

    def _consume_event(
        self,
        event: dict[str, Any],
        reply: Reply,
        blocks: dict[int, dict[str, Any]],
        on_text: Callable[[str], None] | None,
    ) -> bool:
        etype = event.get("type")
        if etype == "content_block_start":
            idx = event["index"]
            cb = event["content_block"]
            if cb["type"] == "tool_use":
                blocks[idx] = {"type": "tool_use", "id": cb["id"], "name": cb["name"], "json": ""}
            else:
                blocks[idx] = {"type": "text"}
        elif etype == "content_block_delta":
            idx = event["index"]
            delta = event["delta"]
            if delta["type"] == "text_delta":
                reply.text += delta["text"]
                if on_text:
                    on_text(delta["text"])
            elif delta["type"] == "input_json_delta":
                blocks.setdefault(idx, {"type": "tool_use", "json": ""})
                blocks[idx]["json"] += delta.get("partial_json", "")
        elif etype == "message_start":
            usage = (event.get("message") or {}).get("usage") or {}
            reply.input_tokens = usage.get("input_tokens", 0) or reply.input_tokens
        elif etype == "message_delta":
            usage = event.get("usage") or {}
            reply.output_tokens = usage.get("output_tokens", 0) or reply.output_tokens
        return etype == "message_stop"


def _safe_json(s: str) -> dict[str, Any]:
    """容错解析 arguments;空串或半个 JSON 都退化为 {}。"""
    s = (s or "").strip()
    if not s:
        return {}
    try:
        result = json.loads(s)
        return result if isinstance(result, dict) else {"value": result}
    except json.JSONDecodeError:
        return {}


def get_provider(config: Any) -> Provider:
    """工厂:按 config.provider 选择实现。"""
    retry_policy = RetryPolicy(
        max_retries=getattr(config, "max_retries", 3),
        base_delay=getattr(config, "retry_base_delay", 1.0),
        max_delay=getattr(config, "retry_max_delay", 8.0),
        max_retry_after=getattr(config, "retry_max_wait", 60.0),
    )
    if config.provider == "anthropic":
        return AnthropicProvider(
            config.api_key,
            config.base_url,
            config.model,
            retry_policy=retry_policy,
        )
    return OpenAIProvider(
        config.api_key,
        config.base_url,
        config.model,
        retry_policy=retry_policy,
    )
