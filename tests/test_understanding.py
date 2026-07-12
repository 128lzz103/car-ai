"""结构化理解编排、规则降级和槽位状态测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from minicoder.agent import Agent
from minicoder.intent import IntentName, IntentStatus
from minicoder.providers import ProviderCapabilities, Reply, ToolCall
from minicoder.understanding import VehicleIntentInterpreter

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=timezone(timedelta(hours=8)))


def _payload(intent, entities=None, evidence=None, confidence="high", ambiguities=None):
    return {
        "intent": intent,
        "entities": entities or {},
        "evidence": evidence or {},
        "confidence": confidence,
        "ambiguities": ambiguities or [],
    }


def _reply(payload, *, input_tokens=10, output_tokens=3):
    return Reply(
        tool_calls=[ToolCall("intent-1", "emit_vehicle_intent", payload)],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class ScriptedProvider:
    capabilities = ProviderCapabilities(tool_calls=True, forced_tool_choice=True, streaming=True)

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def _interpreter(provider, **kwargs):
    return VehicleIntentInterpreter(
        provider,
        model="intent-model",
        now=lambda: NOW,
        **kwargs,
    )


def test_llm_extraction_forces_structured_tool_and_builds_plan():
    payload = _payload(
        "trip_charge_planning",
        {"vehicle_id": "A102", "destination": "南京南站", "departure_time": "明早八点"},
        {"vehicle_id": "A102", "destination": "南京南站", "departure_time": "明早八点"},
    )
    provider = ScriptedProvider([_reply(payload)])

    result = _interpreter(provider, rule_fast_path=False).interpret(
        "A102 明早八点去南京南站，需要充电吗"
    )

    assert result.result.status is IntentStatus.READY
    assert result.plan.status == "ready"
    assert result.input_tokens == 10
    assert provider.calls[0]["tool_choice"] == "emit_vehicle_intent"
    assert provider.calls[0]["stream"] is False
    assert provider.calls[0]["model"] == "intent-model"


def test_high_confidence_rule_path_avoids_llm_call():
    provider = ScriptedProvider([])
    interpretation = _interpreter(provider).interpret("查询 A102 的剩余续航")

    assert interpretation.source == "rule"
    assert interpretation.result.intent is IntentName.RANGE_QUERY
    assert interpretation.result.status is IntentStatus.READY
    assert provider.calls == []


def test_complex_trip_request_takes_priority_over_range_query():
    provider = ScriptedProvider([])

    interpretation = _interpreter(provider).interpret(
        "查询 A102 当前续航，明早八点去上海虹桥站，判断是否需要充电"
    )

    assert interpretation.source == "rule"
    assert interpretation.result.intent is IntentName.TRIP_CHARGE_PLANNING
    assert interpretation.result.status is IntentStatus.READY
    assert interpretation.result.entities == {
        "vehicle_id": "A102",
        "destination": "上海虹桥站",
        "departure_time": "2026-08-04T08:00:00+08:00",
        "departure_timezone": "Asia/Shanghai",
    }
    assert provider.calls == []


def test_missing_slot_is_filled_on_followup_without_another_llm_call():
    provider = ScriptedProvider([_reply(_payload("range_query"))])
    interpreter = _interpreter(provider)

    first = interpreter.interpret("查询续航")
    second = interpreter.interpret("A102")

    assert first.result.missing_fields == ("vehicle_id",)
    assert first.clarification == "请补充车辆 ID。"
    assert second.result.status is IntentStatus.READY
    assert second.result.entities["vehicle_id"] == "A102"
    assert len(provider.calls) == 1


def test_slot_conflict_requires_explicit_confirmation():
    provider = ScriptedProvider(
        [
            _reply(
                _payload(
                    "climate_control",
                    {"vehicle_id": "A102"},
                    {"vehicle_id": "A102"},
                )
            ),
            _reply(
                _payload(
                    "climate_control",
                    {"vehicle_id": "A103", "climate_action": "turn_off"},
                    {"vehicle_id": "A103", "climate_action": "关闭空调"},
                )
            ),
        ]
    )
    interpreter = _interpreter(provider, rule_fast_path=False)

    first = interpreter.interpret("操作 A102 的空调")
    second = interpreter.interpret("关闭 A103 的空调")

    assert first.result.missing_fields == ("climate_action",)
    assert second.result.status is IntentStatus.NEEDS_CLARIFICATION
    assert "slot_conflict:vehicle_id" in second.result.reason_codes
    assert "车辆 ID" in second.clarification


def test_explicit_correction_overrides_previous_slot():
    provider = ScriptedProvider(
        [
            _reply(
                _payload(
                    "climate_control",
                    {"vehicle_id": "A102"},
                    {"vehicle_id": "A102"},
                )
            ),
            _reply(
                _payload(
                    "climate_control",
                    {"vehicle_id": "A103", "climate_action": "turn_off"},
                    {"vehicle_id": "A103", "climate_action": "关闭空调"},
                )
            ),
        ]
    )
    interpreter = _interpreter(provider, rule_fast_path=False)
    interpreter.interpret("操作 A102 的空调")

    corrected = interpreter.interpret("不是 A102，是 A103，关闭空调")

    assert corrected.result.status is IntentStatus.READY
    assert corrected.result.entities["vehicle_id"] == "A103"


def test_provider_failure_uses_safe_rule_fallback():
    provider = ScriptedProvider([RuntimeError("provider down")])
    interpretation = _interpreter(provider, rule_fast_path=False).interpret("查询 A102 的电量")

    assert interpretation.source == "fallback"
    assert interpretation.result.intent is IntentName.VEHICLE_STATUS_QUERY
    assert interpretation.result.status is IntentStatus.READY


def test_provider_failure_without_rule_does_not_create_plan():
    provider = ScriptedProvider([RuntimeError("provider down")])
    interpretation = _interpreter(provider, rule_fast_path=False).interpret("帮我处理一下")

    assert interpretation.result.intent is IntentName.UNKNOWN
    assert interpretation.result.status is IntentStatus.NEEDS_CLARIFICATION
    assert interpretation.plan.status == "blocked"
    assert "extractor_unavailable" in interpretation.result.reason_codes


def test_cancel_clears_pending_state():
    provider = ScriptedProvider([_reply(_payload("range_query"))])
    interpreter = _interpreter(provider)
    interpreter.interpret("查询续航")

    cancelled = interpreter.interpret("取消")

    assert cancelled.result.status is IntentStatus.CANCELLED
    assert interpreter.pending is None


def test_agent_returns_clarification_without_entering_main_loop():
    provider = ScriptedProvider([_reply(_payload("range_query"))])
    interpreter = _interpreter(provider)
    config = SimpleNamespace(model="main", max_rounds=3, context_window=1000, output_reserve=100)
    agent = Agent(provider, config, [], "system", stream=False, interpreter=interpreter)

    response = agent.chat("查询续航")

    assert response == "请补充车辆 ID。"
    assert len(provider.calls) == 1
    assert [message["role"] for message in agent.transcript] == ["user", "assistant"]
    assert agent.transcript[-1]["content"] == response


def test_agent_injects_ready_interpretation_into_main_system():
    provider = ScriptedProvider(
        [
            _reply(
                _payload(
                    "range_query",
                    {"vehicle_id": "A102"},
                    {"vehicle_id": "A102"},
                )
            ),
            Reply(text="尚未接入车辆 API"),
        ]
    )
    interpreter = _interpreter(provider, rule_fast_path=False)
    config = SimpleNamespace(model="main", max_rounds=3, context_window=1000, output_reserve=100)
    agent = Agent(provider, config, [], "system", stream=False, interpreter=interpreter)

    assert agent.chat("查询 A102 的续航") == "尚未接入车辆 API"
    assert len(provider.calls) == 2
    assert "当前请求的已验证结构化理解" in provider.calls[1]["system"]
    assert '"range_query"' in provider.calls[1]["system"]


def test_plain_json_reply_is_accepted_when_tool_call_is_missing():
    payload = _payload(
        "navigation_request",
        {"vehicle_id": "A102", "destination": "南京南站"},
        {"vehicle_id": "A102", "destination": "南京南站"},
    )
    provider = ScriptedProvider([Reply(text=__import__("json").dumps(payload, ensure_ascii=False))])

    interpretation = _interpreter(provider, rule_fast_path=False).interpret(
        "让 A102 导航到南京南站"
    )

    assert interpretation.result.status is IntentStatus.READY
    assert interpretation.result.entities["destination"] == "南京南站"


def test_new_intent_replaces_pending_intent():
    provider = ScriptedProvider(
        [
            _reply(_payload("range_query")),
            _reply(
                _payload(
                    "navigation_request",
                    {"vehicle_id": "A102", "destination": "南京南站"},
                    {"vehicle_id": "A102", "destination": "南京南站"},
                )
            ),
        ]
    )
    interpreter = _interpreter(provider, rule_fast_path=False)
    interpreter.interpret("查询续航")

    navigation = interpreter.interpret("改为让 A102 导航到南京南站")

    assert navigation.result.intent is IntentName.NAVIGATION_REQUEST
    assert navigation.result.status is IntentStatus.READY
    assert interpreter.pending is None


def test_pending_state_expires_after_three_unrecognized_followups():
    provider = ScriptedProvider(
        [
            _reply(_payload("range_query")),
            Reply(text="not-json"),
            Reply(text="not-json"),
            Reply(text="not-json"),
        ]
    )
    interpreter = _interpreter(provider, rule_fast_path=False)
    interpreter.interpret("查询续航")

    first = interpreter.interpret("我不清楚")
    second = interpreter.interpret("还是不清楚")
    expired = interpreter.interpret("确实不知道")

    assert first.result.reason_codes == ("followup_unrecognized",)
    assert second.result.reason_codes == ("followup_unrecognized",)
    assert expired.result.reason_codes == ("pending_expired",)
    assert interpreter.pending is None


def test_third_followup_can_still_complete_pending_state():
    provider = ScriptedProvider(
        [_reply(_payload("range_query")), Reply(text="bad"), Reply(text="bad")]
    )
    interpreter = _interpreter(provider, rule_fast_path=False)
    interpreter.interpret("查询续航")
    interpreter.interpret("不知道")
    interpreter.interpret("再想想")

    completed = interpreter.interpret("A102")

    assert completed.result.status is IntentStatus.READY
    assert completed.result.entities["vehicle_id"] == "A102"
    assert interpreter.pending is None


def test_pending_fast_paths_cover_temperature_destination_and_location():
    temperature_provider = ScriptedProvider(
        [
            _reply(
                _payload(
                    "climate_control",
                    {"vehicle_id": "A102", "climate_action": "set_temperature"},
                    {"vehicle_id": "A102", "climate_action": "设置"},
                )
            )
        ]
    )
    temperature = _interpreter(temperature_provider)
    temperature.interpret("设置 A102 空调温度")
    assert temperature.interpret("22度").result.entities["target_temperature"] == 22

    destination_provider = ScriptedProvider(
        [
            _reply(
                _payload(
                    "navigation_request",
                    {"vehicle_id": "A102"},
                    {"vehicle_id": "A102"},
                )
            )
        ]
    )
    destination = _interpreter(destination_provider)
    destination.interpret("让 A102 开始导航")
    assert destination.interpret("南京南站").result.entities["destination"] == "南京南站"

    location_provider = ScriptedProvider([_reply(_payload("charging_station_query"))])
    location = _interpreter(location_provider)
    location.interpret("查询充电站")
    assert location.interpret("新街口").result.entities["location"] == "新街口"


def test_rule_paths_cover_climate_and_route_fallbacks():
    climate_provider = ScriptedProvider([])
    climate = _interpreter(climate_provider).interpret("把 A102 的空调调到 22 度")
    assert climate.result.entities["climate_action"] == "set_temperature"
    assert climate.result.entities["target_temperature"] == 22

    trip_provider = ScriptedProvider([RuntimeError("down")])
    trip = _interpreter(trip_provider, rule_fast_path=False).interpret(
        "A102 去南京南站是否需要充电"
    )
    assert trip.result.intent is IntentName.TRIP_CHARGE_PLANNING

    navigation_provider = ScriptedProvider([RuntimeError("down")])
    navigation = _interpreter(navigation_provider, rule_fast_path=False).interpret(
        "导航到南京南站怎么走"
    )
    assert navigation.result.intent is IntentName.NAVIGATION_REQUEST
