"""汽车意图协议、时间规范化和规则 Planner 测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from minicoder.intent import (
    Confidence,
    IntentStatus,
    TimeNormalizationError,
    normalize_departure_time,
    validate_intent_payload,
)
from minicoder.planner import RuleBasedPlanner

CHINA = timezone(timedelta(hours=8))
NOW = datetime(2026, 8, 3, 12, 0, tzinfo=CHINA)


def _payload(intent: str, entities=None, evidence=None, confidence="high", ambiguities=None):
    return {
        "intent": intent,
        "entities": entities or {},
        "evidence": evidence or {},
        "confidence": confidence,
        "ambiguities": ambiguities or [],
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("明早八点", "2026-08-04T08:00:00+08:00"),
        ("后天下午三点半", "2026-08-05T15:30:00+08:00"),
        ("2026-08-06 09:15", "2026-08-06T09:15:00+08:00"),
        ("8月5日上午10点", "2026-08-05T10:00:00+08:00"),
    ],
)
def test_normalize_departure_time(raw, expected):
    assert normalize_departure_time(raw, now=NOW) == expected


@pytest.mark.parametrize(
    ("raw", "reason"),
    [("八点", "missing_date"), ("今天上午十点", "time_in_past"), ("明天", "missing_time")],
)
def test_ambiguous_or_invalid_time_requires_clarification(raw, reason):
    with pytest.raises(TimeNormalizationError, match=reason):
        normalize_departure_time(raw, now=NOW)


def test_trip_intent_is_normalized_and_ready():
    text = "帮我看看 A102 明早八点去南京南站是否需要充电"
    result = validate_intent_payload(
        _payload(
            "trip_charge_planning",
            {
                "vehicle_id": "A102",
                "destination": "南京南站",
                "departure_time": "明早八点",
            },
            {
                "vehicle_id": "A102",
                "destination": "南京南站",
                "departure_time": "明早八点",
            },
        ),
        text,
        now=NOW,
    )

    assert result.status is IntentStatus.READY
    assert result.entities["vehicle_id"] == "A102"
    assert result.entities["departure_time"] == "2026-08-04T08:00:00+08:00"
    assert result.entities["departure_timezone"] == "Asia/Shanghai"


def test_entity_without_source_evidence_is_not_trusted():
    result = validate_intent_payload(
        _payload(
            "range_query",
            {"vehicle_id": "A102"},
            {"vehicle_id": "B999"},
        ),
        "查一下续航",
        now=NOW,
    )

    assert result.status is IntentStatus.NEEDS_CLARIFICATION
    assert result.entities == {}
    assert result.missing_fields == ("vehicle_id",)
    assert "missing_evidence:vehicle_id" in result.reason_codes


def test_climate_temperature_is_conditionally_required_and_bounded():
    missing = validate_intent_payload(
        _payload(
            "climate_control",
            {"vehicle_id": "A102", "climate_action": "set_temperature"},
            {"vehicle_id": "A102", "climate_action": "调到"},
        ),
        "把 A102 空调调到合适温度",
        now=NOW,
    )
    invalid = validate_intent_payload(
        _payload(
            "climate_control",
            {
                "vehicle_id": "A102",
                "climate_action": "set_temperature",
                "target_temperature": 35,
            },
            {"vehicle_id": "A102", "climate_action": "调到", "target_temperature": "35"},
        ),
        "把 A102 空调调到 35 度",
        now=NOW,
    )

    assert "target_temperature" in missing.missing_fields
    assert "target_temperature" in invalid.missing_fields
    assert "invalid_target_temperature" in invalid.reason_codes


def test_charging_station_requires_location_or_vehicle():
    result = validate_intent_payload(_payload("charging_station_query"), "找充电站", now=NOW)
    assert result.missing_fields == ("location_or_vehicle_id",)


def test_other_intent_is_out_of_scope():
    result = validate_intent_payload(_payload("other"), "写一首诗", now=NOW)
    assert result.status is IntentStatus.OUT_OF_SCOPE
    assert result.confidence is Confidence.HIGH


def test_rule_planner_only_accepts_ready_intents():
    ready = validate_intent_payload(
        _payload(
            "trip_charge_planning",
            {"vehicle_id": "A102", "destination": "南京南站"},
            {"vehicle_id": "A102", "destination": "南京南站"},
        ),
        "A102 去南京南站需要充电吗",
        now=NOW,
    )
    blocked = validate_intent_payload(_payload("range_query"), "查询续航", now=NOW)

    plan = RuleBasedPlanner().create_plan(ready)
    assert plan.status == "ready"
    assert [step.action for step in plan.steps] == [
        "get_vehicle_status",
        "estimate_route",
        "calculate_energy_requirement",
        "get_charging_stations",
        "build_charging_recommendation",
        "search_vehicle_knowledge",
    ]
    assert plan.steps[2].depends_on == ("vehicle_status", "route")
    assert RuleBasedPlanner().create_plan(blocked).status == "blocked"


@pytest.mark.parametrize(
    ("intent", "text", "entities", "evidence", "action"),
    [
        (
            "vehicle_status_query",
            "查询 A102 的车辆状态",
            {"vehicle_id": "A102"},
            {"vehicle_id": "A102"},
            "get_vehicle_status",
        ),
        (
            "climate_control",
            "关闭 A102 的空调",
            {"vehicle_id": "A102", "climate_action": "turn_off"},
            {"vehicle_id": "A102", "climate_action": "关闭"},
            "control_vehicle_climate",
        ),
        (
            "navigation_request",
            "让 A102 导航到南京南站",
            {"vehicle_id": "A102", "destination": "南京南站"},
            {"vehicle_id": "A102", "destination": "南京南站"},
            "estimate_route",
        ),
        (
            "charging_station_query",
            "查询新街口充电站",
            {"location": "新街口"},
            {"location": "新街口"},
            "get_charging_stations",
        ),
        (
            "vehicle_knowledge_query",
            "查询电池保养知识",
            {"query": "电池保养"},
            {"query": "电池保养"},
            "search_vehicle_knowledge",
        ),
    ],
)
def test_planner_templates(intent, text, entities, evidence, action):
    result = validate_intent_payload(
        _payload(intent, entities, evidence),
        text,
        now=NOW,
    )
    plan = RuleBasedPlanner().create_plan(result)
    assert plan.status == "ready"
    assert plan.steps[0].action == action
    assert plan.to_dict()["steps"][0]["required"] is True


def test_additional_time_and_payload_validation_edges():
    assert normalize_departure_time("明天中午一点", now=NOW).endswith("T13:00:00+08:00")
    assert normalize_departure_time("明天早上十二点", now=NOW).endswith("T00:00:00+08:00")
    with pytest.raises(TimeNormalizationError, match="invalid_date"):
        normalize_departure_time("2026年2月30日上午八点", now=NOW)
    with pytest.raises(TimeNormalizationError, match="invalid_time"):
        normalize_departure_time("明天二十五点", now=NOW)

    assert validate_intent_payload([], "text", now=NOW).reason_codes == ("invalid_payload",)
    assert validate_intent_payload(
        {"intent": "made_up", "entities": {}, "evidence": {}}, "text", now=NOW
    ).reason_codes == ("unknown_intent",)
    assert validate_intent_payload(
        {"intent": "range_query", "entities": [], "evidence": {}}, "text", now=NOW
    ).reason_codes == ("invalid_payload",)
