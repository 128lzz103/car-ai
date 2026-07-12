"""简单引用 PlanExecutor 与复杂行程充电规划 E2E。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from minicoder.executor import PlanExecutionError, PlanExecutor
from minicoder.intent import validate_intent_payload
from minicoder.knowledge import BM25Retriever, JiebaTokenizer, KnowledgeIndex, KnowledgeLoader
from minicoder.knowledge.service import packaged_knowledge_dir
from minicoder.planner import PlanStep, RuleBasedPlanner, TaskPlan
from minicoder.security import PermissionMode
from minicoder.vehicle.actions import build_vehicle_registry
from minicoder.vehicle.client import VehicleClient
from minicoder.vehicle.mock_server import create_app
from minicoder.vehicle.policy import VehicleActionPolicy

NOW = datetime(2026, 8, 3, 12, 0, tzinfo=timezone(timedelta(hours=8)))


def _complex_intent():
    text = "查询 A102 当前续航，明早八点去上海虹桥站，判断是否需要充电"
    return validate_intent_payload(
        {
            "intent": "trip_charge_planning",
            "entities": {
                "vehicle_id": "A102",
                "destination": "上海虹桥站",
                "departure_time": "明早八点",
            },
            "evidence": {
                "vehicle_id": "A102",
                "destination": "上海虹桥站",
                "departure_time": "明早八点",
            },
            "confidence": "high",
            "ambiguities": [],
        },
        text,
        now=NOW,
    )


def test_complex_trip_plan_executes_vehicle_api_and_knowledge_end_to_end(tmp_path):
    app = create_app(token="token")
    client = VehicleClient(
        "http://localhost",
        "token",
        client=TestClient(app),
    )
    policy = VehicleActionPolicy(mode=PermissionMode.ALLOW)
    knowledge_index = KnowledgeIndex(
        KnowledgeLoader(packaged_knowledge_dir()),
        tmp_path / "knowledge.db",
        JiebaTokenizer(),
    )
    knowledge_index.build()
    registry = build_vehicle_registry(client, policy, BM25Retriever(knowledge_index))
    plan = RuleBasedPlanner().create_plan(_complex_intent())

    report = PlanExecutor(registry).execute(plan)

    assert report.status == "completed"
    assert [step.status for step in report.steps] == ["completed"] * 6
    assert report.outputs["vehicle_status"]["estimated_range_km"] == 142
    assert report.outputs["route"]["distance_km"] == 295
    assert report.outputs["energy"]["needs_charging"] is True
    recommendation = report.outputs["recommendation"]
    assert recommendation["recommended_station"]["station_id"] == "CS-WX-001"
    assert "补能一次" in recommendation["message"]
    assert report.outputs["knowledge"]["retrieval"]["result_count"] > 0
    assert report.outputs["knowledge"]["citations"]


def test_executor_rejects_undeclared_or_deep_references():
    plan = TaskPlan(
        _complex_intent().intent,
        "ready",
        (
            PlanStep("one", "constant", {}),
            PlanStep("two", "identity", {"value": {"$ref": "one.value"}}),
        ),
    )
    report = PlanExecutor(
        {"constant": lambda _args: {"value": 1}, "identity": lambda args: args}
    ).execute(plan)
    assert report.status == "failed"
    assert "未声明为依赖" in report.steps[-1].error

    deep = TaskPlan(
        _complex_intent().intent,
        "ready",
        (
            PlanStep("one", "constant", {}),
            PlanStep(
                "two",
                "identity",
                {"value": {"$ref": "one.a.b.c.d"}},
                ("one",),
            ),
        ),
    )
    assert (
        PlanExecutor({"constant": lambda _args: {"a": {}}, "identity": lambda args: args})
        .execute(deep)
        .status
        == "failed"
    )


def test_executor_blocks_invalid_plan_and_missing_action():
    blocked = TaskPlan(_complex_intent().intent, "blocked", reason="not ready")
    with pytest.raises(PlanExecutionError):
        PlanExecutor({}).execute(blocked)

    missing = TaskPlan(
        _complex_intent().intent,
        "ready",
        (PlanStep("step", "missing", {}),),
    )
    report = PlanExecutor({}).execute(missing)
    assert report.status == "failed"
    assert "未注册" in report.steps[0].error
