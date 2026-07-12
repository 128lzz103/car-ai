"""统一 ActionRegistry、Schema 校验和 PlanValidator 测试。"""

from __future__ import annotations

import pytest

from minicoder.actions import (
    ActionArgumentError,
    ActionDefinition,
    ActionRegistrationError,
    ActionRegistry,
)
from minicoder.executor import PlanExecutor, PlanValidationError, PlanValidator
from minicoder.intent import IntentName, validate_intent_payload
from minicoder.planner import ConstrainedHybridPlanner, PlanStep, TaskPlan
from minicoder.providers import Reply, ToolCall


def _definition(name="echo", *, read_only=True):
    return ActionDefinition(
        name=name,
        description="返回输入值",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"value": {"type": "string", "minLength": 1}},
            "required": ["value"],
        },
        handler=lambda args: {"value": args["value"]},
        output_schema={"type": "object"},
        read_only=read_only,
        risk="low" if read_only else "high",
    )


def _plan(*steps):
    return TaskPlan(IntentName.VEHICLE_KNOWLEDGE_QUERY, "ready", tuple(steps))


def test_registry_is_single_source_for_descriptor_handler_and_schema():
    registry = ActionRegistry((_definition(),))

    assert registry["echo"].invoke({"value": "ok"}) == {"value": "ok"}
    assert registry.handlers()["echo"]({"value": "ok"}) == {"value": "ok"}
    assert "handler" not in registry.public_descriptors()[0]
    with pytest.raises(ActionArgumentError, match="必填"):
        registry["echo"].invoke({})
    with pytest.raises(ActionArgumentError, match="未知字段"):
        registry["echo"].invoke({"value": "ok", "extra": True})


def test_registry_rejects_duplicate_and_invalid_definitions():
    registry = ActionRegistry((_definition(),))
    with pytest.raises(ActionRegistrationError, match="重复"):
        registry.register(_definition())
    with pytest.raises(ActionRegistrationError, match="名称"):
        _definition("Bad-Name")


def test_plan_validator_checks_dependencies_references_and_read_only():
    registry = ActionRegistry((_definition("first"), _definition("write", read_only=False)))
    validator = PlanValidator(registry, read_only=True)

    valid = _plan(
        PlanStep("one", "first", {"value": "ok"}),
        PlanStep("two", "first", {"value": {"$ref": "one.value"}}, ("one",)),
    )
    validator.validate(valid)

    with pytest.raises(PlanValidationError, match="只读模式"):
        validator.validate(_plan(PlanStep("write", "write", {"value": "x"})))
    with pytest.raises(PlanValidationError, match="不是前序"):
        validator.validate(_plan(PlanStep("two", "first", {"value": "x"}, ("missing",))))
    with pytest.raises(PlanValidationError, match="未声明"):
        validator.validate(
            _plan(
                PlanStep("one", "first", {"value": "ok"}),
                PlanStep("two", "first", {"value": {"$ref": "one.value"}}),
            )
        )


def test_plan_validator_limits_steps_and_registered_actions():
    registry = ActionRegistry((_definition(),))
    validator = PlanValidator(registry, max_steps=2)

    with pytest.raises(PlanValidationError, match="步骤超过"):
        validator.validate(
            _plan(
                PlanStep("one", "echo", {"value": "1"}),
                PlanStep("two", "echo", {"value": "2"}),
                PlanStep("three", "echo", {"value": "3"}),
            )
        )
    with pytest.raises(PlanValidationError, match="未注册"):
        validator.validate(_plan(PlanStep("one", "missing", {})))


def test_executor_with_registry_validates_resolved_arguments_and_output():
    registry = ActionRegistry((_definition("first"), _definition("second")))
    report = PlanExecutor(registry).execute(
        _plan(
            PlanStep("one", "first", {"value": "resolved"}),
            PlanStep(
                "two",
                "second",
                {"value": {"$ref": "one.value"}},
                ("one",),
            ),
        )
    )

    assert report.status == "completed"
    assert report.outputs["two"] == {"value": "resolved"}


def test_optional_unregistered_action_does_not_fail_legacy_executor():
    report = PlanExecutor({"echo": lambda args: args}).execute(
        _plan(
            PlanStep("one", "echo", {"value": "ok"}),
            PlanStep("optional", "missing", {}, required=False),
        )
    )

    assert report.status == "completed"
    assert report.steps[-1].status == "failed"


class PlannerProvider:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def _knowledge_intent():
    return validate_intent_payload(
        {
            "intent": "vehicle_knowledge_query",
            "entities": {"query": "电池保养"},
            "evidence": {"query": "电池保养"},
            "confidence": "high",
            "ambiguities": [],
        },
        "电池保养",
    )


def test_hybrid_planner_accepts_only_validated_structured_candidate():
    registry = ActionRegistry((_definition(),))
    provider = PlannerProvider(
        Reply(
            tool_calls=[
                ToolCall(
                    "plan-1",
                    "emit_task_plan",
                    {
                        "steps": [
                            {
                                "id": "answer",
                                "action": "echo",
                                "arguments": {"value": "ok"},
                                "depends_on": [],
                                "required": True,
                            }
                        ]
                    },
                )
            ],
            input_tokens=12,
            output_tokens=7,
        )
    )
    validator = PlanValidator(registry)
    planner = ConstrainedHybridPlanner(
        provider,
        "planner-model",
        registry,
        validate=validator.validate,
        enabled=True,
    )

    plan = planner.create_plan(_knowledge_intent())

    assert plan.strategy == "llm"
    assert plan.steps[0].action == "echo"
    assert planner.last_input_tokens == 12
    assert planner.last_output_tokens == 7
    assert provider.calls[0]["tool_choice"] == "emit_task_plan"
    assert provider.calls[0]["stream"] is False
    assert "handler" not in provider.calls[0]["system"]


def test_hybrid_planner_falls_back_when_candidate_is_invalid():
    registry = ActionRegistry((_definition(),))
    provider = PlannerProvider(
        Reply(
            text='{"steps":[{"id":"bad","action":"missing","arguments":{},'
            '"depends_on":[],"required":true}]}'
        )
    )
    planner = ConstrainedHybridPlanner(
        provider,
        "planner-model",
        registry,
        validate=PlanValidator(registry).validate,
        enabled=True,
    )

    plan = planner.create_plan(_knowledge_intent())

    assert plan.strategy == "rule"
    assert plan.steps[0].action == "search_vehicle_knowledge"
    assert "未注册" in planner.last_fallback_reason
