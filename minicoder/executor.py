"""受限顺序 PlanExecutor：固定 Action Registry + 简单前序输出引用。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .actions import ActionRegistry, validate_schema
from .planner import TaskPlan


class PlanExecutionError(RuntimeError):
    pass


class PlanValidationError(PlanExecutionError):
    pass


@dataclass(frozen=True)
class StepExecution:
    step_id: str
    action: str
    status: str
    output: Any = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "status": self.status,
            "output": self.output,
            "error": self.error,
        }


@dataclass(frozen=True)
class ExecutionReport:
    status: str
    steps: tuple[StepExecution, ...]

    @property
    def outputs(self) -> dict[str, Any]:
        return {item.step_id: item.output for item in self.steps if item.status == "completed"}

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "steps": [step.to_dict() for step in self.steps]}


class PlanExecutor:
    """不支持循环、并行、表达式或动态重规划。"""

    def __init__(
        self,
        actions: dict[str, Any] | ActionRegistry,
        *,
        validator: PlanValidator | None = None,
        read_only: bool = False,
    ) -> None:
        if isinstance(actions, ActionRegistry):
            self.registry: ActionRegistry | None = actions
            self.actions = actions.handlers()
            self.validator = validator or PlanValidator(actions, read_only=read_only)
        else:
            self.registry = None
            self.actions = dict(actions)
            self.validator = validator

    def execute(self, plan: TaskPlan) -> ExecutionReport:
        if plan.status != "ready":
            raise PlanExecutionError(f"计划不可执行：{plan.reason or plan.status}")
        if self.validator is not None:
            self.validator.validate(plan)
        outputs: dict[str, Any] = {}
        executions: list[StepExecution] = []
        known_steps: set[str] = set()
        for step in plan.steps:
            if step.id in known_steps:
                raise PlanExecutionError(f"计划包含重复步骤 ID：{step.id}")
            known_steps.add(step.id)
            missing_dependencies = [item for item in step.depends_on if item not in outputs]
            if missing_dependencies:
                error = f"前置步骤未完成：{', '.join(missing_dependencies)}"
                executions.append(StepExecution(step.id, step.action, "blocked", error=error))
                return ExecutionReport("failed", tuple(executions))
            handler = self.actions.get(step.action)
            if handler is None:
                error = f"未注册 action：{step.action}"
                executions.append(StepExecution(step.id, step.action, "failed", error=error))
                if step.required:
                    return ExecutionReport("failed", tuple(executions))
                continue
            try:
                arguments = _resolve_value(step.arguments, outputs, set(step.depends_on))
                output = handler(arguments)
            except Exception as error:  # noqa: BLE001 -- 执行报告必须保留业务错误
                executions.append(
                    StepExecution(
                        step.id,
                        step.action,
                        "failed",
                        error=f"{type(error).__name__}: {error}",
                    )
                )
                if step.required:
                    return ExecutionReport("failed", tuple(executions))
                continue
            outputs[step.id] = output
            executions.append(StepExecution(step.id, step.action, "completed", output=output))
        return ExecutionReport("completed", tuple(executions))


class PlanValidator:
    def __init__(
        self,
        registry: ActionRegistry,
        *,
        max_steps: int = 8,
        read_only: bool = False,
    ) -> None:
        self.registry = registry
        self.max_steps = max_steps
        self.read_only = read_only

    def validate(self, plan: TaskPlan) -> None:
        if plan.status != "ready":
            raise PlanValidationError(f"计划不可验证：{plan.reason or plan.status}")
        if not plan.steps:
            raise PlanValidationError("可执行计划不能为空")
        if len(plan.steps) > self.max_steps:
            raise PlanValidationError(f"计划步骤超过限制：{self.max_steps}")
        seen: set[str] = set()
        for step in plan.steps:
            if not step.id or step.id in seen:
                raise PlanValidationError(f"计划步骤 ID 为空或重复：{step.id}")
            if step.action not in self.registry:
                raise PlanValidationError(f"计划使用未注册 action：{step.action}")
            if len(set(step.depends_on)) != len(step.depends_on):
                raise PlanValidationError(f"步骤 {step.id} 包含重复依赖")
            missing = [item for item in step.depends_on if item not in seen]
            if missing:
                raise PlanValidationError(
                    f"步骤 {step.id} 依赖不存在或不是前序步骤：{', '.join(missing)}"
                )
            definition = self.registry[step.action]
            if self.read_only and not definition.read_only:
                raise PlanValidationError(f"只读模式禁止 action：{step.action}")
            try:
                validate_schema(
                    step.arguments,
                    definition.input_schema,
                    path=f"steps.{step.id}.arguments",
                    allow_references=True,
                )
                self._validate_references(step.arguments, set(step.depends_on), seen)
            except Exception as error:
                raise PlanValidationError(f"步骤 {step.id} 参数无效：{error}") from error
            seen.add(step.id)

    @classmethod
    def _validate_references(
        cls,
        value: Any,
        dependencies: set[str],
        seen: set[str],
    ) -> None:
        if isinstance(value, dict) and set(value) == {"$ref"}:
            reference = value["$ref"]
            if not isinstance(reference, str):
                raise PlanValidationError("$ref 必须是字符串")
            parts = reference.split(".")
            if not 1 <= len(parts) <= 4 or any(not item for item in parts):
                raise PlanValidationError(f"不支持的引用：{reference}")
            if parts[0] not in dependencies or parts[0] not in seen:
                raise PlanValidationError(f"引用未声明的前序依赖：{parts[0]}")
            return
        if isinstance(value, dict):
            for item in value.values():
                cls._validate_references(item, dependencies, seen)
        elif isinstance(value, list):
            for item in value:
                cls._validate_references(item, dependencies, seen)


def _resolve_value(value: Any, outputs: dict[str, Any], dependencies: set[str]) -> Any:
    if isinstance(value, dict) and set(value) == {"$ref"}:
        reference = value["$ref"]
        if not isinstance(reference, str):
            raise PlanExecutionError("$ref 必须是字符串")
        parts = reference.split(".")
        if not 1 <= len(parts) <= 4 or any(not part for part in parts):
            raise PlanExecutionError(f"不支持的引用：{reference}")
        step_id, *path = parts
        if step_id not in dependencies:
            raise PlanExecutionError(f"引用步骤未声明为依赖：{step_id}")
        if step_id not in outputs:
            raise PlanExecutionError(f"引用步骤尚无输出：{step_id}")
        current = outputs[step_id]
        for field in path:
            if not isinstance(current, dict) or field not in current:
                raise PlanExecutionError(f"引用字段不存在：{reference}")
            current = current[field]
        return current
    if isinstance(value, dict):
        return {key: _resolve_value(item, outputs, dependencies) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item, outputs, dependencies) for item in value]
    return value
