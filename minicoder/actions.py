"""统一 Action 能力定义，供 Planner、Executor 与 ToolAdapter 共享。"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

ActionHandler = Callable[[dict[str, Any]], Any]
ActionRisk = Literal["low", "medium", "high"]
_ACTION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class ActionError(RuntimeError):
    pass


class ActionRegistrationError(ActionError):
    pass


class ActionArgumentError(ActionError):
    pass


@dataclass(frozen=True)
class ActionDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ActionHandler = field(repr=False, compare=False)
    output_schema: dict[str, Any] | None = None
    read_only: bool = False
    risk: ActionRisk = "medium"
    fallback_actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if _ACTION_NAME_RE.fullmatch(self.name) is None:
            raise ActionRegistrationError(f"非法 action 名称：{self.name}")
        if not self.description.strip():
            raise ActionRegistrationError(f"action {self.name} 缺少描述")
        if self.risk not in {"low", "medium", "high"}:
            raise ActionRegistrationError(f"action {self.name} 风险等级无效")
        if self.input_schema.get("type") != "object":
            raise ActionRegistrationError(f"action {self.name} 输入 Schema 必须是对象")

    def invoke(self, arguments: dict[str, Any]) -> Any:
        validate_schema(arguments, self.input_schema, path=f"{self.name}.arguments")
        output = self.handler(arguments)
        if self.output_schema is not None:
            validate_schema(output, self.output_schema, path=f"{self.name}.output")
        return output

    def public_descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "read_only": self.read_only,
            "risk": self.risk,
            "fallback_actions": list(self.fallback_actions),
        }


class ActionRegistry(Mapping[str, ActionDefinition]):
    def __init__(self, actions: list[ActionDefinition] | tuple[ActionDefinition, ...] = ()) -> None:
        self._actions: dict[str, ActionDefinition] = {}
        for action in actions:
            self.register(action)

    def register(self, action: ActionDefinition) -> None:
        if action.name in self._actions:
            raise ActionRegistrationError(f"重复 action：{action.name}")
        self._actions[action.name] = action

    def __getitem__(self, name: str) -> ActionDefinition:
        return self._actions[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._actions)

    def __len__(self) -> int:
        return len(self._actions)

    def __contains__(self, name: object) -> bool:
        return name in self._actions

    def handlers(self) -> dict[str, ActionHandler]:
        return {name: definition.invoke for name, definition in self._actions.items()}

    def public_descriptors(self) -> list[dict[str, Any]]:
        return [definition.public_descriptor() for definition in self._actions.values()]


def validate_schema(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str = "value",
    allow_references: bool = False,
) -> None:
    if allow_references and isinstance(value, dict) and set(value) == {"$ref"}:
        if not isinstance(value["$ref"], str):
            raise ActionArgumentError(f"{path}.$ref 必须是字符串")
        return
    if "enum" in schema and value not in schema["enum"]:
        raise ActionArgumentError(f"{path} 不在允许值范围内")
    expected = schema.get("type")
    expected_types = expected if isinstance(expected, list) else [expected]
    if expected is not None and not any(_matches_type(value, item) for item in expected_types):
        raise ActionArgumentError(f"{path} 类型错误，期望 {expected}")
    if value is None:
        return
    if "object" in expected_types:
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for name in required:
            if name not in value:
                raise ActionArgumentError(f"{path} 缺少必填字段 {name}")
        if schema.get("additionalProperties") is False:
            unknown = set(value) - set(properties)
            if unknown:
                raise ActionArgumentError(f"{path} 包含未知字段 {sorted(unknown)[0]}")
        for name, item in value.items():
            if name in properties:
                validate_schema(
                    item,
                    properties[name],
                    path=f"{path}.{name}",
                    allow_references=allow_references,
                )
    if "array" in expected_types:
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            raise ActionArgumentError(f"{path} 元素过多")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                validate_schema(
                    item,
                    item_schema,
                    path=f"{path}[{index}]",
                    allow_references=allow_references,
                )
    if "string" in expected_types:
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            raise ActionArgumentError(f"{path} 长度不足")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise ActionArgumentError(f"{path} 长度超限")
    if any(item in {"number", "integer"} for item in expected_types):
        if "minimum" in schema and value < schema["minimum"]:
            raise ActionArgumentError(f"{path} 小于最小值")
        if "maximum" in schema and value > schema["maximum"]:
            raise ActionArgumentError(f"{path} 大于最大值")


def _matches_type(value: Any, expected: str | None) -> bool:
    if expected == "null":
        return value is None
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True
