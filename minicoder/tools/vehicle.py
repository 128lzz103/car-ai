"""把统一 ActionDefinition 适配为 Agent function tools。"""

from __future__ import annotations

import json
from typing import Any

from ..actions import ActionDefinition, ActionRegistry
from ..knowledge import KnowledgeRetriever
from ..vehicle.actions import build_vehicle_registry
from ..vehicle.client import VehicleClient
from ..vehicle.policy import VehicleActionPolicy
from .base import Tool

_EXPOSED_ACTIONS = (
    "get_vehicle_status",
    "estimate_route",
    "get_charging_stations",
    "control_vehicle_climate",
    "search_vehicle_knowledge",
)


class RegisteredActionTool(Tool):
    def __init__(self, definition: ActionDefinition) -> None:
        self.definition = definition
        self.name = definition.name
        self.description = definition.description
        self.parameters = definition.input_schema

    def run(self, args: dict[str, Any]) -> str:
        return json.dumps(self.definition.invoke(args), ensure_ascii=False, indent=2)

    def is_read_only(self) -> bool:
        return self.definition.read_only


def build_action_tools(registry: ActionRegistry) -> list[Tool]:
    return [RegisteredActionTool(registry[name]) for name in _EXPOSED_ACTIONS if name in registry]


def build_vehicle_tools(
    client: VehicleClient,
    policy: VehicleActionPolicy,
    knowledge_retriever: KnowledgeRetriever | None = None,
    *,
    knowledge_top_k: int = 5,
) -> list[Tool]:
    return build_action_tools(
        build_vehicle_registry(
            client,
            policy,
            knowledge_retriever,
            knowledge_top_k=knowledge_top_k,
        )
    )
