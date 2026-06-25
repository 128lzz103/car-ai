"""车辆 API、本地知识检索与确定性计算的统一 Action Registry。"""

from __future__ import annotations

from typing import Any

from ..actions import ActionDefinition, ActionHandler, ActionRegistry
from ..knowledge import KnowledgeFilters, KnowledgeRetriever
from .client import VehicleClient
from .errors import VehiclePermissionError, VehicleValidationError
from .policy import VehicleActionPolicy

_OBJECT_SCHEMA = {"type": "object"}
_ARRAY_SCHEMA = {"type": "array"}


def build_vehicle_registry(
    client: VehicleClient,
    policy: VehicleActionPolicy,
    knowledge_retriever: KnowledgeRetriever | None = None,
    *,
    knowledge_top_k: int = 5,
) -> ActionRegistry:
    def get_vehicle_status(args: dict[str, Any]) -> dict[str, Any]:
        vehicle_id = str(args.get("vehicle_id", ""))
        decision = policy.authorize_query(vehicle_id)
        if not decision.allowed:
            raise VehiclePermissionError(decision.reason, code="VEHICLE_QUERY_DENIED")
        return client.get_vehicle_status(vehicle_id).to_dict()

    def estimate_route(args: dict[str, Any]) -> dict[str, Any]:
        vehicle_id = args.get("vehicle_id")
        decision = (
            policy.authorize_query(str(vehicle_id)) if vehicle_id else policy.authorize_query()
        )
        if not decision.allowed:
            raise VehiclePermissionError(decision.reason, code="VEHICLE_QUERY_DENIED")
        return client.estimate_route(
            destination=str(args.get("destination", "")),
            origin=str(args["origin"]) if args.get("origin") else None,
            vehicle_id=str(vehicle_id) if vehicle_id else None,
            departure_time=str(args["departure_time"]) if args.get("departure_time") else None,
        ).to_dict()

    def get_charging_stations(args: dict[str, Any]) -> list[dict[str, Any]]:
        vehicle_id = args.get("vehicle_id")
        decision = (
            policy.authorize_query(str(vehicle_id)) if vehicle_id else policy.authorize_query()
        )
        if not decision.allowed:
            raise VehiclePermissionError(decision.reason, code="VEHICLE_QUERY_DENIED")
        corridor = args.get("corridor")
        return [
            station.to_dict()
            for station in client.get_charging_stations(
                location=str(args["location"]) if args.get("location") else None,
                corridor=[str(item) for item in corridor] if corridor else None,
                vehicle_id=str(vehicle_id) if vehicle_id else None,
                latitude=args.get("latitude"),
                longitude=args.get("longitude"),
                radius_km=args.get("radius_km", 20.0),
            )
        ]

    def control_vehicle_climate(args: dict[str, Any]) -> dict[str, Any]:
        vehicle_id = str(args.get("vehicle_id", ""))
        action = str(args.get("climate_action", ""))
        temperature = args.get("target_temperature")
        description = f"对车辆 {vehicle_id} 执行空调操作 {action}"
        if temperature is not None:
            description += f"，目标温度 {temperature}℃"
        decision = policy.authorize_control(vehicle_id, action=action, description=description)
        if not decision.allowed:
            raise VehiclePermissionError(decision.reason, code="VEHICLE_CONTROL_DENIED")
        return client.control_climate(
            vehicle_id,
            action=action,
            target_temperature=temperature,
        )

    definitions = [
        ActionDefinition(
            "get_vehicle_status",
            "查询指定车辆的电量、预计续航、位置、充电状态和空调状态。",
            _schema({"vehicle_id": {"type": "string", "minLength": 1}}, ["vehicle_id"]),
            get_vehicle_status,
            _OBJECT_SCHEMA,
            read_only=True,
            risk="low",
        ),
        ActionDefinition(
            "estimate_route",
            "使用模拟路线服务估算距离、时间、路线走廊和车辆预计耗电。",
            _schema(
                {
                    "vehicle_id": {"type": "string"},
                    "origin": {"type": "string"},
                    "destination": {"type": "string", "minLength": 1},
                    "departure_time": {"type": "string"},
                    "departure_timezone": {"type": "string"},
                },
                ["destination"],
            ),
            estimate_route,
            _OBJECT_SCHEMA,
            read_only=True,
            risk="low",
        ),
        ActionDefinition(
            "get_charging_stations",
            "按地点、车辆位置或路线走廊查询模拟充电站。",
            _schema(
                {
                    "vehicle_id": {"type": "string"},
                    "location": {"type": "string"},
                    "corridor": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 20,
                    },
                    "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                    "longitude": {"type": "number", "minimum": -180, "maximum": 180},
                    "radius_km": {"type": "number", "minimum": 0.1, "maximum": 100},
                }
            ),
            get_charging_stations,
            _ARRAY_SCHEMA,
            read_only=True,
            risk="low",
        ),
        ActionDefinition(
            "control_vehicle_climate",
            "经独立车辆权限确认后，开关空调或设置 16～30℃ 的目标温度。",
            _schema(
                {
                    "vehicle_id": {"type": "string", "minLength": 1},
                    "climate_action": {
                        "type": "string",
                        "enum": ["turn_on", "turn_off", "set_temperature"],
                    },
                    "target_temperature": {
                        "type": "number",
                        "minimum": 16,
                        "maximum": 30,
                    },
                },
                ["vehicle_id", "climate_action"],
            ),
            control_vehicle_climate,
            _OBJECT_SCHEMA,
            read_only=False,
            risk="high",
        ),
        ActionDefinition(
            "calculate_energy_requirement",
            "根据车辆可用电量和路线预计能耗计算补能缺口。",
            _schema({"vehicle": _OBJECT_SCHEMA, "route": _OBJECT_SCHEMA}, ["vehicle", "route"]),
            calculate_energy_requirement,
            _OBJECT_SCHEMA,
            read_only=True,
            risk="low",
        ),
        ActionDefinition(
            "build_charging_recommendation",
            "结合能耗、路线和充电站生成确定性充电建议。",
            _schema(
                {"energy": _OBJECT_SCHEMA, "route": _OBJECT_SCHEMA, "stations": _ARRAY_SCHEMA},
                ["energy", "route", "stations"],
            ),
            build_charging_recommendation,
            _OBJECT_SCHEMA,
            read_only=True,
            risk="low",
        ),
    ]
    if knowledge_retriever is not None:

        def search_vehicle_knowledge(args: dict[str, Any]) -> dict[str, Any]:
            models = tuple(str(item).upper() for item in args.get("vehicle_models", []))
            vehicle_id = str(args.get("vehicle_id", "")).strip()
            if vehicle_id:
                decision = policy.authorize_query(vehicle_id)
                if not decision.allowed:
                    raise VehiclePermissionError(decision.reason, code="VEHICLE_QUERY_DENIED")
                models = tuple(
                    dict.fromkeys((*models, client.get_vehicle_status(vehicle_id).model))
                )
            results = knowledge_retriever.search(
                str(args.get("query", "")),
                filters=KnowledgeFilters(
                    vehicle_models=models,
                    category=str(args["category"]) if args.get("category") else None,
                    component=str(args["component"]) if args.get("component") else None,
                ),
                top_k=int(args.get("top_k", knowledge_top_k)),
            )
            return {
                "query": args["query"],
                "results": [item.to_dict() for item in results],
                "citations": [item.citation for item in results],
                "retrieval": {"source": "bm25", "result_count": len(results)},
            }

        definitions.append(
            ActionDefinition(
                "search_vehicle_knowledge",
                "检索本地汽车知识库，返回原文片段、车型元数据和可追踪引用。",
                _schema(
                    {
                        "query": {"type": "string", "minLength": 1, "maxLength": 500},
                        "vehicle_id": {"type": "string"},
                        "vehicle_models": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 10,
                        },
                        "category": {"type": "string"},
                        "component": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    ["query"],
                ),
                search_vehicle_knowledge,
                _OBJECT_SCHEMA,
                read_only=True,
                risk="low",
            )
        )
    return ActionRegistry(tuple(definitions))


def build_vehicle_actions(
    client: VehicleClient,
    policy: VehicleActionPolicy,
    knowledge_retriever: KnowledgeRetriever | None = None,
    *,
    knowledge_top_k: int = 5,
) -> dict[str, ActionHandler]:
    """兼容旧调用方；新代码应直接使用 build_vehicle_registry。"""
    return build_vehicle_registry(
        client,
        policy,
        knowledge_retriever,
        knowledge_top_k=knowledge_top_k,
    ).handlers()


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def calculate_energy_requirement(args: dict[str, Any]) -> dict[str, Any]:
    vehicle = args.get("vehicle")
    route = args.get("route")
    if not isinstance(vehicle, dict) or not isinstance(route, dict):
        raise VehicleValidationError(
            "能耗计算需要 vehicle 与 route 对象",
            code="INVALID_ENERGY_INPUT",
        )
    try:
        capacity = float(vehicle["battery_capacity_kwh"])
        battery_soc = float(vehicle["battery_soc"])
        route_energy = float(route["estimated_energy_kwh"])
    except (KeyError, TypeError, ValueError) as error:
        raise VehicleValidationError("能耗计算输入字段无效", code="INVALID_ENERGY_INPUT") from error
    available = round(capacity * battery_soc / 100, 2)
    reserve = round(capacity * 0.10, 2)
    deficit = round(max(0.0, route_energy + reserve - available), 2)
    return {
        "available_energy_kwh": available,
        "route_energy_kwh": route_energy,
        "reserve_energy_kwh": reserve,
        "energy_deficit_kwh": deficit,
        "needs_charging": deficit > 0,
    }


def build_charging_recommendation(args: dict[str, Any]) -> dict[str, Any]:
    energy = args.get("energy")
    stations = args.get("stations")
    route = args.get("route")
    if (
        not isinstance(energy, dict)
        or not isinstance(stations, list)
        or not isinstance(route, dict)
    ):
        raise VehicleValidationError(
            "充电建议需要 energy、stations 与 route",
            code="INVALID_RECOMMENDATION_INPUT",
        )
    if not energy.get("needs_charging"):
        return {
            "needs_charging": False,
            "message": "当前电量可以覆盖行程并保留 10% 安全电量。",
            "recommended_station": None,
        }
    available = [item for item in stations if int(item.get("available_piles", 0)) > 0]
    if not available:
        return {
            "needs_charging": True,
            "message": "预计存在能量缺口，但模拟路线中没有可用充电站。",
            "recommended_station": None,
        }
    station = next(
        (item for item in available if "无锡" in str(item.get("name", ""))), available[0]
    )
    return {
        "needs_charging": True,
        "energy_deficit_kwh": energy.get("energy_deficit_kwh"),
        "route_distance_km": route.get("distance_km"),
        "message": f"建议途中在{station.get('name')}补能一次。",
        "recommended_station": station,
    }
