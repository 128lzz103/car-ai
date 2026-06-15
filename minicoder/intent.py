"""汽车领域的结构化意图、实体校验与时间规范化。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class IntentName(str, Enum):
    RANGE_QUERY = "range_query"
    VEHICLE_STATUS_QUERY = "vehicle_status_query"
    CLIMATE_CONTROL = "climate_control"
    NAVIGATION_REQUEST = "navigation_request"
    TRIP_CHARGE_PLANNING = "trip_charge_planning"
    CHARGING_STATION_QUERY = "charging_station_query"
    VEHICLE_KNOWLEDGE_QUERY = "vehicle_knowledge_query"
    OTHER = "other"
    UNKNOWN = "unknown"


class IntentStatus(str, Enum):
    READY = "ready"
    NEEDS_CLARIFICATION = "needs_clarification"
    OUT_OF_SCOPE = "out_of_scope"
    CANCELLED = "cancelled"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class IntentSpec:
    required: frozenset[str] = frozenset()
    optional: frozenset[str] = frozenset()


INTENT_SPECS: dict[IntentName, IntentSpec] = {
    IntentName.RANGE_QUERY: IntentSpec(frozenset({"vehicle_id"})),
    IntentName.VEHICLE_STATUS_QUERY: IntentSpec(frozenset({"vehicle_id"})),
    IntentName.CLIMATE_CONTROL: IntentSpec(
        frozenset({"vehicle_id", "climate_action"}),
        frozenset({"target_temperature"}),
    ),
    IntentName.NAVIGATION_REQUEST: IntentSpec(
        frozenset({"destination"}),
        frozenset({"origin", "vehicle_id", "departure_time"}),
    ),
    IntentName.TRIP_CHARGE_PLANNING: IntentSpec(
        frozenset({"vehicle_id", "destination"}),
        frozenset({"origin", "departure_time"}),
    ),
    IntentName.CHARGING_STATION_QUERY: IntentSpec(
        frozenset(), frozenset({"location", "vehicle_id"})
    ),
    IntentName.VEHICLE_KNOWLEDGE_QUERY: IntentSpec(frozenset({"query"}), frozenset({"vehicle_id"})),
    IntentName.OTHER: IntentSpec(),
    IntentName.UNKNOWN: IntentSpec(),
}

ENTITY_NAMES = frozenset(
    {
        "vehicle_id",
        "origin",
        "destination",
        "departure_time",
        "departure_timezone",
        "climate_action",
        "target_temperature",
        "location",
        "query",
    }
)
CLIMATE_ACTIONS = frozenset({"turn_on", "turn_off", "set_temperature"})
_VEHICLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


INTENT_TOOL_SCHEMA: dict[str, Any] = {
    "name": "emit_vehicle_intent",
    "description": "只返回用户车辆请求的结构化意图与原文证据，不回答请求。",
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intent": {"type": "string", "enum": [item.value for item in IntentName]},
            "entities": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "vehicle_id": {"type": ["string", "null"]},
                    "origin": {"type": ["string", "null"]},
                    "destination": {"type": ["string", "null"]},
                    "departure_time": {"type": ["string", "null"]},
                    "climate_action": {
                        "type": ["string", "null"],
                        "enum": [*sorted(CLIMATE_ACTIONS), None],
                    },
                    "target_temperature": {"type": ["number", "null"]},
                    "location": {"type": ["string", "null"]},
                    "query": {"type": ["string", "null"]},
                },
            },
            "evidence": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    name: {"type": ["string", "null"]}
                    for name in sorted(ENTITY_NAMES - {"departure_timezone"})
                },
            },
            "confidence": {
                "type": "string",
                "enum": [item.value for item in Confidence],
            },
            "ambiguities": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        },
        "required": ["intent", "entities", "evidence", "confidence", "ambiguities"],
    },
}


@dataclass(frozen=True)
class IntentResult:
    intent: IntentName
    status: IntentStatus
    confidence: Confidence
    entities: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)
    missing_fields: tuple[str, ...] = ()
    ambiguities: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "domain": "vehicle",
            "intent": self.intent.value,
            "status": self.status.value,
            "confidence": self.confidence.value,
            "entities": dict(self.entities),
            "evidence": dict(self.evidence),
            "missing_fields": list(self.missing_fields),
            "ambiguities": list(self.ambiguities),
            "reason_codes": list(self.reason_codes),
        }


class TimeNormalizationError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def get_timezone(name: str) -> tzinfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as error:
        if name == "Asia/Shanghai":
            return timezone(timedelta(hours=8), name)
        raise ValueError(f"未知时区: {name!r}") from error


def normalize_departure_time(
    raw: str,
    *,
    timezone_name: str = "Asia/Shanghai",
    now: datetime | None = None,
) -> str:
    """将首版支持的中文时间表达转换为带 offset 的 ISO 8601。"""
    zone = get_timezone(timezone_name)
    reference = now or datetime.now(zone)
    reference = reference.astimezone(zone) if reference.tzinfo else reference.replace(tzinfo=zone)
    text = _clean_string(raw, 80)
    if not text:
        raise TimeNormalizationError("time_empty")

    try:
        parsed_iso = datetime.fromisoformat(text)
    except ValueError:
        parsed_iso = None
    if parsed_iso is not None:
        if parsed_iso.tzinfo is None:
            parsed_iso = parsed_iso.replace(tzinfo=zone)
        normalized = parsed_iso.astimezone(zone)
        _validate_future_time(normalized, reference)
        return normalized.isoformat()

    text = text.replace("明早", "明天早上").replace("今晚", "今天晚上")
    target_date = None
    offsets = {"今天": 0, "明天": 1, "后天": 2}
    for marker, offset in offsets.items():
        if marker in text:
            target_date = (reference + timedelta(days=offset)).date()
            break

    full_date = re.search(r"(\d{4})[-年](\d{1,2})[-月](\d{1,2})日?", text)
    short_date = re.search(r"(?<!\d)(\d{1,2})月(\d{1,2})日?", text)
    try:
        if full_date:
            target_date = datetime(
                int(full_date.group(1)), int(full_date.group(2)), int(full_date.group(3))
            ).date()
        elif short_date:
            target_date = datetime(
                reference.year, int(short_date.group(1)), int(short_date.group(2))
            ).date()
    except ValueError as error:
        raise TimeNormalizationError("invalid_date") from error
    if target_date is None:
        raise TimeNormalizationError("missing_date")

    hour, minute = _extract_clock(text)
    period = _time_period(text)
    if period in {"afternoon", "evening"} and hour < 12:
        hour += 12
    elif period == "noon" and hour < 11:
        hour += 12
    elif period == "morning" and hour == 12:
        hour = 0
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise TimeNormalizationError("invalid_time")

    result = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        hour,
        minute,
        tzinfo=zone,
    )
    roundtrip = result.astimezone(timezone.utc).astimezone(zone)
    if roundtrip.replace(fold=result.fold) != result:
        raise TimeNormalizationError("nonexistent_local_time")
    _validate_future_time(result, reference)
    return result.isoformat()


def validate_intent_payload(
    payload: dict[str, Any],
    source_text: str,
    *,
    timezone_name: str = "Asia/Shanghai",
    now: datetime | None = None,
) -> IntentResult:
    """将不可信的模型参数转换成可供 Planner 使用的本地协议。"""
    if not isinstance(payload, dict):
        return invalid_intent("invalid_payload")
    try:
        intent = IntentName(str(payload.get("intent", "unknown")))
    except ValueError:
        return invalid_intent("unknown_intent")
    if intent is IntentName.UNKNOWN:
        return invalid_intent("unknown_intent")

    raw_entities = payload.get("entities")
    raw_evidence = payload.get("evidence")
    if not isinstance(raw_entities, dict) or not isinstance(raw_evidence, dict):
        return invalid_intent("invalid_payload")

    entities: dict[str, Any] = {}
    evidence: dict[str, str] = {}
    reason_codes: list[str] = []
    ambiguities = _clean_list(payload.get("ambiguities"), max_items=5, max_length=120)

    for name, raw_value in raw_entities.items():
        if name not in ENTITY_NAMES or name == "departure_timezone" or raw_value is None:
            continue
        try:
            value = _normalize_entity(name, raw_value, timezone_name=timezone_name, now=now)
        except (TypeError, ValueError, TimeNormalizationError) as error:
            reason = (
                error.reason if isinstance(error, TimeNormalizationError) else f"invalid_{name}"
            )
            reason_codes.append(reason)
            continue

        raw_proof = raw_evidence.get(name)
        proof = _clean_string(raw_proof, 120) if isinstance(raw_proof, str) else ""
        if not proof and isinstance(raw_value, str) and _contains(source_text, raw_value):
            proof = raw_value.strip()
        if not proof or not _contains(source_text, proof):
            reason_codes.append(f"missing_evidence:{name}")
            continue
        entities[name] = value
        evidence[name] = proof

    if "departure_time" in entities:
        entities["departure_timezone"] = timezone_name

    missing = _missing_fields(intent, entities)
    confidence = _parse_confidence(payload.get("confidence"))
    if reason_codes or ambiguities or missing:
        confidence = Confidence.LOW
    elif confidence is Confidence.HIGH and len(evidence) < len(entities):
        confidence = Confidence.MEDIUM

    if intent is IntentName.OTHER:
        status = IntentStatus.OUT_OF_SCOPE
    elif missing or ambiguities or reason_codes:
        status = IntentStatus.NEEDS_CLARIFICATION
    else:
        status = IntentStatus.READY
    return IntentResult(
        intent=intent,
        status=status,
        confidence=confidence,
        entities=entities,
        evidence=evidence,
        missing_fields=tuple(missing),
        ambiguities=tuple(ambiguities),
        reason_codes=tuple(dict.fromkeys(reason_codes)),
    )


def invalid_intent(reason: str) -> IntentResult:
    return IntentResult(
        intent=IntentName.UNKNOWN,
        status=IntentStatus.NEEDS_CLARIFICATION,
        confidence=Confidence.LOW,
        reason_codes=(reason,),
    )


def cancelled_intent() -> IntentResult:
    return IntentResult(
        intent=IntentName.OTHER,
        status=IntentStatus.CANCELLED,
        confidence=Confidence.HIGH,
        reason_codes=("cancelled_by_user",),
    )


def _normalize_entity(
    name: str,
    value: Any,
    *,
    timezone_name: str,
    now: datetime | None,
) -> Any:
    if name == "target_temperature":
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise TypeError(name)
        number = float(value)
        if not 16 <= number <= 30:
            raise ValueError(name)
        return int(number) if number.is_integer() else number
    if not isinstance(value, str):
        raise TypeError(name)
    cleaned = _clean_string(value, 500 if name == "query" else 120)
    if not cleaned:
        raise ValueError(name)
    if name == "vehicle_id":
        if not _VEHICLE_ID_RE.fullmatch(cleaned):
            raise ValueError(name)
        return cleaned.upper()
    if name == "climate_action":
        if cleaned not in CLIMATE_ACTIONS:
            raise ValueError(name)
        return cleaned
    if name == "departure_time":
        return normalize_departure_time(cleaned, timezone_name=timezone_name, now=now)
    return cleaned


def _missing_fields(intent: IntentName, entities: dict[str, Any]) -> list[str]:
    missing = sorted(INTENT_SPECS[intent].required - entities.keys())
    if (
        intent is IntentName.CLIMATE_CONTROL
        and entities.get("climate_action") == "set_temperature"
        and "target_temperature" not in entities
    ):
        missing.append("target_temperature")
    if intent is IntentName.CHARGING_STATION_QUERY and not (
        entities.get("location") or entities.get("vehicle_id")
    ):
        missing.append("location_or_vehicle_id")
    if intent is IntentName.NAVIGATION_REQUEST and not (
        entities.get("origin") or entities.get("vehicle_id")
    ):
        missing.append("origin_or_vehicle_id")
    return list(dict.fromkeys(missing))


def _clean_string(value: Any, max_length: int) -> str:
    if not isinstance(value, str):
        return ""
    return _CONTROL_RE.sub("", value).strip()[:max_length]


def _clean_list(value: Any, *, max_items: int, max_length: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        cleaned
        for item in value[:max_items]
        if isinstance(item, str) and (cleaned := _clean_string(item, max_length))
    ]


def _contains(source: str, evidence: str) -> bool:
    compact_source = re.sub(r"\s+", "", source).casefold()
    compact_evidence = re.sub(r"\s+", "", evidence).casefold()
    return bool(compact_evidence and compact_evidence in compact_source)


def _parse_confidence(value: Any) -> Confidence:
    try:
        return Confidence(str(value))
    except ValueError:
        return Confidence.LOW


def _extract_clock(text: str) -> tuple[int, int]:
    colon = re.search(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)", text)
    if colon:
        return int(colon.group(1)), int(colon.group(2))
    match = re.search(
        r"([零〇一二两三四五六七八九十\d]{1,3})点(?:(半)|([零〇一二三四五六七八九十\d]{1,3})分?)?",
        text,
    )
    if not match:
        raise TimeNormalizationError("missing_time")
    hour = _chinese_number(match.group(1))
    minute = 30 if match.group(2) else _chinese_number(match.group(3) or "0")
    return hour, minute


def _chinese_number(value: str) -> int:
    if value.isdigit():
        return int(value)
    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if value == "十":
        return 10
    if "十" in value:
        left, _, right = value.partition("十")
        return (digits.get(left, 1) * 10) + digits.get(right, 0)
    if len(value) == 1 and value in digits:
        return digits[value]
    raise TimeNormalizationError("invalid_time")


def _time_period(text: str) -> str:
    if any(word in text for word in ("下午", "傍晚")):
        return "afternoon"
    if any(word in text for word in ("晚上", "晚间")):
        return "evening"
    if "中午" in text:
        return "noon"
    if any(word in text for word in ("早上", "上午", "早晨", "凌晨")):
        return "morning"
    return "unspecified"


def _validate_future_time(value: datetime, reference: datetime) -> None:
    if value <= reference:
        raise TimeNormalizationError("time_in_past")
