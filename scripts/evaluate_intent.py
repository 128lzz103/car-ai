"""对汽车意图预测 JSONL 计算准确率、实体 F1 和安全失败率。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        item_id = str(item.get("id", ""))
        if not item_id or item_id in rows:
            raise ValueError(f"{path}:{line_number}: id 缺失或重复")
        rows[item_id] = item
    return rows


def _normalized(name: str, value: Any) -> Any:
    if name == "vehicle_id" and isinstance(value, str):
        return value.upper()
    if name == "target_temperature":
        return float(value)
    if name == "departure_time" and isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo:
            return parsed.astimezone(timezone.utc).isoformat()
    return value


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(
    expected_rows: dict[str, dict[str, Any]],
    predicted_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    missing_predictions = sorted(expected_rows.keys() - predicted_rows.keys())
    if missing_predictions:
        raise ValueError(f"缺少预测: {', '.join(missing_predictions)}")

    intent_correct = exact_correct = entity_tp = entity_fp = entity_fn = 0
    missing_tp = missing_fp = missing_fn = unsafe_ready = 0
    for item_id, expected in expected_rows.items():
        predicted = predicted_rows[item_id]
        expected_intent = expected.get("intent")
        predicted_intent = predicted.get("intent")
        if expected_intent == predicted_intent:
            intent_correct += 1

        expected_entities = expected.get("entities") or {}
        predicted_entities = predicted.get("entities") or {}
        all_names = expected_entities.keys() | predicted_entities.keys()
        entities_match = True
        for name in all_names:
            expected_has = name in expected_entities
            predicted_has = name in predicted_entities
            equal = (
                expected_has
                and predicted_has
                and _normalized(name, expected_entities[name])
                == _normalized(name, predicted_entities[name])
            )
            if equal:
                entity_tp += 1
            else:
                entities_match = False
                entity_fp += int(predicted_has)
                entity_fn += int(expected_has)

        expected_missing = set(expected.get("missing_fields") or [])
        predicted_missing = set(predicted.get("missing_fields") or [])
        missing_tp += len(expected_missing & predicted_missing)
        missing_fp += len(predicted_missing - expected_missing)
        missing_fn += len(expected_missing - predicted_missing)
        if expected_missing and predicted.get("status") == "ready":
            unsafe_ready += 1
        if expected_intent == predicted_intent and entities_match:
            exact_correct += 1

    total = len(expected_rows)
    entity_precision = _ratio(entity_tp, entity_tp + entity_fp)
    entity_recall = _ratio(entity_tp, entity_tp + entity_fn)
    missing_precision = _ratio(missing_tp, missing_tp + missing_fp)
    missing_recall = _ratio(missing_tp, missing_tp + missing_fn)
    return {
        "samples": total,
        "intent_accuracy": _ratio(intent_correct, total),
        "exact_match": _ratio(exact_correct, total),
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "entity_f1": _ratio(2 * entity_precision * entity_recall, entity_precision + entity_recall),
        "missing_field_precision": missing_precision,
        "missing_field_recall": missing_recall,
        "missing_field_f1": _ratio(
            2 * missing_precision * missing_recall, missing_precision + missing_recall
        ),
        "unsafe_ready_rate": _ratio(unsafe_ready, total),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    metrics = evaluate(_load(args.expected), _load(args.predicted))
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
