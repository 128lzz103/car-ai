"""汽车意图离线评测脚本测试。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def test_evaluation_script_scores_normalized_predictions():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_intent.py"),
            "--expected",
            str(FIXTURES / "vehicle_intents.jsonl"),
            "--predicted",
            str(FIXTURES / "vehicle_intent_predictions.example.jsonl"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads(result.stdout)
    assert metrics["samples"] == 5
    assert metrics["intent_accuracy"] == 1.0
    assert metrics["entity_f1"] == 1.0
    assert metrics["missing_field_recall"] == 1.0
    assert metrics["unsafe_ready_rate"] == 0.0
