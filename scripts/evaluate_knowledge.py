"""对本地知识 Retriever 计算 Recall@K、MRR 与无答案拒答率。"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from minicoder.knowledge import (
    BM25Retriever,
    JiebaTokenizer,
    KnowledgeFilters,
    KnowledgeIndex,
    KnowledgeLoader,
)
from minicoder.knowledge.service import packaged_knowledge_dir


def load_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number}: 每行必须是 JSON 对象")
        item_id = str(item.get("id", ""))
        query = item.get("query")
        if not item_id or item_id in seen or not isinstance(query, str) or not query.strip():
            raise ValueError(f"{path}:{line_number}: id 缺失/重复或 query 无效")
        expected = item.get("expected_document_id")
        if expected is not None and not isinstance(expected, str):
            raise ValueError(f"{path}:{line_number}: expected_document_id 必须是字符串或 null")
        seen.add(item_id)
        examples.append(item)
    return examples


def evaluate(
    retriever: BM25Retriever,
    examples: list[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    answerable = recall_hits = 0
    reciprocal_rank = 0.0
    no_answer = no_answer_correct = 0
    cases: list[dict[str, Any]] = []
    for example in examples:
        filters = KnowledgeFilters(
            vehicle_models=tuple(example.get("vehicle_models") or ()),
            category=example.get("category"),
            component=example.get("component"),
        )
        results = retriever.search(example["query"], filters=filters, top_k=top_k)
        retrieved = [item.document_id for item in results]
        expected = example.get("expected_document_id")
        rank = None
        if expected is None:
            no_answer += 1
            no_answer_correct += int(not results)
        else:
            answerable += 1
            if expected in retrieved:
                rank = retrieved.index(expected) + 1
                recall_hits += 1
                reciprocal_rank += 1 / rank
        cases.append(
            {
                "id": example["id"],
                "expected_document_id": expected,
                "retrieved_document_ids": retrieved,
                "expected_rank": rank,
            }
        )
    return {
        "samples": len(examples),
        f"recall_at_{top_k}": _ratio(recall_hits, answerable),
        "mrr": _ratio(reciprocal_rank, answerable),
        "no_answer_accuracy": _ratio(no_answer_correct, no_answer),
        "answerable_samples": answerable,
        "no_answer_samples": no_answer,
        "cases": cases,
    }


def _ratio(numerator: float, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评测本地汽车知识检索")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--knowledge-dir", type=Path, default=packaged_knowledge_dir())
    parser.add_argument("--index", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--min-mrr", type=float, default=0.0)
    args = parser.parse_args(argv)
    if not 1 <= args.top_k <= 20:
        parser.error("--top-k 必须在 1～20 之间")
    examples = load_examples(args.dataset)
    with tempfile.TemporaryDirectory(prefix="minicoder-knowledge-eval-") as temporary:
        index_path = args.index or Path(temporary) / "knowledge.db"
        index = KnowledgeIndex(
            KnowledgeLoader(args.knowledge_dir),
            index_path,
            JiebaTokenizer(),
        )
        index.build()
        metrics = evaluate(BM25Retriever(index), examples, top_k=args.top_k)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    recall = metrics[f"recall_at_{args.top_k}"]
    return int(recall < args.min_recall or metrics["mrr"] < args.min_mrr)


if __name__ == "__main__":
    raise SystemExit(main())
