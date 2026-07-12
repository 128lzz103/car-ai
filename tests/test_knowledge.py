"""中文知识加载、FTS5 索引和 BM25 检索测试。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from minicoder.knowledge import (
    BM25Retriever,
    JiebaTokenizer,
    KnowledgeError,
    KnowledgeFilters,
    KnowledgeIndex,
    KnowledgeLoader,
    KnowledgeSecurityError,
    split_document,
)
from minicoder.knowledge.service import packaged_knowledge_dir
from scripts.evaluate_knowledge import evaluate, load_examples


def _write_library(root: Path, content: str = "动力电池需要定期保养。") -> Path:
    root.mkdir()
    manifest = {
        "schema_version": 1,
        "documents": [
            {
                "document_id": "battery",
                "path": "battery.md",
                "title": "电池指南",
                "category": "maintenance",
                "component": "traction_battery",
                "vehicle_models": ["MC-EV1"],
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (root / "battery.md").write_text(f"# 电池指南\n\n## 保养\n\n{content}", encoding="utf-8")
    return root


def _retriever(tmp_path: Path, source: Path | None = None):
    tokenizer = JiebaTokenizer()
    index = KnowledgeIndex(
        KnowledgeLoader(source or packaged_knowledge_dir()),
        tmp_path / "knowledge.db",
        tokenizer,
    )
    report = index.build()
    return BM25Retriever(index), index, report


def test_jieba_search_mode_tokenizes_chinese_domain_terms():
    tokenizer = JiebaTokenizer()

    tokens = tokenizer.tokenize("动力电池的日常保养建议")

    assert "动力电池" in tokens
    assert "保养" in tokens
    assert tokenizer.for_match('电池保养 OR "异常"')


def test_packaged_documents_load_and_split_by_heading():
    documents = KnowledgeLoader(packaged_knowledge_dir()).load()
    chunks = [chunk for document in documents for chunk in split_document(document)]

    assert len(documents) == 5
    assert {item.document_id for item in documents} >= {"battery-care", "trip-planning"}
    assert any(chunk.section == "日常充电建议" for chunk in chunks)
    assert all(len(chunk.content_hash) == 64 for chunk in chunks)


def test_bm25_retrieves_chinese_and_applies_metadata_filters(tmp_path: Path):
    retriever, _index, report = _retriever(tmp_path)

    battery = retriever.search("电池保养", top_k=3)
    filtered = retriever.search(
        "警告灯怎么处理",
        filters=KnowledgeFilters(vehicle_models=("MC-EV1",), category="warning"),
    )
    excluded = retriever.search(
        "警告灯怎么处理",
        filters=KnowledgeFilters(vehicle_models=("MC-EV2",), category="warning"),
    )

    assert report.documents == 5
    assert battery[0].document_id == "battery-care"
    assert battery[0].citation.startswith("battery-care.md#")
    assert all(item.rank == index for index, item in enumerate(battery, start=1))
    assert filtered[0].document_id == "warning-lights"
    assert excluded == []


def test_index_update_is_incremental_and_atomically_replaced(tmp_path: Path):
    source = _write_library(tmp_path / "source")
    retriever, index, first = _retriever(tmp_path, source)
    original_inode = index.index_path.stat().st_ino

    second = index.build()
    (source / "battery.md").write_text(
        "# 电池指南\n\n## 保养\n\n动力电池保养时应避免长期极低电量停放。",
        encoding="utf-8",
    )
    third = index.build()

    assert first.added == 1
    assert second.unchanged == 1
    assert second.added == second.updated == second.deleted == 0
    assert third.updated == 1
    assert index.index_path.stat().st_ino != original_inode
    assert retriever.search("极低电量")[0].document_id == "battery"


def test_corrupted_index_is_rebuilt(tmp_path: Path):
    source = _write_library(tmp_path / "source")
    tokenizer = JiebaTokenizer()
    path = tmp_path / "knowledge.db"
    path.write_bytes(b"not sqlite")
    index = KnowledgeIndex(KnowledgeLoader(source), path, tokenizer)

    report = index.build()

    assert report.rebuilt_from_scratch is True
    assert "existing_index_corrupted_rebuilt" in report.warnings
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"


def test_loader_rejects_escape_and_oversized_files(tmp_path: Path):
    root = tmp_path / "knowledge"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "documents": [
            {
                "document_id": "escape",
                "path": "../outside.md",
                "title": "Escape",
                "category": "test",
                "component": "test",
                "vehicle_models": [],
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(KnowledgeSecurityError):
        KnowledgeLoader(root).load()

    safe = _write_library(tmp_path / "safe", "x" * 100)
    with pytest.raises(KnowledgeError, match="超过"):
        KnowledgeLoader(safe, max_file_bytes=20).load()


def test_search_handles_empty_unknown_and_fts_syntax_text(tmp_path: Path):
    retriever, _index, _report = _retriever(tmp_path)

    assert retriever.search("") == []
    assert retriever.search("完全不存在的火星曲速引擎") == []
    assert isinstance(retriever.search('电池 OR "保养" NOT *'), list)
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("电池", top_k=0)
    with pytest.raises(ValueError, match="500"):
        retriever.search("电" * 501)


def test_knowledge_evaluation_reports_recall_mrr_and_no_answer(tmp_path: Path):
    retriever, _index, _report = _retriever(tmp_path)
    examples = load_examples(Path("tests/fixtures/vehicle_knowledge_queries.jsonl"))

    metrics = evaluate(retriever, examples, top_k=3)

    assert metrics["samples"] == 11
    assert metrics["recall_at_3"] >= 0.9
    assert metrics["mrr"] >= 0.8
    assert metrics["no_answer_accuracy"] == 1.0
