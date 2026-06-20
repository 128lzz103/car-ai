"""从运行配置创建并更新本地知识 Retriever。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .index import KnowledgeIndex
from .loader import KnowledgeLoader
from .retriever import BM25Retriever
from .tokenizer import JiebaTokenizer


def packaged_knowledge_dir() -> Path:
    return Path(__file__).resolve().parent / "data"


def build_knowledge_retriever(config: Any, workspace: Path) -> BM25Retriever:
    raw_source = str(getattr(config, "knowledge_dir", "")).strip()
    source = Path(raw_source) if raw_source else packaged_knowledge_dir()
    if not source.is_absolute():
        source = workspace / source
    raw_index = str(getattr(config, "knowledge_index", ".minicoder/knowledge.db")).strip()
    index_path = Path(raw_index or ".minicoder/knowledge.db")
    if not index_path.is_absolute():
        index_path = workspace / index_path
    index = KnowledgeIndex(KnowledgeLoader(source), index_path, JiebaTokenizer())
    if bool(getattr(config, "knowledge_auto_rebuild", True)):
        index.build()
    return BM25Retriever(index)
