"""知识库领域对象与错误类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class KnowledgeError(RuntimeError):
    pass


class KnowledgeSecurityError(KnowledgeError):
    pass


class KnowledgeIndexError(KnowledgeError):
    pass


@dataclass(frozen=True)
class KnowledgeDocument:
    document_id: str
    title: str
    content: str
    category: str
    component: str
    vehicle_models: tuple[str, ...]
    source_path: str
    version: str = "1.0"
    updated_at: str = ""


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    document_id: str
    title: str
    section: str
    content: str
    category: str
    component: str
    vehicle_models: tuple[str, ...]
    source_path: str
    content_hash: str
    version: str = "1.0"
    updated_at: str = ""


@dataclass(frozen=True)
class KnowledgeFilters:
    vehicle_models: tuple[str, ...] = ()
    category: str | None = None
    component: str | None = None


@dataclass(frozen=True)
class KnowledgeSearchResult:
    chunk_id: str
    document_id: str
    content: str
    title: str
    section: str
    citation: str
    metadata: dict[str, Any]
    rank: int
    score: float
    source: str = "bm25"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "title": self.title,
            "section": self.section,
            "citation": self.citation,
            "metadata": dict(self.metadata),
            "rank": self.rank,
            "score": self.score,
            "source": self.source,
        }


@dataclass(frozen=True)
class IndexBuildReport:
    documents: int
    chunks: int
    added: int
    updated: int
    deleted: int
    unchanged: int
    rebuilt_from_scratch: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "added": self.added,
            "updated": self.updated,
            "deleted": self.deleted,
            "unchanged": self.unchanged,
            "rebuilt_from_scratch": self.rebuilt_from_scratch,
            "warnings": list(self.warnings),
        }
