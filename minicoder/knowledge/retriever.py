"""知识检索接口与第一阶段 BM25 实现。"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from typing import Protocol

from .index import KnowledgeIndex
from .models import KnowledgeFilters, KnowledgeIndexError, KnowledgeSearchResult


class KnowledgeRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        filters: KnowledgeFilters | None = None,
        top_k: int = 5,
    ) -> list[KnowledgeSearchResult]: ...


class BM25Retriever:
    def __init__(self, index: KnowledgeIndex) -> None:
        self.index = index

    def search(
        self,
        query: str,
        *,
        filters: KnowledgeFilters | None = None,
        top_k: int = 5,
    ) -> list[KnowledgeSearchResult]:
        query = query.strip()
        if not query:
            return []
        if len(query) > 500:
            raise ValueError("知识查询不能超过 500 个字符")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k 必须在 1～20 之间")
        query_tokens = self.index.tokenizer.tokenize(query)
        expression = self.index.tokenizer.for_match(query)
        if not expression:
            return []
        selected = filters or KnowledgeFilters()
        clauses = ["knowledge_fts MATCH ?"]
        parameters: list[object] = [expression]
        if selected.category:
            clauses.append("chunks.category = ?")
            parameters.append(selected.category)
        if selected.component:
            clauses.append("chunks.component = ?")
            parameters.append(selected.component)
        if selected.vehicle_models:
            placeholders = ",".join("?" for _ in selected.vehicle_models)
            clauses.append(
                "EXISTS (SELECT 1 FROM knowledge_chunk_models models "
                "WHERE models.chunk_id = chunks.chunk_id "
                f"AND models.vehicle_model IN ({placeholders}))"
            )
            parameters.extend(model.upper() for model in selected.vehicle_models)
        candidate_limit = min(100, max(top_k * 5, 20))
        parameters.append(candidate_limit)
        sql = f"""
            SELECT chunks.*, bm25(knowledge_fts, 0.0, 5.0, 1.0) AS raw_score
            FROM knowledge_fts
            JOIN knowledge_chunks chunks ON chunks.chunk_id = knowledge_fts.chunk_id
            WHERE {" AND ".join(clauses)}
            ORDER BY raw_score ASC, chunks.chunk_id ASC
            LIMIT ?
        """
        try:
            with closing(self.index.connect_readonly()) as connection:
                rows = connection.execute(sql, parameters).fetchall()
        except sqlite3.Error as error:
            raise KnowledgeIndexError(f"知识检索失败：{error}") from error
        selected_rows = [row for row in rows if self._has_sufficient_overlap(row, query_tokens)][
            :top_k
        ]
        return [self._result(row, rank) for rank, row in enumerate(selected_rows, start=1)]

    def _has_sufficient_overlap(self, row: sqlite3.Row, query_tokens: tuple[str, ...]) -> bool:
        document_tokens = set(
            self.index.tokenizer.tokenize(f"{row['title']} {row['section']} {row['content']}")
        )
        matches = sum(token in document_tokens for token in query_tokens)
        return matches >= min(2, len(query_tokens))

    @staticmethod
    def _result(row: sqlite3.Row, rank: int) -> KnowledgeSearchResult:
        models = json.loads(row["vehicle_models_json"])
        return KnowledgeSearchResult(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            content=row["content"],
            title=row["title"],
            section=row["section"],
            citation=f"{row['source_path']}#{row['section']}",
            metadata={
                "category": row["category"],
                "component": row["component"],
                "vehicle_models": models,
                "version": row["version"],
                "updated_at": row["updated_at"],
            },
            rank=rank,
            score=round(-float(row["raw_score"]), 6),
        )
