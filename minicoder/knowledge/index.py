"""SQLite FTS5 知识索引：增量更新后通过临时数据库原子发布。"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import threading
from pathlib import Path

from .loader import KnowledgeLoader, split_document
from .models import IndexBuildReport, KnowledgeChunk, KnowledgeIndexError
from .tokenizer import JiebaTokenizer

SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    title TEXT NOT NULL,
    section TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT NOT NULL,
    component TEXT NOT NULL,
    vehicle_models_json TEXT NOT NULL,
    source_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    version TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS knowledge_chunk_models (
    chunk_id TEXT NOT NULL REFERENCES knowledge_chunks(chunk_id) ON DELETE CASCADE,
    vehicle_model TEXT NOT NULL,
    PRIMARY KEY (chunk_id, vehicle_model)
);
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    chunk_id UNINDEXED,
    title_tokens,
    content_tokens,
    tokenize='unicode61'
);
CREATE INDEX IF NOT EXISTS knowledge_chunks_document_id
    ON knowledge_chunks(document_id);
CREATE INDEX IF NOT EXISTS knowledge_chunks_filters
    ON knowledge_chunks(category, component);
CREATE INDEX IF NOT EXISTS knowledge_chunk_models_model
    ON knowledge_chunk_models(vehicle_model);
"""


class KnowledgeIndex:
    def __init__(
        self,
        loader: KnowledgeLoader,
        index_path: str | Path,
        tokenizer: JiebaTokenizer,
        *,
        max_chunks: int = 10_000,
    ) -> None:
        self.loader = loader
        self.index_path = Path(index_path).resolve()
        self.tokenizer = tokenizer
        self.max_chunks = max_chunks
        self._lock = threading.RLock()

    def build(self) -> IndexBuildReport:
        with self._lock:
            documents = self.loader.load()
            chunks = [chunk for document in documents for chunk in split_document(document)]
            if len(chunks) > self.max_chunks:
                raise KnowledgeIndexError(f"知识分块数量超过限制：{self.max_chunks}")
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self._new_temp_path()
            rebuilt = False
            warnings: list[str] = []
            try:
                if self.index_path.is_file() and self._is_valid_database(self.index_path):
                    shutil.copy2(self.index_path, temp_path)
                else:
                    rebuilt = True
                    if self.index_path.exists():
                        warnings.append("existing_index_corrupted_rebuilt")
                report = self._update_database(
                    temp_path,
                    documents=len(documents),
                    chunks=chunks,
                    rebuilt=rebuilt,
                    warnings=tuple(warnings),
                )
                os.replace(temp_path, self.index_path)
                return report
            except sqlite3.Error as error:
                raise KnowledgeIndexError(f"知识索引构建失败：{error}") from error
            finally:
                temp_path.unlink(missing_ok=True)

    def connect_readonly(self) -> sqlite3.Connection:
        if not self.index_path.is_file():
            raise KnowledgeIndexError("知识索引不存在，请先构建索引")
        try:
            connection = sqlite3.connect(
                f"file:{self.index_path.as_posix()}?mode=ro",
                uri=True,
                timeout=5,
            )
        except sqlite3.Error as error:
            raise KnowledgeIndexError(f"无法打开知识索引：{error}") from error
        connection.row_factory = sqlite3.Row
        return connection

    def _update_database(
        self,
        path: Path,
        *,
        documents: int,
        chunks: list[KnowledgeChunk],
        rebuilt: bool,
        warnings: tuple[str, ...],
    ) -> IndexBuildReport:
        connection = sqlite3.connect(path)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.executescript(_SCHEMA)
            version = connection.execute(
                "SELECT value FROM knowledge_meta WHERE key='schema_version'"
            ).fetchone()
            if version is not None and int(version[0]) != SCHEMA_VERSION:
                raise KnowledgeIndexError(f"不支持的知识索引版本：{version[0]}")
            existing = {
                row[0]: row[1]
                for row in connection.execute(
                    "SELECT chunk_id, content_hash FROM knowledge_chunks"
                ).fetchall()
            }
            desired = {chunk.chunk_id: chunk for chunk in chunks}
            deleted_ids = sorted(existing.keys() - desired.keys())
            changed = [
                chunk for chunk in chunks if existing.get(chunk.chunk_id) != chunk.content_hash
            ]
            added = sum(chunk.chunk_id not in existing for chunk in changed)
            updated = len(changed) - added
            unchanged = len(chunks) - len(changed)
            with connection:
                for chunk_id in deleted_ids:
                    self._delete_chunk(connection, chunk_id)
                for chunk in changed:
                    self._delete_chunk(connection, chunk.chunk_id)
                    self._insert_chunk(connection, chunk)
                connection.execute(
                    "INSERT OR REPLACE INTO knowledge_meta(key, value) VALUES('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
                connection.execute(
                    "INSERT OR REPLACE INTO knowledge_meta(key, value) VALUES('tokenizer', ?)",
                    ("jieba.cut_for_search",),
                )
            integrity = connection.execute("PRAGMA quick_check").fetchone()
            if integrity is None or integrity[0] != "ok":
                raise KnowledgeIndexError("知识索引完整性检查失败")
        finally:
            connection.close()
        return IndexBuildReport(
            documents=documents,
            chunks=len(chunks),
            added=added,
            updated=updated,
            deleted=len(deleted_ids),
            unchanged=unchanged,
            rebuilt_from_scratch=rebuilt,
            warnings=warnings,
        )

    def _insert_chunk(self, connection: sqlite3.Connection, chunk: KnowledgeChunk) -> None:
        connection.execute(
            """
            INSERT INTO knowledge_chunks(
                chunk_id, document_id, title, section, content, category, component,
                vehicle_models_json, source_path, content_hash, version, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.title,
                chunk.section,
                chunk.content,
                chunk.category,
                chunk.component,
                json.dumps(chunk.vehicle_models, ensure_ascii=False),
                chunk.source_path,
                chunk.content_hash,
                chunk.version,
                chunk.updated_at,
            ),
        )
        connection.executemany(
            "INSERT INTO knowledge_chunk_models(chunk_id, vehicle_model) VALUES (?, ?)",
            [(chunk.chunk_id, model) for model in chunk.vehicle_models],
        )
        connection.execute(
            "INSERT INTO knowledge_fts(chunk_id, title_tokens, content_tokens) VALUES (?, ?, ?)",
            (
                chunk.chunk_id,
                self.tokenizer.for_index(f"{chunk.title} {chunk.section}"),
                self.tokenizer.for_index(chunk.content),
            ),
        )

    @staticmethod
    def _delete_chunk(connection: sqlite3.Connection, chunk_id: str) -> None:
        connection.execute("DELETE FROM knowledge_fts WHERE chunk_id = ?", (chunk_id,))
        connection.execute("DELETE FROM knowledge_chunks WHERE chunk_id = ?", (chunk_id,))

    @staticmethod
    def _is_valid_database(path: Path) -> bool:
        try:
            connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=2)
            try:
                row = connection.execute("PRAGMA quick_check").fetchone()
                version = connection.execute(
                    "SELECT value FROM knowledge_meta WHERE key='schema_version'"
                ).fetchone()
                chunk_count = connection.execute("SELECT COUNT(*) FROM knowledge_chunks").fetchone()
                fts_count = connection.execute("SELECT COUNT(*) FROM knowledge_fts").fetchone()
                return (
                    row is not None
                    and row[0] == "ok"
                    and version is not None
                    and int(version[0]) == SCHEMA_VERSION
                    and chunk_count is not None
                    and fts_count is not None
                    and chunk_count[0] == fts_count[0]
                )
            finally:
                connection.close()
        except (sqlite3.Error, OSError):
            return False

    def _new_temp_path(self) -> Path:
        handle, raw_path = tempfile.mkstemp(
            prefix=f".{self.index_path.name}.",
            suffix=".tmp",
            dir=self.index_path.parent,
        )
        os.close(handle)
        path = Path(raw_path)
        path.unlink(missing_ok=True)
        return path
