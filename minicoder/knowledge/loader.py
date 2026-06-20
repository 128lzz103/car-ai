"""受限目录内的结构化知识清单加载与 Markdown 标题感知分块。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .models import KnowledgeChunk, KnowledgeDocument, KnowledgeError, KnowledgeSecurityError

_DOCUMENT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_METADATA_RE = re.compile(r"^[A-Za-z0-9_\-\u3400-\u4dbf\u4e00-\u9fff]{1,64}$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class KnowledgeLoader:
    def __init__(
        self,
        root: str | Path,
        *,
        max_file_bytes: int = 512_000,
        max_documents: int = 100,
    ) -> None:
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise KnowledgeError(f"知识库目录不存在：{self.root}")
        self.max_file_bytes = max_file_bytes
        self.max_documents = max_documents

    def load(self) -> list[KnowledgeDocument]:
        manifest_path = self._resolve_file("manifest.json", suffix=".json")
        payload = self._read_json(manifest_path)
        if payload.get("schema_version") != 1 or not isinstance(payload.get("documents"), list):
            raise KnowledgeError("知识库 manifest.json 必须使用 schema_version=1 和 documents 数组")
        entries = payload["documents"]
        if len(entries) > self.max_documents:
            raise KnowledgeError(f"知识文档数量超过限制：{self.max_documents}")
        documents: list[KnowledgeDocument] = []
        seen_ids: set[str] = set()
        seen_paths: set[str] = set()
        for raw in entries:
            if not isinstance(raw, dict):
                raise KnowledgeError("manifest documents 每一项必须是对象")
            document = self._load_document(raw)
            if document.document_id in seen_ids:
                raise KnowledgeError(f"重复 document_id：{document.document_id}")
            if document.source_path in seen_paths:
                raise KnowledgeError(f"重复知识文档路径：{document.source_path}")
            seen_ids.add(document.document_id)
            seen_paths.add(document.source_path)
            documents.append(document)
        return documents

    def _load_document(self, raw: dict[str, Any]) -> KnowledgeDocument:
        document_id = _required_string(raw, "document_id", pattern=_DOCUMENT_ID_RE)
        source_path = _required_string(raw, "path", max_length=240)
        path = self._resolve_file(source_path, suffix=".md")
        title = _required_string(raw, "title", max_length=120)
        category = _required_string(raw, "category", pattern=_METADATA_RE)
        component = _required_string(raw, "component", pattern=_METADATA_RE)
        models = raw.get("vehicle_models", [])
        if not isinstance(models, list) or not all(isinstance(item, str) for item in models):
            raise KnowledgeError(f"{document_id}.vehicle_models 必须是字符串数组")
        vehicle_models = tuple(
            dict.fromkeys(item.strip().upper() for item in models if item.strip())
        )
        content = self._read_text(path)
        return KnowledgeDocument(
            document_id=document_id,
            title=title,
            content=content,
            category=category,
            component=component,
            vehicle_models=vehicle_models,
            source_path=source_path.replace("\\", "/"),
            version=str(raw.get("version", "1.0"))[:32],
            updated_at=str(raw.get("updated_at", ""))[:32],
        )

    def _resolve_file(self, raw_path: str, *, suffix: str) -> Path:
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != suffix:
            raise KnowledgeSecurityError(f"非法知识库路径：{raw_path}")
        candidate = self.root / relative
        current = self.root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise KnowledgeSecurityError(f"知识库不允许符号链接：{raw_path}")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.root)
        except (FileNotFoundError, ValueError) as error:
            raise KnowledgeSecurityError(f"知识文件不存在或逃逸配置目录：{raw_path}") from error
        if not resolved.is_file():
            raise KnowledgeSecurityError(f"知识路径不是文件：{raw_path}")
        if resolved.stat().st_size > self.max_file_bytes:
            raise KnowledgeError(f"知识文件超过 {self.max_file_bytes} 字节限制：{raw_path}")
        return resolved

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise KnowledgeError(f"知识清单不是有效 UTF-8 JSON：{path.name}") from error
        if not isinstance(value, dict):
            raise KnowledgeError("知识清单根节点必须是对象")
        return value

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            content = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError as error:
            raise KnowledgeError(f"知识文档不是有效 UTF-8：{path.name}") from error
        if not content:
            raise KnowledgeError(f"知识文档不能为空：{path.name}")
        return content


def split_document(
    document: KnowledgeDocument,
    *,
    max_chars: int = 900,
    overlap_chars: int = 100,
) -> list[KnowledgeChunk]:
    if max_chars < 200 or overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("分块参数无效")
    sections: list[tuple[str, str]] = []
    current_title = document.title
    lines: list[str] = []
    for line in document.content.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            body = "\n".join(lines).strip()
            if body:
                sections.append((current_title, body))
            current_title = match.group(2).strip()[:120]
            lines = []
        else:
            lines.append(line)
    body = "\n".join(lines).strip()
    if body:
        sections.append((current_title, body))
    if not sections:
        sections = [(document.title, document.content)]

    chunks: list[KnowledgeChunk] = []
    for section_index, (section, content) in enumerate(sections):
        for part_index, part in enumerate(_split_text(content, max_chars, overlap_chars)):
            chunk_id = f"{document.document_id}:{section_index:03d}:{part_index:03d}"
            digest_payload = json.dumps(
                {
                    "content": part,
                    "section": section,
                    "category": document.category,
                    "component": document.component,
                    "vehicle_models": document.vehicle_models,
                    "version": document.version,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    title=document.title,
                    section=section,
                    content=part,
                    category=document.category,
                    component=document.component,
                    vehicle_models=document.vehicle_models,
                    source_path=document.source_path,
                    content_hash=hashlib.sha256(digest_payload).hexdigest(),
                    version=document.version,
                    updated_at=document.updated_at,
                )
            )
    return chunks


def _split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if len(normalized) <= max_chars:
        return [normalized]
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + max_chars)
        if end < len(normalized):
            candidates = [
                normalized.rfind(mark, start + max_chars // 2, end) for mark in ("\n", "。", "；")
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + 1
        chunks.append(normalized[start:end].strip())
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap_chars)
    return [item for item in chunks if item]


def _required_string(
    value: dict[str, Any],
    name: str,
    *,
    max_length: int = 120,
    pattern: re.Pattern[str] | None = None,
) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item.strip():
        raise KnowledgeError(f"manifest 字段 {name} 必须是非空字符串")
    item = item.strip()
    if len(item) > max_length or (pattern is not None and pattern.fullmatch(item) is None):
        raise KnowledgeError(f"manifest 字段 {name} 格式无效")
    return item
