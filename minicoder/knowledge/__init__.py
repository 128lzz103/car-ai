"""本地汽车知识库：安全文档加载、中文 BM25 检索和可扩展 Retriever 协议。"""

from .index import KnowledgeIndex
from .loader import KnowledgeLoader, split_document
from .models import (
    IndexBuildReport,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeError,
    KnowledgeFilters,
    KnowledgeIndexError,
    KnowledgeSearchResult,
    KnowledgeSecurityError,
)
from .retriever import BM25Retriever, KnowledgeRetriever
from .tokenizer import JiebaTokenizer

__all__ = [
    "BM25Retriever",
    "IndexBuildReport",
    "JiebaTokenizer",
    "KnowledgeChunk",
    "KnowledgeDocument",
    "KnowledgeError",
    "KnowledgeFilters",
    "KnowledgeIndex",
    "KnowledgeIndexError",
    "KnowledgeLoader",
    "KnowledgeRetriever",
    "KnowledgeSearchResult",
    "KnowledgeSecurityError",
    "split_document",
]
