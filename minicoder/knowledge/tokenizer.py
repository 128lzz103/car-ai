"""jieba 中文检索分词；索引和查询必须共用同一规范化流程。"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable

from .models import KnowledgeIndexError

_TOKEN_RE = re.compile(r"[A-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff]+")
_DEFAULT_TERMS = (
    "动力电池",
    "剩余续航",
    "充电桩",
    "快充",
    "慢充",
    "热管理",
    "故障灯",
    "制动能量回收",
    "长途充电",
)
_STOP_WORDS = frozenset(
    {"的", "了", "和", "与", "及", "是", "在", "不", "请", "一下", "怎么", "完全", "存在"}
)


class JiebaTokenizer:
    def __init__(
        self,
        *,
        domain_terms: Iterable[str] = _DEFAULT_TERMS,
        stop_words: frozenset[str] = _STOP_WORDS,
        max_tokens: int = 128,
    ) -> None:
        try:
            import jieba
        except ImportError as error:
            raise KnowledgeIndexError(
                '中文知识检索需要 jieba；请安装 pip install -e ".[automotive]"'
            ) from error
        jieba.setLogLevel(logging.WARNING)
        self._tokenizer = jieba.Tokenizer()
        for term in domain_terms:
            normalized = self.normalize(term)
            if normalized:
                self._tokenizer.add_word(normalized)
        self.stop_words = stop_words
        self.max_tokens = max_tokens

    @staticmethod
    def normalize(text: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", text).lower().split())

    def tokenize(self, text: str) -> tuple[str, ...]:
        normalized = self.normalize(text)
        if not normalized:
            return ()
        tokens: list[str] = []
        seen: set[str] = set()
        for raw in self._tokenizer.cut_for_search(normalized):
            token = raw.strip()
            if (
                not token
                or token in self.stop_words
                or token in seen
                or _TOKEN_RE.fullmatch(token) is None
            ):
                continue
            seen.add(token)
            tokens.append(token)
            if len(tokens) >= self.max_tokens:
                break
        return tuple(tokens)

    def for_index(self, text: str) -> str:
        return " ".join(self.tokenize(text))

    def for_match(self, text: str) -> str:
        tokens = self.tokenize(text)
        return " OR ".join(f'"{token}"' for token in tokens)
