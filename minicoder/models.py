"""模型级上下文窗口与本地 token 编码配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    context_window: int
    output_reserve: int
    chars_per_token: float
    cjk_chars_per_token: float
    message_overhead: int = 4


_DEFAULT_PROFILE = ModelProfile(128_000, 8_192, 4.0, 1.2)
_MODEL_PROFILES: tuple[tuple[tuple[str, ...], ModelProfile], ...] = (
    (("gpt-5",), ModelProfile(400_000, 16_384, 4.2, 1.6)),
    (("gpt-4.1",), ModelProfile(1_047_576, 16_384, 4.2, 1.6)),
    (("gpt-4o", "o1", "o3", "o4"), ModelProfile(128_000, 16_384, 4.2, 1.6)),
    (("claude-",), ModelProfile(200_000, 16_384, 3.8, 1.2)),
)


def get_model_profile(model: str) -> ModelProfile:
    normalized = (model or "").strip().lower()
    for prefixes, profile in _MODEL_PROFILES:
        if normalized.startswith(prefixes):
            return profile
    return _DEFAULT_PROFILE
