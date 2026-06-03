"""配置加载 —— 环境变量 + 项目根 .env 文件。

配置两个 provider 各自的 key/base_url,加上模型与轮数覆盖。
没有引入 python-dotenv,自己解析 .env,保持零额外依赖、可读。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .intent import get_timezone
from .models import get_model_profile
from .security import PermissionMode
from .vehicle.models import normalize_vehicle_id

# provider -> 默认模型
_DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5",
}
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _load_dotenv(path: Path) -> None:
    """把 .env 里的键值读进 os.environ(不覆盖已存在的真实环境变量)。"""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass
class Config:
    provider: str
    model: str
    api_key: str
    base_url: str | None
    max_rounds: int
    permission_mode: str = "ask"
    read_only: bool = False
    context_window: int = 128_000
    output_reserve: int = 8_192
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 8.0
    retry_max_wait: float = 60.0
    autosave: bool = True
    autosave_name: str = "autosave"
    application_profile: str = "coding"
    intent_model: str = ""
    timezone: str = "Asia/Shanghai"
    intent_rule_fast_path: bool = True
    vehicle_api_url: str = "http://127.0.0.1:8765"
    vehicle_api_token: str = "local-demo-token"
    vehicle_api_timeout: float = 5.0
    vehicle_api_allow_remote: bool = False
    vehicle_permission_mode: str = "ask"
    vehicle_allowed_ids: tuple[str, ...] = ()
    knowledge_dir: str = ""
    knowledge_index: str = ".minicoder/knowledge.db"
    knowledge_top_k: int = 5
    knowledge_auto_rebuild: bool = True
    planner_mode: str = "rules"
    planner_model: str = ""

    @classmethod
    def load(cls, dotenv_path: str | Path | None = None) -> Config:
        # 先加载 .env(默认当前工作目录),真实环境变量优先
        _load_dotenv(Path(dotenv_path) if dotenv_path else Path.cwd() / ".env")

        provider = os.environ.get("MINICODER_PROVIDER", "openai").strip().lower()
        if provider not in _DEFAULT_MODELS:
            raise ValueError(f"未知 provider: {provider!r},支持 {list(_DEFAULT_MODELS)}")

        model = os.environ.get("MINICODER_MODEL", "").strip() or _DEFAULT_MODELS[provider]
        model_profile = get_model_profile(model)

        if provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or _DEFAULT_OPENAI_BASE_URL
        else:  # anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            base_url = (
                os.environ.get("ANTHROPIC_BASE_URL", "").strip() or "https://api.anthropic.com"
            )

        try:
            max_rounds = int(os.environ.get("MINICODER_MAX_ROUNDS", "50"))
        except ValueError:
            max_rounds = 50
        permission_mode = PermissionMode.parse(
            os.environ.get("MINICODER_PERMISSION_MODE", "ask")
        ).value

        def positive_int(name: str, default: int) -> int:
            try:
                value = int(os.environ.get(name, str(default)))
            except ValueError:
                return default
            return value if value > 0 else default

        def nonnegative_int(name: str, default: int) -> int:
            try:
                value = int(os.environ.get(name, str(default)))
            except ValueError:
                return default
            return value if value >= 0 else default

        def nonnegative_float(name: str, default: float) -> float:
            try:
                value = float(os.environ.get(name, str(default)))
            except ValueError:
                return default
            return value if value >= 0 else default

        def positive_float(name: str, default: float) -> float:
            value = nonnegative_float(name, default)
            return value if value > 0 else default

        def boolean(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            normalized = raw.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            raise ValueError(f"{name} 必须是 true/false、yes/no、on/off 或 1/0")

        context_window = positive_int("MINICODER_CONTEXT_WINDOW", model_profile.context_window)
        output_reserve = positive_int("MINICODER_OUTPUT_RESERVE", model_profile.output_reserve)
        if output_reserve >= context_window:
            raise ValueError("MINICODER_OUTPUT_RESERVE 必须小于 MINICODER_CONTEXT_WINDOW")
        max_retries = nonnegative_int("MINICODER_MAX_RETRIES", 3)
        retry_base_delay = nonnegative_float("MINICODER_RETRY_BASE_DELAY", 1.0)
        retry_max_delay = nonnegative_float("MINICODER_RETRY_MAX_DELAY", 8.0)
        retry_max_wait = nonnegative_float("MINICODER_RETRY_MAX_WAIT", 60.0)
        autosave = boolean("MINICODER_AUTOSAVE", True)
        autosave_name = os.environ.get("MINICODER_AUTOSAVE_NAME", "").strip() or "autosave"
        read_only = boolean("MINICODER_READ_ONLY", False)
        application_profile = os.environ.get("MINICODER_PROFILE", "coding").strip().lower()
        if application_profile not in {"coding", "automotive"}:
            raise ValueError("MINICODER_PROFILE 必须是 coding 或 automotive")
        intent_model = os.environ.get("MINICODER_INTENT_MODEL", "").strip() or model
        timezone_name = os.environ.get("MINICODER_TIMEZONE", "").strip() or "Asia/Shanghai"
        get_timezone(timezone_name)
        intent_rule_fast_path = boolean("MINICODER_INTENT_RULE_FAST_PATH", True)
        vehicle_api_url = (
            os.environ.get("MINICODER_VEHICLE_API_URL", "").strip() or "http://127.0.0.1:8765"
        )
        vehicle_api_token = (
            os.environ.get("MINICODER_VEHICLE_API_TOKEN", "").strip() or "local-demo-token"
        )
        vehicle_api_timeout = positive_float("MINICODER_VEHICLE_API_TIMEOUT", 5.0)
        vehicle_api_allow_remote = boolean("MINICODER_VEHICLE_API_ALLOW_REMOTE", False)
        vehicle_permission_mode = PermissionMode.parse(
            os.environ.get("MINICODER_VEHICLE_PERMISSION_MODE", "ask")
        ).value
        raw_allowed_ids = os.environ.get("MINICODER_VEHICLE_ALLOWED_IDS", "").strip()
        vehicle_allowed_ids = tuple(
            normalize_vehicle_id(item) for item in raw_allowed_ids.split(",") if item.strip()
        )
        knowledge_dir = os.environ.get("MINICODER_KNOWLEDGE_DIR", "").strip()
        knowledge_index = (
            os.environ.get("MINICODER_KNOWLEDGE_INDEX", "").strip() or ".minicoder/knowledge.db"
        )
        knowledge_top_k = positive_int("MINICODER_KNOWLEDGE_TOP_K", 5)
        if knowledge_top_k > 20:
            raise ValueError("MINICODER_KNOWLEDGE_TOP_K 必须在 1～20 之间")
        knowledge_auto_rebuild = boolean("MINICODER_KNOWLEDGE_AUTO_REBUILD", True)
        planner_mode = os.environ.get("MINICODER_PLANNER_MODE", "rules").strip().lower()
        if planner_mode not in {"rules", "hybrid"}:
            raise ValueError("MINICODER_PLANNER_MODE 必须是 rules 或 hybrid")
        planner_model = os.environ.get("MINICODER_PLANNER_MODEL", "").strip() or intent_model

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_rounds=max_rounds,
            permission_mode=permission_mode,
            read_only=read_only,
            context_window=context_window,
            output_reserve=output_reserve,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
            retry_max_wait=retry_max_wait,
            autosave=autosave,
            autosave_name=autosave_name,
            application_profile=application_profile,
            intent_model=intent_model,
            timezone=timezone_name,
            intent_rule_fast_path=intent_rule_fast_path,
            vehicle_api_url=vehicle_api_url,
            vehicle_api_token=vehicle_api_token,
            vehicle_api_timeout=vehicle_api_timeout,
            vehicle_api_allow_remote=vehicle_api_allow_remote,
            vehicle_permission_mode=vehicle_permission_mode,
            vehicle_allowed_ids=vehicle_allowed_ids,
            knowledge_dir=knowledge_dir,
            knowledge_index=knowledge_index,
            knowledge_top_k=knowledge_top_k,
            knowledge_auto_rebuild=knowledge_auto_rebuild,
            planner_mode=planner_mode,
            planner_model=planner_model,
        )

    def require_api_key(self) -> None:
        """在真正发起请求前调用;缺 key 时给出清晰指引。"""
        if not self.api_key:
            var = "OPENAI_API_KEY" if self.provider == "openai" else "ANTHROPIC_API_KEY"
            raise RuntimeError(f"缺少 {var}。请设置环境变量或在 .env 中填写(参考 .env.example)。")
