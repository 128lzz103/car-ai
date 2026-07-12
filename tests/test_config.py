"""配置单测:权限模式默认值、覆盖和非法值。"""

from __future__ import annotations

from pathlib import Path

import pytest

from minicoder.config import Config
from minicoder.providers import get_provider


def _base_env(monkeypatch) -> None:
    monkeypatch.setenv("MINICODER_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("MINICODER_MODEL", raising=False)
    monkeypatch.delenv("MINICODER_PROFILE", raising=False)
    monkeypatch.delenv("MINICODER_INTENT_MODEL", raising=False)
    monkeypatch.delenv("MINICODER_TIMEZONE", raising=False)
    monkeypatch.delenv("MINICODER_INTENT_RULE_FAST_PATH", raising=False)
    monkeypatch.delenv("MINICODER_VEHICLE_API_URL", raising=False)
    monkeypatch.delenv("MINICODER_VEHICLE_API_TOKEN", raising=False)
    monkeypatch.delenv("MINICODER_VEHICLE_API_TIMEOUT", raising=False)
    monkeypatch.delenv("MINICODER_VEHICLE_API_ALLOW_REMOTE", raising=False)
    monkeypatch.delenv("MINICODER_VEHICLE_PERMISSION_MODE", raising=False)
    monkeypatch.delenv("MINICODER_VEHICLE_ALLOWED_IDS", raising=False)
    monkeypatch.delenv("MINICODER_KNOWLEDGE_DIR", raising=False)
    monkeypatch.delenv("MINICODER_KNOWLEDGE_INDEX", raising=False)
    monkeypatch.delenv("MINICODER_KNOWLEDGE_TOP_K", raising=False)
    monkeypatch.delenv("MINICODER_KNOWLEDGE_AUTO_REBUILD", raising=False)
    monkeypatch.delenv("MINICODER_PLANNER_MODE", raising=False)
    monkeypatch.delenv("MINICODER_PLANNER_MODEL", raising=False)


def test_permission_mode_defaults_to_ask(tmp_path: Path, monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.delenv("MINICODER_PERMISSION_MODE", raising=False)
    config = Config.load(tmp_path / "missing.env")
    assert config.permission_mode == "ask"


def test_permission_mode_from_environment(tmp_path: Path, monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("MINICODER_PERMISSION_MODE", "TRUSTED")
    config = Config.load(tmp_path / "missing.env")
    assert config.permission_mode == "allow"


def test_invalid_permission_mode_rejected(tmp_path: Path, monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("MINICODER_PERMISSION_MODE", "unrestricted")
    with pytest.raises(ValueError, match="未知权限模式"):
        Config.load(tmp_path / "missing.env")


def test_context_budget_from_environment(tmp_path: Path, monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("MINICODER_CONTEXT_WINDOW", "64000")
    monkeypatch.setenv("MINICODER_OUTPUT_RESERVE", "4096")
    config = Config.load(tmp_path / "missing.env")
    assert config.context_window == 64_000
    assert config.output_reserve == 4_096


def test_output_reserve_must_fit_context_window(tmp_path: Path, monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("MINICODER_CONTEXT_WINDOW", "1000")
    monkeypatch.setenv("MINICODER_OUTPUT_RESERVE", "1000")
    with pytest.raises(ValueError, match="必须小于"):
        Config.load(tmp_path / "missing.env")


def test_retry_config_from_environment(tmp_path: Path, monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("MINICODER_MAX_RETRIES", "0")
    monkeypatch.setenv("MINICODER_RETRY_BASE_DELAY", "0.5")
    monkeypatch.setenv("MINICODER_RETRY_MAX_DELAY", "4")
    monkeypatch.setenv("MINICODER_RETRY_MAX_WAIT", "30")
    config = Config.load(tmp_path / "missing.env")
    assert config.max_retries == 0
    assert config.retry_base_delay == 0.5
    assert config.retry_max_delay == 4
    assert config.retry_max_wait == 30
    provider = get_provider(config)
    assert provider.retry.policy.max_retries == 0
    assert provider.retry.policy.base_delay == 0.5


def test_autosave_config_from_environment(tmp_path: Path, monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("MINICODER_AUTOSAVE", "off")
    monkeypatch.setenv("MINICODER_AUTOSAVE_NAME", "daily")

    config = Config.load(tmp_path / "missing.env")

    assert config.autosave is False
    assert config.autosave_name == "daily"


def test_invalid_autosave_boolean_is_rejected(tmp_path: Path, monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("MINICODER_AUTOSAVE", "sometimes")

    with pytest.raises(ValueError, match="MINICODER_AUTOSAVE 必须"):
        Config.load(tmp_path / "missing.env")


def test_read_only_from_environment(tmp_path: Path, monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("MINICODER_READ_ONLY", "true")
    assert Config.load(tmp_path / "missing.env").read_only is True


def test_dotenv_loads_values_without_overriding_environment(tmp_path: Path, monkeypatch):
    _base_env(monkeypatch)
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        """# comment
invalid line
OPENAI_API_KEY=file-key
MINICODER_MODEL='file-model'
EMPTY_KEY=
""",
        encoding="utf-8",
    )

    config = Config.load(dotenv)

    assert config.api_key == "test-key"
    assert config.model == "file-model"


def test_anthropic_defaults(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MINICODER_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("MINICODER_MODEL", raising=False)

    config = Config.load(tmp_path / "missing.env")

    assert config.model == "claude-sonnet-4-5"
    assert config.api_key == "anthropic-key"
    assert config.base_url == "https://api.anthropic.com"


def test_invalid_numeric_values_fall_back_to_defaults(tmp_path: Path, monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("MINICODER_MAX_ROUNDS", "invalid")
    monkeypatch.setenv("MINICODER_CONTEXT_WINDOW", "-1")
    monkeypatch.setenv("MINICODER_OUTPUT_RESERVE", "invalid")
    monkeypatch.setenv("MINICODER_MAX_RETRIES", "invalid")
    monkeypatch.setenv("MINICODER_RETRY_BASE_DELAY", "invalid")
    monkeypatch.setenv("MINICODER_RETRY_MAX_DELAY", "-1")
    monkeypatch.setenv("MINICODER_RETRY_MAX_WAIT", "-1")

    config = Config.load(tmp_path / "missing.env")

    assert config.max_rounds == 50
    assert config.context_window == 128_000
    assert config.output_reserve == 16_384
    assert config.max_retries == 3
    assert config.retry_base_delay == 1.0
    assert config.retry_max_delay == 8.0
    assert config.retry_max_wait == 60.0


def test_unknown_provider_and_missing_keys_are_rejected(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MINICODER_PROVIDER", "unknown")
    with pytest.raises(ValueError, match="未知 provider"):
        Config.load(tmp_path / "missing.env")

    openai = Config("openai", "model", "", None, 1)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        openai.require_api_key()

    anthropic = Config("anthropic", "model", "", None, 1)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        anthropic.require_api_key()


def test_model_profile_provides_default_context_window(tmp_path: Path, monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("MINICODER_MODEL", "gpt-4.1")
    monkeypatch.delenv("MINICODER_CONTEXT_WINDOW", raising=False)
    monkeypatch.delenv("MINICODER_OUTPUT_RESERVE", raising=False)

    config = Config.load(tmp_path / "missing.env")

    assert config.context_window == 1_047_576
    assert config.output_reserve == 16_384


def test_automotive_profile_config(tmp_path: Path, monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("MINICODER_PROFILE", "automotive")
    monkeypatch.setenv("MINICODER_INTENT_MODEL", "small-intent-model")
    monkeypatch.setenv("MINICODER_TIMEZONE", "Asia/Shanghai")
    monkeypatch.setenv("MINICODER_INTENT_RULE_FAST_PATH", "false")

    config = Config.load(tmp_path / "missing.env")

    assert config.application_profile == "automotive"
    assert config.intent_model == "small-intent-model"
    assert config.timezone == "Asia/Shanghai"
    assert config.intent_rule_fast_path is False


def test_vehicle_api_config(tmp_path: Path, monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("MINICODER_VEHICLE_API_URL", "https://vehicle.example/v1")
    monkeypatch.setenv("MINICODER_VEHICLE_API_TOKEN", "vehicle-token")
    monkeypatch.setenv("MINICODER_VEHICLE_API_TIMEOUT", "2.5")
    monkeypatch.setenv("MINICODER_VEHICLE_API_ALLOW_REMOTE", "true")
    monkeypatch.setenv("MINICODER_VEHICLE_PERMISSION_MODE", "deny")
    monkeypatch.setenv("MINICODER_VEHICLE_ALLOWED_IDS", "a102, A103")

    config = Config.load(tmp_path / "missing.env")

    assert config.vehicle_api_url == "https://vehicle.example/v1"
    assert config.vehicle_api_token == "vehicle-token"
    assert config.vehicle_api_timeout == 2.5
    assert config.vehicle_api_allow_remote is True
    assert config.vehicle_permission_mode == "deny"
    assert config.vehicle_allowed_ids == ("A102", "A103")


def test_knowledge_config(tmp_path: Path, monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("MINICODER_KNOWLEDGE_DIR", "knowledge-custom")
    monkeypatch.setenv("MINICODER_KNOWLEDGE_INDEX", "cache/knowledge.db")
    monkeypatch.setenv("MINICODER_KNOWLEDGE_TOP_K", "7")
    monkeypatch.setenv("MINICODER_KNOWLEDGE_AUTO_REBUILD", "false")
    monkeypatch.setenv("MINICODER_PLANNER_MODE", "hybrid")
    monkeypatch.setenv("MINICODER_PLANNER_MODEL", "planner-model")

    config = Config.load(tmp_path / "missing.env")

    assert config.knowledge_dir == "knowledge-custom"
    assert config.knowledge_index == "cache/knowledge.db"
    assert config.knowledge_top_k == 7
    assert config.knowledge_auto_rebuild is False
    assert config.planner_mode == "hybrid"
    assert config.planner_model == "planner-model"


def test_knowledge_top_k_is_bounded(tmp_path: Path, monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("MINICODER_KNOWLEDGE_TOP_K", "21")
    with pytest.raises(ValueError, match="1～20"):
        Config.load(tmp_path / "missing.env")


def test_invalid_planner_mode_is_rejected(tmp_path: Path, monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("MINICODER_PLANNER_MODE", "autonomous")
    with pytest.raises(ValueError, match="rules 或 hybrid"):
        Config.load(tmp_path / "missing.env")


def test_invalid_vehicle_api_config_is_rejected(tmp_path: Path, monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("MINICODER_VEHICLE_API_ALLOW_REMOTE", "sometimes")
    with pytest.raises(ValueError, match="MINICODER_VEHICLE_API_ALLOW_REMOTE 必须"):
        Config.load(tmp_path / "missing.env")


def test_invalid_automotive_config_is_rejected(tmp_path: Path, monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("MINICODER_PROFILE", "vehicle")
    with pytest.raises(ValueError, match="MINICODER_PROFILE"):
        Config.load(tmp_path / "missing.env")

    monkeypatch.setenv("MINICODER_PROFILE", "automotive")
    monkeypatch.setenv("MINICODER_TIMEZONE", "Mars/Olympus")
    with pytest.raises(ValueError, match="未知时区"):
        Config.load(tmp_path / "missing.env")
