from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SETTINGS_PATHS = (
    Path("config/settings.yaml"),
    Path("config/settings.yml"),
    Path("config/settings.json"),
)
SUPPORTED_PROVIDERS = ("openai", "google", "anthropic", "openrouter", "vllm")
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL_BY_PROVIDER = {
    "openai": "gpt-5-mini",
    "google": "gemini-3-flash",
    "anthropic": "claude-sonnet-4-6",
    "openrouter": "anthropic/claude-sonnet-4-6",
    "vllm": "local-model",
}
PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "vllm": "OPENAI_API_KEY",
}
PROVIDER_DEFAULT_API_BASE = {
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "anthropic": "https://api.anthropic.com",
    "openrouter": "https://openrouter.ai/api/v1",
    "vllm": "http://localhost:8000/v1",
}
DEFAULT_OUTPUT_DIR = Path("generation_pipeline/outputs")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_EXPERIMENTS_DIR = Path("experiments")
DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_VERIFICATION_THRESHOLD = 90.0
AGENT_LLM_SECTION_NAMES = ("agent_llm", "simulation_llm")
STAGE_LLM_SECTION_NAMES = {
    1: ("stage1_llm", "filter_llm"),
    2: ("stage2_llm", "extraction_llm"),
    3: ("stage3_llm", "patch_llm"),
    4: ("stage4_llm", "config_llm", "gym_llm"),
    5: ("stage5_llm", "agent_llm", "simulation_llm"),
}
STAGE_ENV_PREFIXES = {
    1: ("STAGE1", "FILTER"),
    2: ("STAGE2", "EXTRACTION"),
    3: ("STAGE3", "PATCH"),
    4: ("STAGE4", "CONFIG", "GYM"),
    5: ("STAGE5", "AGENT", "SIM"),
}


@dataclass(frozen=True)
class AppSettings:
    data: dict[str, Any]
    path: Path | None = None

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name, {})
        return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class ResolvedLLMConfig:
    provider: str
    model: str
    api_key: str | None
    api_base: str | None


def load_settings(path: str | Path | None = None) -> AppSettings:
    """Load optional settings file and `.env` variables."""
    _load_dotenv()
    settings_path = _resolve_settings_path(path)
    if settings_path is None:
        return AppSettings(data={}, path=None)
    return AppSettings(data=_read_settings_file(settings_path), path=settings_path)


def resolve_llm_config(
    settings: AppSettings | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> ResolvedLLMConfig:
    """Resolve LLM provider/model/API settings from CLI/env/config/defaults."""
    settings = settings or load_settings()
    llm = settings.section("llm")

    resolved_provider = str(
        _first_non_empty(
            provider,
            os.getenv("PROVIDER"),
            llm.get("provider"),
            settings.data.get("provider"),
            DEFAULT_PROVIDER,
        )
    ).lower()
    if resolved_provider not in PROVIDER_API_KEY_ENV:
        raise ValueError(
            f"Unknown provider: {resolved_provider}. "
            f"Use one of: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    resolved_model = str(
        _first_non_empty(
            model,
            os.getenv("MODEL"),
            llm.get("model"),
            settings.data.get("model"),
            DEFAULT_MODEL_BY_PROVIDER.get(resolved_provider),
        )
    )

    resolved_api_key = _first_non_empty(
        api_key,
        os.getenv(PROVIDER_API_KEY_ENV[resolved_provider]),
        llm.get("api_key"),
        settings.data.get("api_key"),
        "EMPTY" if resolved_provider == "vllm" else None,
    )

    resolved_api_base = _first_non_empty(
        api_base,
        os.getenv("BASE_URL"),
        llm.get("base_url"),
        llm.get("api_base"),
        settings.data.get("base_url"),
        settings.data.get("api_base"),
        PROVIDER_DEFAULT_API_BASE.get(resolved_provider),
    )

    return ResolvedLLMConfig(
        provider=resolved_provider,
        model=resolved_model,
        api_key=str(resolved_api_key) if resolved_api_key is not None else None,
        api_base=str(resolved_api_base) if resolved_api_base is not None else None,
    )


def resolve_stage_llm_config(
    settings: AppSettings | None = None,
    *,
    stage: int,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> ResolvedLLMConfig:
    """Resolve the LLM config for one pipeline stage.

    Resolution order:
    CLI override → stage-specific env → stage-specific settings section →
    provider env / global llm fallback → provider defaults.
    """
    settings = settings or load_settings()
    base = resolve_llm_config(settings)
    sections = [settings.section(name) for name in STAGE_LLM_SECTION_NAMES.get(stage, ())]
    if stage == 5:
        sections = _stage5_active_sections(sections, base)
    prefixes = STAGE_ENV_PREFIXES.get(stage, (f"STAGE{stage}",))

    def from_sections(key: str) -> Any:
        return _first_non_empty(*(section.get(key) for section in sections))

    provider_override = _first_non_empty(
        provider,
        *(_env(prefix, "PROVIDER") for prefix in prefixes),
        from_sections("provider"),
    )
    provider_was_overridden = provider_override is not None
    resolved_provider = str(provider_override or base.provider).lower()
    if resolved_provider not in PROVIDER_API_KEY_ENV:
        raise ValueError(
            f"Unknown provider for stage {stage}: {resolved_provider}. "
            f"Use one of: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    same_provider_as_base = resolved_provider == base.provider

    resolved_model = _first_non_empty(
        model,
        *(_env(prefix, "MODEL") for prefix in prefixes),
        from_sections("model"),
        DEFAULT_MODEL_BY_PROVIDER.get(resolved_provider) if provider_was_overridden else base.model,
        DEFAULT_MODEL_BY_PROVIDER.get(resolved_provider),
    )

    resolved_api_key = _first_non_empty(
        api_key,
        *(_env(prefix, "API_KEY") for prefix in prefixes),
        from_sections("api_key"),
        os.getenv(PROVIDER_API_KEY_ENV[resolved_provider]),
        base.api_key if same_provider_as_base else None,
        "EMPTY" if resolved_provider == "vllm" else None,
    )

    resolved_api_base = _first_non_empty(
        api_base,
        *(_env(prefix, "BASE_URL") for prefix in prefixes),
        *(_env(prefix, "API_BASE") for prefix in prefixes),
        from_sections("base_url"),
        from_sections("api_base"),
        base.api_base if same_provider_as_base else None,
        PROVIDER_DEFAULT_API_BASE.get(resolved_provider),
    )

    return ResolvedLLMConfig(
        provider=resolved_provider,
        model=str(resolved_model),
        api_key=str(resolved_api_key) if resolved_api_key is not None else None,
        api_base=str(resolved_api_base) if resolved_api_base is not None else None,
    )


def _stage5_active_sections(sections: list[dict[str, Any]], base: ResolvedLLMConfig) -> list[dict[str, Any]]:
    """Ignore copied Stage-5 cross-provider defaults that have no usable key.

    Stage 5 should inherit the global `llm` config unless the user explicitly
    makes a stage-specific provider runnable by setting a matching key in the
    section or environment. Same-provider model overrides are still active and
    inherit the global key/base URL.
    """
    active: list[dict[str, Any]] = []
    for section in sections:
        provider = _first_non_empty(section.get("provider"))
        if provider is None:
            active.append(section)
            continue
        provider_name = str(provider).lower()
        if provider_name not in PROVIDER_API_KEY_ENV:
            active.append(section)
            continue
        if provider_name == base.provider:
            active.append(section)
            continue
        if provider_name == "vllm":
            active.append(section)
            continue
        if _first_non_empty(section.get("api_key"), os.getenv(PROVIDER_API_KEY_ENV[provider_name])):
            active.append(section)
    return active


def resolve_agent_llm_config(
    settings: AppSettings | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> ResolvedLLMConfig:
    """ Resolve the LLM config reserved for Stage 5 agent simulation."""
    settings = settings or load_settings()
    extraction_config = resolve_llm_config(settings)
    agent_llm = settings.section(AGENT_LLM_SECTION_NAMES[0])
    simulation_llm = settings.section(AGENT_LLM_SECTION_NAMES[1])

    provider_override = _first_non_empty(
        provider,
        os.getenv("AGENT_PROVIDER"),
        os.getenv("SIM_PROVIDER"),
        agent_llm.get("provider"),
        simulation_llm.get("provider"),
    )
    provider_was_overridden = provider_override is not None
    resolved_provider = str(provider_override or extraction_config.provider).lower()
    if resolved_provider not in PROVIDER_API_KEY_ENV:
        raise ValueError(
            f"Unknown provider: {resolved_provider}. "
            f"Use one of: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    same_provider_as_extraction = resolved_provider == extraction_config.provider

    resolved_model = _first_non_empty(
        model,
        os.getenv("AGENT_MODEL"),
        os.getenv("SIM_MODEL"),
        agent_llm.get("model"),
        simulation_llm.get("model"),
        DEFAULT_MODEL_BY_PROVIDER.get(resolved_provider) if provider_was_overridden else extraction_config.model,
        DEFAULT_MODEL_BY_PROVIDER.get(resolved_provider),
    )

    resolved_api_key = _first_non_empty(
        api_key,
        os.getenv("AGENT_API_KEY"),
        os.getenv("SIM_API_KEY"),
        os.getenv(PROVIDER_API_KEY_ENV[resolved_provider]),
        agent_llm.get("api_key"),
        simulation_llm.get("api_key"),
        "EMPTY" if resolved_provider == "vllm" and provider_was_overridden else None,
        extraction_config.api_key if same_provider_as_extraction else None,
        "EMPTY" if resolved_provider == "vllm" else None,
    )

    resolved_api_base = _first_non_empty(
        api_base,
        os.getenv("AGENT_BASE_URL"),
        os.getenv("AGENT_API_BASE"),
        os.getenv("SIM_BASE_URL"),
        os.getenv("SIM_API_BASE"),
        agent_llm.get("base_url"),
        agent_llm.get("api_base"),
        simulation_llm.get("base_url"),
        simulation_llm.get("api_base"),
        extraction_config.api_base if same_provider_as_extraction else None,
        PROVIDER_DEFAULT_API_BASE.get(resolved_provider),
    )

    return ResolvedLLMConfig(
        provider=resolved_provider,
        model=str(resolved_model),
        api_key=str(resolved_api_key) if resolved_api_key is not None else None,
        api_base=str(resolved_api_base) if resolved_api_base is not None else None,
    )


def resolve_output_dir(settings: AppSettings | None = None, output_dir: str | Path | None = None) -> Path:
    settings = settings or load_settings()
    paths = settings.section("paths")
    value = _first_non_empty(output_dir, os.getenv("OUTPUT_DIR"), settings.data.get("output_dir"), paths.get("output_dir"), DEFAULT_OUTPUT_DIR)
    return Path(value)


def resolve_experiments_dir(settings: AppSettings | None = None, experiments_dir: str | Path | None = None) -> Path:
    settings = settings or load_settings()
    paths = settings.section("paths")
    value = _first_non_empty(
        experiments_dir,
        os.getenv("EXPERIMENTS_DIR"),
        settings.data.get("experiments_dir"),
        paths.get("experiments_dir"),
        DEFAULT_EXPERIMENTS_DIR,
    )
    return Path(value)


def resolve_data_dir(settings: AppSettings | None = None, data_dir: str | Path | None = None) -> Path:
    settings = settings or load_settings()
    paths = settings.section("paths")
    value = _first_non_empty(
        data_dir,
        os.getenv("DATA_DIR"),
        settings.data.get("data_dir"),
        paths.get("data_dir"),
        DEFAULT_DATA_DIR,
    )
    return Path(value)


def resolve_runs_dir(settings: AppSettings | None = None, runs_dir: str | Path | None = None) -> Path:
    settings = settings or load_settings()
    paths = settings.section("paths")
    value = _first_non_empty(
        runs_dir,
        os.getenv("RUNS_DIR"),
        settings.data.get("runs_dir"),
        paths.get("runs_dir"),
        DEFAULT_RUNS_DIR,
    )
    return Path(value)


def resolve_verification_threshold(
    settings: AppSettings | None = None,
    threshold: float | None = None,
) -> float:
    settings = settings or load_settings()
    verification = settings.section("verification")
    value = _first_non_empty(
        threshold,
        os.getenv("VERIFICATION_THRESHOLD"),
        settings.data.get("verification_threshold"),
        verification.get("threshold"),
        DEFAULT_VERIFICATION_THRESHOLD,
    )
    return float(value)


def _resolve_settings_path(path: str | Path | None) -> Path | None:
    explicit = path or os.getenv("SETTINGS_PATH")
    if explicit:
        settings_path = Path(explicit)
        if not settings_path.exists():
            raise FileNotFoundError(f"Settings file not found: {settings_path}")
        return settings_path

    for candidate in DEFAULT_SETTINGS_PATHS:
        if candidate.exists():
            return candidate
    return None


def _read_settings_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError:
            data = _read_simple_yaml(text)
        else:
            data = yaml.safe_load(text) or {}
    else:
        raise ValueError(f"Unsupported settings file type: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"Settings file must contain an object at top level: {path}")
    return data


def _read_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small settings YAML subset used by config/settings.yaml."""
    data: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    current_indent = 0

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = _strip_inline_comment(raw_line).strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported settings YAML line: {raw_line}")

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Unsupported settings YAML line: {raw_line}")

        if indent == 0:
            if value == "":
                current_section = {}
                current_indent = indent
                data[key] = current_section
            else:
                data[key] = _parse_scalar(value)
                current_section = None
        elif current_section is not None and indent > current_indent:
            current_section[key] = _parse_scalar(value)
        else:
            raise ValueError(f"Unsupported nested settings YAML line: {raw_line}")

    return data


def _strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "none", "~", ""}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def _env(prefix: str, suffix: str) -> str | None:
    return os.getenv(f"{prefix}_{suffix}")


def settings_to_public_dict(
    settings: AppSettings,
    llm_config: ResolvedLLMConfig,
    agent_llm_config: ResolvedLLMConfig | None = None,
) -> dict[str, Any]:
    """Return a secret-safe view useful for logging/debugging."""
    public = {
        "settings_path": str(settings.path) if settings.path else None,
        "llm": {
            "provider": llm_config.provider,
            "model": llm_config.model,
            "api_base": llm_config.api_base,
            "api_key_set": bool(llm_config.api_key),
        },
    }
    if agent_llm_config is not None:
        public["agent_llm"] = {
            "provider": agent_llm_config.provider,
            "model": agent_llm_config.model,
            "api_base": agent_llm_config.api_base,
            "api_key_set": bool(agent_llm_config.api_key),
        }
    return public
