from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.agents.llm_participant_agent import ParticipantPool
from src.core.benchmark import HumanStudyBench
from src.core.study import Study
from src.core.study_config import get_study_config


@dataclass
class Stage5Options:
    n_participants: Optional[int] = None
    n_agents: Optional[int] = None
    max_agents: Optional[int] = None
    repeats: int = 1
    seed: int = 42
    temperature: float = 1.0
    num_workers: Optional[int] = None
    mock: bool = False
    dry_run: bool = False
    use_cache: bool = False
    cache_dir: Optional[Path] = None
    profiles_json: Optional[Path] = None
    system_prompt_file: Optional[Path] = None
    system_prompt_preset: str = "v3_human_plus_demo"
    reasoning: str = "default"
    enable_reasoning: bool = False
    api_key: Optional[str] = None
    api_base: Optional[str] = None

    def requested_participants(self) -> Optional[int]:
        value = self.n_participants if self.n_participants is not None else self.n_agents
        if value is None:
            return None
        value = int(value)
        if self.max_agents is not None:
            value = min(value, int(self.max_agents))
        return max(value, 1)

    @property
    def use_real_llm(self) -> bool:
        return not (self.mock or self.dry_run)


def _safe_model_slug(model: str) -> str:
    return str(model).replace("/", "_").replace("-", "_").replace(":", "_")


def _load_profiles(path: Optional[Path]) -> Optional[List[Dict[str, Any]]]:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("profiles")
    if not isinstance(data, list):
        raise ValueError(f"profiles_json must contain a list or {{'profiles': list}}: {path}")
    return data


def _read_optional_text(path: Optional[Path]) -> Optional[str]:
    if not path:
        return None
    return Path(path).read_text(encoding="utf-8")


def _is_study_package_dir(path: Path) -> bool:
    flat = (
        path.is_dir()
        and (path / "metadata.json").exists()
        and (path / "specification.json").exists()
        and (path / "ground_truth.json").exists()
        and (path / "materials").is_dir()
    )
    source = path / "source"
    hub = (
        path.is_dir()
        and source.is_dir()
        and (source / "metadata.json").exists()
        and (source / "specification.json").exists()
        and (source / "ground_truth.json").exists()
        and (source / "materials").is_dir()
    )
    return flat or hub


def _resolve_study(study: str | Path, data_dir: Path) -> Tuple[str, Path, Optional[Path]]:
    candidate = Path(study)
    if candidate.exists() and candidate.is_dir():
        if _is_study_package_dir(candidate):
            study_id = candidate.name
            if candidate.parent.name == "studies":
                return study_id, candidate.parent.parent, None
            return study_id, data_dir, candidate
        if (candidate / "registry.json").exists():
            raise ValueError("When passing a data directory to Stage 5, also pass --study-id.")
    return str(study), data_dir, None


def _choose_trial_mode(
    specification: Dict[str, Any],
    requested_n: Optional[int],
    trials: List[Dict[str, Any]],
) -> Tuple[int, bool]:
    by_sub_study = specification.get("participants", {}).get("by_sub_study")
    if requested_n is not None or by_sub_study:
        return len(trials), True
    n_default = specification.get("participants", {}).get("n") or 30
    return int(n_default), False


def _cache_path(options: Stage5Options, study_id: str, model: str, repeat_idx: int) -> Optional[Path]:
    if not options.use_cache:
        return None
    cache_dir = Path(options.cache_dir or ".cache/stage5")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{study_id}_{_safe_model_slug(model)}_repeat_{repeat_idx}.json"


def _run_single_repeat(
    *,
    study_id: str,
    study: Any,
    study_config: Any,
    model: str,
    repeat_idx: int,
    options: Stage5Options,
    profiles: Optional[List[Dict[str, Any]]],
    system_prompt_override: Optional[str],
) -> Dict[str, Any]:
    cache_path = _cache_path(options, study_id, model, repeat_idx)
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and cached.get("raw_results"):
            return cached["raw_results"]

    requested_n = options.requested_participants()
    trials = study_config.create_trials(n_trials=requested_n)
    instructions = study_config.get_instructions()
    builder = study_config.get_prompt_builder()

    participant_pool_kwargs = {
        "study_specification": study.specification,
        "use_real_llm": options.use_real_llm,
        "model": model,
        "api_key": options.api_key,
        "api_base": options.api_base,
        "random_seed": options.seed + repeat_idx,
        "num_workers": int(options.num_workers) if options.num_workers is not None else None,
        "profiles": profiles,
        "prompt_builder": builder,
        "system_prompt_override": system_prompt_override,
        "system_prompt_preset": options.system_prompt_preset,
        "reasoning": options.reasoning,
        "enable_reasoning": options.enable_reasoning,
        "study_id": study_id,
        "temperature": options.temperature,
    }

    if getattr(study_config, "REQUIRES_GROUP_TRIALS", False) and hasattr(study_config, "run_group_experiment"):
        raw_results = study_config.run_group_experiment(
            trials,
            instructions,
            participant_pool_kwargs,
            prompt_builder=builder,
        )
    else:
        n_participants, one_to_one = _choose_trial_mode(study.specification, requested_n, trials)
        participant_pool_kwargs["n_participants"] = n_participants
        pool = ParticipantPool(**participant_pool_kwargs)
        raw_results = pool.run_experiment(
            trials,
            instructions,
            prompt_builder=builder,
            one_to_one=one_to_one,
        )

    if cache_path:
        cache_path.write_text(
            json.dumps(
                {"version": 1, "study_id": study_id, "model": model, "repeat_idx": repeat_idx, "raw_results": raw_results},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return raw_results


def run_stage5(
    study: str | Path,
    *,
    runs_dir: Path = Path("results/benchmark"),
    models: Optional[List[str]] = None,
    client_factory: Any = None,
    options: Optional[Stage5Options] = None,
    data_dir: Path = Path("data"),
) -> Dict[str, Any]:
    del client_factory
    options = options or Stage5Options()
    model_names = models or ["mistralai/mistral-nemo"]
    study_id, resolved_data_dir, direct_study_path = _resolve_study(study, Path(data_dir))

    if direct_study_path is not None:
        loaded_study = Study.load(direct_study_path)
        study_path = direct_study_path
    else:
        benchmark = HumanStudyBench(resolved_data_dir)
        loaded_study = benchmark.load_study(study_id)
        study_path = resolved_data_dir / "studies" / study_id
    study_config = get_study_config(study_id, study_path, loaded_study.specification)

    profiles = _load_profiles(options.profiles_json)
    system_prompt_override = _read_optional_text(options.system_prompt_file)
    start_time = time.time()
    all_model_runs: List[Dict[str, Any]] = []

    for model in model_names:
        raw_runs: List[Dict[str, Any]] = []
        for repeat_idx in range(max(int(options.repeats), 1)):
            raw_runs.append(
                _run_single_repeat(
                    study_id=study_id,
                    study=loaded_study,
                    study_config=study_config,
                    model=model,
                    repeat_idx=repeat_idx,
                    options=options,
                    profiles=profiles,
                    system_prompt_override=system_prompt_override,
                )
            )

        try:
            aggregate = study_config.aggregate_results(raw_runs[0] if raw_runs else {"individual_data": []})
        except Exception as exc:
            aggregate = {"descriptive_statistics": {}, "inferential_statistics": {}, "error": str(exc)}

        model_dir = Path(runs_dir) / study_id / _safe_model_slug(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        output_path = model_dir / "full_benchmark.json"
        save_data = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "stage5_version": "human-study-bench",
            "study_id": study_id,
            "title": loaded_study.metadata.get("title", ""),
            "model": model,
            "use_real_llm": options.use_real_llm,
            "system_prompt_preset": options.system_prompt_preset,
            "reasoning": options.reasoning,
            "random_seed": options.seed,
            "repeats": len(raw_runs),
            "elapsed_time": time.time() - start_time,
            "descriptive_statistics": aggregate.get("descriptive_statistics", {}),
            "inferential_statistics": aggregate.get("inferential_statistics", {}),
            "individual_data": raw_runs[0].get("individual_data", []) if raw_runs else [],
            "all_runs_raw_results": [
                {"individual_data": run.get("individual_data", [])}
                for run in raw_runs
            ],
        }
        output_path.write_text(json.dumps(save_data, indent=2, ensure_ascii=False), encoding="utf-8")
        all_model_runs.append(
            {
                "model": model,
                "output_path": str(output_path),
                "repeats": len(raw_runs),
                "responses": sum(len(run.get("individual_data", [])) for run in raw_runs),
                "aggregate_error": aggregate.get("error"),
            }
        )

    return {
        "stage5_version": "human-study-bench",
        "study_id": study_id,
        "data_dir": str(resolved_data_dir),
        "study_path": str(study_path),
        "runs_dir": str(runs_dir),
        "models": model_names,
        "run_count": sum(item["repeats"] for item in all_model_runs),
        "completed": len(all_model_runs),
        "use_real_llm": options.use_real_llm,
        "runs": all_model_runs,
    }
