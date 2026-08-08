"""Load a study's own trial builder and evaluator from `studies/<id>/scripts/`.

Every study ships a self-contained `scripts/config.py` (a `BaseStudyConfig`
subclass that builds trials and knows the study's prompts) and a
`scripts/evaluator.py` exposing `evaluate_study(results)`. The playground drives
those files rather than reimplementing any study logic.
"""

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Tuple


class StudyNotRunnable(Exception):
    """The study exists but does not expose the files the playground needs."""


def _import_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise StudyNotRunnable(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    # A study's scripts import their sibling modules by bare name, so the study's
    # own scripts directory has to be importable while the module executes.
    sys.modules[name] = module
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec.loader.exec_module(module)
    return module


def study_dir(repo_root: Path, study_id: str) -> Path:
    safe = "".join(character for character in study_id if character.isalnum() or character in "_-")
    if not safe or safe != study_id:
        raise StudyNotRunnable(f"Invalid study id: {study_id}")
    path = repo_root / "studies" / safe
    if not path.is_dir():
        raise StudyNotRunnable(f"Study {safe} is not in this benchmark.")
    return path


def source_dir(path: Path) -> Path:
    source = path / "source"
    return source if (source / "specification.json").exists() else path


def load_specification(path: Path) -> Dict[str, Any]:
    spec_path = source_dir(path) / "specification.json"
    if not spec_path.exists():
        raise StudyNotRunnable(f"{path.name} has no specification.json, so it cannot be replayed.")
    with open(spec_path, "r", encoding="utf-8", errors="replace") as handle:
        return json.load(handle)


def load_metadata(path: Path) -> Dict[str, Any]:
    for candidate in (source_dir(path) / "metadata.json", path / "index.json"):
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8", errors="replace") as handle:
                return json.load(handle)
    return {}


def load_study_config(path: Path, specification: Dict[str, Any]) -> Any:
    """Instantiate the study's `BaseStudyConfig` subclass."""
    config_path = path / "scripts" / "config.py"
    if not config_path.exists():
        raise StudyNotRunnable(f"{path.name} has no scripts/config.py, so its trials cannot be built.")
    module = _import_module(config_path, f"playground_{path.name}_config")
    candidates = [
        value for _, value in inspect.getmembers(module, inspect.isclass)
        if value.__module__ == module.__name__
        and hasattr(value, "create_trials")
        and not inspect.isabstract(value)
        and any(base.__name__ == "BaseStudyConfig" for base in value.__mro__[1:])
    ]
    if not candidates:
        raise StudyNotRunnable(f"{path.name} has no study config class in scripts/config.py.")
    return candidates[0](path, specification)


def load_evaluator(path: Path) -> ModuleType:
    evaluator_path = path / "scripts" / "evaluator.py"
    if not evaluator_path.exists():
        raise StudyNotRunnable(f"{path.name} has no scripts/evaluator.py, so agent runs cannot be scored.")
    module = _import_module(evaluator_path, f"playground_{path.name}_evaluator")
    if not hasattr(module, "evaluate_study"):
        raise StudyNotRunnable(f"{path.name}'s evaluator does not expose evaluate_study().")
    return module


def load_study(repo_root: Path, study_id: str) -> Tuple[Path, Dict[str, Any], Any, ModuleType]:
    path = study_dir(repo_root, study_id)
    specification = load_specification(path)
    return path, specification, load_study_config(path, specification), load_evaluator(path)
