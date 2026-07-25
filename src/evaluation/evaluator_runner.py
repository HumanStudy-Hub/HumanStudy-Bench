"""Locate and execute study-specific evaluator modules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional


_evaluator_cache: Dict[str, Optional[Any]] = {}


def find_evaluator_path(
    study_id: str,
    study_path: Optional[str | Path] = None,
) -> Optional[Path]:
    """Resolve a Hub package evaluator, with legacy-path fallback."""

    candidates = []
    if study_path is not None:
        package_path = Path(study_path)
        if package_path.name == "source":
            package_path = package_path.parent
        candidates.append(package_path / "scripts" / "evaluator.py")
    candidates.extend(
        [
            Path("studies") / study_id / "scripts" / "evaluator.py",
            Path("src") / "studies" / f"{study_id}_evaluator.py",
        ]
    )

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def load_evaluator(
    study_id: str,
    study_path: Optional[str | Path] = None,
) -> Optional[Any]:
    """Load and cache the evaluator associated with a study package."""

    evaluator_path = find_evaluator_path(study_id, study_path)
    if evaluator_path is None:
        return None

    cache_key = str(evaluator_path)
    if cache_key in _evaluator_cache:
        return _evaluator_cache[cache_key]

    try:
        module_name = f"{study_id}_evaluator_{abs(hash(cache_key))}"
        spec = importlib.util.spec_from_file_location(module_name, evaluator_path)
        if spec is None or spec.loader is None:
            _evaluator_cache[cache_key] = None
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        if not hasattr(module, "evaluate_study"):
            _evaluator_cache[cache_key] = None
            return None
        _evaluator_cache[cache_key] = module
        return module
    except Exception:
        _evaluator_cache[cache_key] = None
        return None


def run_evaluator(
    study_id: str,
    results: Dict[str, Any],
    study_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run a package evaluator and return a structured failure if unavailable."""

    evaluator = load_evaluator(study_id, study_path)
    if evaluator is None:
        return {
            "passed": False,
            "total_score": 0.0,
            "error": "Study evaluator not found or failed to load",
        }
    try:
        evaluation = evaluator.evaluate_study(results)
        if not isinstance(evaluation, dict):
            raise TypeError("evaluate_study must return a dictionary")
        return evaluation
    except Exception as exc:
        return {
            "passed": False,
            "total_score": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }
