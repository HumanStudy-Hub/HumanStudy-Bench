"""Load and run a benchmark study's own evaluator.

Each study ships `studies/<study_id>/scripts/evaluator.py` exposing
`evaluate_study(results)`. Loading goes through `playground.study_loader`, so a
benchmark evaluation and a playground run import the evaluator the same way and
score identically.
"""

import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

# Cache for loaded evaluator modules, keyed by (repo_root, study_id)
_evaluator_cache: Dict[Any, Optional[ModuleType]] = {}


def load_evaluator(study_id: str, repo_root: Optional[Path] = None) -> Optional[ModuleType]:
    """
    Load and cache the evaluator module for a study.

    The module is loaded once and reused, avoiding redundant imports during
    bootstrap iterations.

    Args:
        study_id: Study ID (e.g. 'study_001')
        repo_root: Benchmark root containing `studies/`; defaults to this repository

    Returns:
        The evaluator module object, or None if loading failed
    """
    from playground.study_loader import StudyNotRunnable, study_dir
    from playground.study_loader import load_evaluator as load_study_evaluator

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    key = (str(root), study_id)
    if key in _evaluator_cache:
        return _evaluator_cache[key]

    module: Optional[ModuleType] = None
    try:
        module = load_study_evaluator(study_dir(root, study_id))
    except StudyNotRunnable as e:
        print(f"Evaluator not available for {study_id}: {e}")
    except Exception as e:
        print(f"Error loading evaluator for {study_id}: {e}")
        traceback.print_exc()

    _evaluator_cache[key] = module
    return module


def run_evaluator(
    study_id: str, results: Dict[str, Any], repo_root: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Load and run the evaluator for a study.

    Args:
        study_id: Study ID (e.g. 'study_001')
        results: The agent results dictionary
        repo_root: Benchmark root containing `studies/`; defaults to this repository

    Returns:
        The evaluator's result dict, or {"score": 0.0, "error": ...} on failure
    """
    module = load_evaluator(study_id, repo_root)
    if module is None:
        return {"score": 0.0, "error": "Evaluator not found or failed to load"}

    try:
        result: Dict[str, Any] = module.evaluate_study(results)
        return result
    except Exception as e:
        print(f"Error running evaluator for {study_id}: {e}")
        traceback.print_exc()
        return {"score": 0.0, "error": str(e)}
