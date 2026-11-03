"""Load and run a benchmark study's own evaluator.

Each study ships `studies/<study_id>/scripts/evaluator.py` exposing
`evaluate_study(results)`. Loading goes through `playground.study_loader`, so a
benchmark evaluation and a playground run import the evaluator the same way and
score identically.
"""

import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

# Cache for loaded evaluator modules, keyed by (repo_root, study_id)
_evaluator_cache: Dict[Any, Optional[ModuleType]] = {}


@contextmanager
def _isolated_siblings(scripts_dir: Path) -> Iterator[None]:
    """Keep one study's vendored sibling modules from leaking into another's.

    Every study vendors its own `stats_lib.py` / `study_utils.py` and imports them
    by bare name, and the copies differ between studies. Without this, the first
    study loaded in a process would supply those modules to every later one.
    Siblings are imported at module top level, so the loaded evaluator keeps its
    own references after they are removed from `sys.modules`.
    """
    names = {path.stem for path in scripts_dir.glob("*.py")}
    saved_path = list(sys.path)
    for name in names:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.path[:] = saved_path


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
    # Requires a source checkout: `playground/` sits beside `src/` at the repo root.
    from playground.study_loader import StudyNotRunnable, study_dir
    from playground.study_loader import load_evaluator as load_study_evaluator

    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    key = (str(root), study_id)
    if key in _evaluator_cache:
        return _evaluator_cache[key]

    module: Optional[ModuleType] = None
    try:
        path = study_dir(root, study_id)
        with _isolated_siblings(path / "scripts"):
            module = load_study_evaluator(path)
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
