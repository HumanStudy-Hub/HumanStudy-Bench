import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Cache: key (study_id, resolved study_path) -> module
_evaluator_cache: Dict[tuple, Any] = {}


def _evaluator_path(study_id: str, study_path: Optional[Path] = None) -> Path:
    """Resolve path to evaluator.py for a study (default: data/studies/<study_id>; code in scripts/)."""
    if study_path is None:
        study_path = Path("data/studies") / study_id
    return Path(study_path) / "scripts" / "evaluator.py"


def load_evaluator(study_id: str, study_path: Optional[Path] = None) -> Optional[Any]:
    """
    Load and cache the evaluator module for a study.

    Loads from study_path/evaluator.py (default study_path = data/studies/<study_id>).

    Args:
        study_id: Study ID (e.g. 'study_001')
        study_path: Study directory containing evaluator.py (default: data/studies/<study_id>)

    Returns:
        The evaluator module object, or None if loading failed
    """
    path = _evaluator_path(study_id, study_path)
    path_resolved = path.resolve()
    cache_key = (study_id, str(path_resolved))

    if cache_key in _evaluator_cache:
        return _evaluator_cache[cache_key]

    if not path.exists():
        print(f"Evaluator not found at {path}")
        _evaluator_cache[cache_key] = None
        return None

    try:
        spec = importlib.util.spec_from_file_location(f"{study_id}_evaluator", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{study_id}_evaluator"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "evaluate_study"):
            print(f"Evaluator module missing evaluate_study function")
            _evaluator_cache[cache_key] = None
            return None

        _evaluator_cache[cache_key] = module
        return module
    except Exception as e:
        print(f"Error loading evaluator: {e}")
        import traceback
        traceback.print_exc()
        _evaluator_cache[cache_key] = None
        return None


def run_evaluator(
    study_id: str,
    results: Dict[str, Any],
    study_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load and run the evaluator for a study.

    Args:
        study_id: Study ID (e.g. 'study_001')
        results: The agent results dictionary
        study_path: Study directory containing evaluator.py (default: data/studies/<study_id>)

    Returns:
        Dict containing score, pi_human, pi_agent, details
    """
    path = _evaluator_path(study_id, study_path)
    if not path.exists():
        print(f"Evaluator not found at {path}")
        return {"score": 0.0, "error": "Evaluator not found"}

    try:
        module = load_evaluator(study_id, study_path)
        if module is None:
            return {"score": 0.0, "error": "Failed to load evaluator"}
        return module.evaluate_study(results)
    except Exception as e:
        print(f"Error running evaluator: {e}")
        import traceback
        traceback.print_exc()
        return {"score": 0.0, "error": str(e)}
