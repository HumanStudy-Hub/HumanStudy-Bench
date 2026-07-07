import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json
import time

# Cache for loaded evaluator modules
_evaluator_cache = {}

# #region agent log
def _debug_log(location, message, data, hypothesis_id=None):
    try:
        log_path = Path("/Users/assassin808/Desktop/xuan-hs/HS_bench/.cursor/debug.log")
        with open(log_path, "a", encoding="utf-8") as f:
            log_entry = {
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000)
            }
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
# #endregion

def load_evaluator(study_id: str) -> Optional[Any]:
    """
    Load and cache the evaluator module for a study.

    This function loads the evaluator module once and caches it for reuse,
    avoiding redundant file I/O and imports during bootstrap iterations.

    Args:
        study_id: Study ID (e.g. 'study_001')

    Returns:
        The evaluator module object, or None if loading failed
    """
    # #region agent log
    _debug_log("evaluator_runner.py:23", "load_evaluator called", {"study_id": study_id}, "A")
    # #endregion

    # Check cache first
    if study_id in _evaluator_cache:
        # #region agent log
        _debug_log("evaluator_runner.py:26", "Returning cached evaluator", {"study_id": study_id}, "A")
        # #endregion
        return _evaluator_cache[study_id]

    evaluator_path = Path(f"src/studies/{study_id}_evaluator.py")

    # #region agent log
    _debug_log("evaluator_runner.py:30", "Checking evaluator path", {"study_id": study_id, "evaluator_path": str(evaluator_path), "exists": evaluator_path.exists(), "absolute_path": str(evaluator_path.resolve())}, "A")
    # #endregion

    if not evaluator_path.exists():
        print(f"Evaluator not found at {evaluator_path}")
        # #region agent log
        _debug_log("evaluator_runner.py:34", "Evaluator file not found", {"study_id": study_id, "evaluator_path": str(evaluator_path)}, "A")
        # #endregion
        _evaluator_cache[study_id] = None
        return None

    try:
        # #region agent log
        _debug_log("evaluator_runner.py:38", "Starting module load", {"study_id": study_id}, "B")
        # #endregion

        # Dynamic import
        spec = importlib.util.spec_from_file_location(f"{study_id}_evaluator", evaluator_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{study_id}_evaluator"] = module

        # #region agent log
        _debug_log("evaluator_runner.py:45", "Before exec_module", {"study_id": study_id}, "B")
        # #endregion

        spec.loader.exec_module(module)

        # #region agent log
        _debug_log("evaluator_runner.py:48", "After exec_module", {"study_id": study_id, "module_has_evaluate_study": hasattr(module, "evaluate_study")}, "C")
        # #endregion

        # Verify it has the required function
        if not hasattr(module, "evaluate_study"):
            print(f"Evaluator module missing evaluate_study function")
            # #region agent log
            _debug_log("evaluator_runner.py:53", "Missing evaluate_study function", {"study_id": study_id, "module_dir": dir(module)}, "C")
            # #endregion
            _evaluator_cache[study_id] = None
            return None

        # Cache and return
        _evaluator_cache[study_id] = module
        # #region agent log
        _debug_log("evaluator_runner.py:60", "Successfully loaded evaluator", {"study_id": study_id}, "A")
        # #endregion
        return module

    except Exception as e:
        print(f"Error loading evaluator: {e}")
        import traceback
        exc_traceback = traceback.format_exc()
        # #region agent log
        _debug_log("evaluator_runner.py:67", "Exception during load", {"study_id": study_id, "error": str(e), "traceback": exc_traceback}, "B")
        # #endregion
        traceback.print_exc()
        _evaluator_cache[study_id] = None
        return None

def run_evaluator(study_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dynamically loads and runs the generated evaluator for a study.

    Args:
        study_id: Study ID (e.g. 'study_001')
        results: The agent results dictionary

    Returns:
        Dict containing score, pi_human, pi_agent, details
    """
    evaluator_path = Path(f"src/studies/{study_id}_evaluator.py")

    if not evaluator_path.exists():
        print(f"Evaluator not found at {evaluator_path}")
        return {"score": 0.0, "error": "Evaluator not found"}

    try:
        # Try to use cached module first
        module = _evaluator_cache.get(study_id)
        if module is None:
            # Load if not cached
            module = load_evaluator(study_id)
            if module is None:
                return {"score": 0.0, "error": "Failed to load evaluator"}

        # Run evaluation
        return module.evaluate_study(results)

    except Exception as e:
        print(f"Error running evaluator: {e}")
        import traceback
        traceback.print_exc()
        return {"score": 0.0, "error": str(e)}
