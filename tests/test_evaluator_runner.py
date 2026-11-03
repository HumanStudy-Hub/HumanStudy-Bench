from pathlib import Path

from src.evaluation import evaluator_runner

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_study(root: Path, study_id: str, body: str) -> None:
    scripts = root / "studies" / study_id / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "helper.py").write_text("FACTOR = 2\n")
    (scripts / "evaluator.py").write_text(body)


def test_loads_real_study_evaluator_from_studies_folder():
    module = evaluator_runner.load_evaluator("study_001")
    assert module is not None
    assert hasattr(module, "evaluate_study")
    assert Path(module.__file__).resolve() == (
        REPO_ROOT / "studies" / "study_001" / "scripts" / "evaluator.py"
    )


def test_runs_evaluator_with_sibling_imports(tmp_path):
    write_study(
        tmp_path,
        "study_901",
        "from helper import FACTOR\n"
        "def evaluate_study(results):\n"
        "    return {'score': results['x'] * FACTOR}\n",
    )
    assert evaluator_runner.run_evaluator("study_901", {"x": 0.25}, repo_root=tmp_path) == {
        "score": 0.5
    }


def test_missing_study_returns_error(tmp_path):
    (tmp_path / "studies").mkdir()
    result = evaluator_runner.run_evaluator("study_999", {}, repo_root=tmp_path)
    assert result["score"] == 0.0
    assert "error" in result


def test_evaluator_exception_returns_error(tmp_path):
    write_study(
        tmp_path,
        "study_902",
        "def evaluate_study(results):\n    raise ValueError('boom')\n",
    )
    result = evaluator_runner.run_evaluator("study_902", {}, repo_root=tmp_path)
    assert result == {"score": 0.0, "error": "boom"}
