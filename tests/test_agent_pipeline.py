import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "study.json",
    "source/paper_metadata.json",
    "source/extraction.json",
    "source/evidence.json",
    "source/open_materials.json",
    "materials/materials.json",
    "task/task.json",
    "task/adapter.py",
    "evaluation/evaluation.py",
    "audit/provenance.json",
    "audit/missing_information.json",
    "audit/agent_report.md",
)


def test_build_prompt_includes_job_inputs(tmp_path: Path) -> None:
    job = tmp_path / "jobs" / "example"
    (job / "input").mkdir(parents=True)
    (job / "input" / "paper.pdf").write_bytes(b"%PDF-test")
    (job / "job.json").write_text(json.dumps({
        "paperName": "paper.pdf",
        "contributorName": "Researcher",
        "osfUrl": "https://osf.io/example/",
    }))
    output = job / "agent-prompt.md"

    subprocess.run([
        sys.executable,
        str(ROOT / "agent_pipeline/build_prompt.py"),
        "--contract",
        str(ROOT / "agent_pipeline/CLAUDE.md"),
        "--job",
        str(job),
        "--output",
        str(output),
    ], check=True)

    prompt = output.read_text()
    assert "https://osf.io/example/" in prompt
    assert str((job / "input/paper.pdf").resolve()) in prompt
    assert "Do not invent study facts" in prompt


def test_validate_complete_agent_package(tmp_path: Path) -> None:
    package = tmp_path / "package"
    study = package / "paper-name"
    for relative in REQUIRED:
        path = study / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text("{}\n")
        elif relative == "task/adapter.py":
            path.write_text("import sys\nraise SystemExit(0 if '--smoke-test' in sys.argv else 1)\n")
        else:
            path.write_text("Generated test file\n")

    result = subprocess.run([
        sys.executable,
        str(ROOT / "agent_pipeline/validate_package.py"),
        str(package),
    ], capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Validated agent package" in result.stdout


def test_validate_rejects_missing_file(tmp_path: Path) -> None:
    study = tmp_path / "package" / "paper-name"
    study.mkdir(parents=True)

    result = subprocess.run([
        sys.executable,
        str(ROOT / "agent_pipeline/validate_package.py"),
        str(tmp_path / "package"),
    ], capture_output=True, text=True)

    assert result.returncode != 0
    assert "missing required package files" in result.stderr
