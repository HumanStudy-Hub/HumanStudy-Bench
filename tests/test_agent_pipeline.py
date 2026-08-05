import json
import os
import subprocess
import sys
from pathlib import Path

from agent_pipeline.run_agent import ProgressPublisher, package_progress


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


def test_build_prompt_disables_search_without_user_url(tmp_path: Path) -> None:
    job = tmp_path / "jobs" / "paper-only"
    (job / "input").mkdir(parents=True)
    (job / "input" / "paper.pdf").write_bytes(b"%PDF-test")
    (job / "job.json").write_text(json.dumps({
        "paperName": "paper.pdf",
        "contributorName": "Researcher",
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
    assert "Network research is not authorized" in prompt
    assert "Use only the uploaded PDF" in prompt


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


def test_watchdog_progress_counts_required_files(tmp_path: Path) -> None:
    package = tmp_path / "package"
    study = package / "paper-name"
    (study / "source").mkdir(parents=True)
    (study / "README.md").write_text("Package\n")
    (study / "study.json").write_text("{}\n")
    (study / "source/paper_metadata.json").write_text("{}\n")
    (study / "extra.txt").write_text("Additional material\n")

    completed, total, missing = package_progress(package)

    assert completed == 3
    assert total == 4
    assert "task/adapter.py" in missing


def test_progress_publisher_updates_existing_progress(monkeypatch) -> None:
    calls = []

    def fake_request(self, method, body=None):
        calls.append((method, body))
        return {"sha": "old-sha"} if method == "GET" else {"content": {"sha": "new-sha"}}

    monkeypatch.setattr(ProgressPublisher, "_request", fake_request)
    publisher = ProgressPublisher("secret", "owner/jobs", "jobs/example", "jobs/example/progress.json")
    publisher.publish({"completedRequired": 4, "totalRequired": 13})

    assert publisher.sha == "new-sha"
    assert calls[1][1]["sha"] == "old-sha"
    assert calls[1][1]["branch"] == "jobs/example"


def test_watchdog_stops_agent_after_validation(tmp_path: Path) -> None:
    job = tmp_path / "job"
    (job / "logs").mkdir(parents=True)
    prompt = job / "prompt.md"
    prompt.write_text("Build the package")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_claude = bin_dir / "claude"
    fake_claude.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys, time\n"
        "job = pathlib.Path(sys.argv[sys.argv.index('--add-dir') + 1])\n"
        "(job / 'claude-env.json').write_text(json.dumps({'pipeline_token_visible': 'PIPELINE_PROGRESS_TOKEN' in os.environ}))\n"
        "root = job / 'package' / 'paper'\n"
        f"required = {REQUIRED!r}\n"
        "for relative in required:\n"
        "    path = root / relative\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    if relative == 'task/adapter.py':\n"
        "        path.write_text(\"import sys\\nraise SystemExit(0 if '--smoke-test' in sys.argv else 1)\\n\")\n"
        "    elif path.suffix == '.json':\n"
        "        path.write_text(json.dumps({}) + '\\n')\n"
        "    else:\n"
        "        path.write_text('Generated\\n')\n"
        "print('package written; waiting forever', flush=True)\n"
        "time.sleep(60)\n"
    )
    fake_claude.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "PIPELINE_PROGRESS_TOKEN": "private-test-token"}

    result = subprocess.run([
        sys.executable,
        str(ROOT / "agent_pipeline/run_agent.py"),
        "--job",
        str(job),
        "--prompt",
        str(prompt),
        "--model",
        "test-model",
        "--validator",
        str(ROOT / "agent_pipeline/validate_package.py"),
        "--timeout-minutes",
        "0.1",
        "--check-interval",
        "0.05",
    ], capture_output=True, text=True, env=env, timeout=10)

    assert result.returncode == 0, result.stdout + result.stderr
    watchdog = json.loads((job / "logs/watchdog.json").read_text())
    assert watchdog["reason"] == "validator_passed"
    assert watchdog["package_valid"] is True
    assert json.loads((job / "claude-env.json").read_text())["pipeline_token_visible"] is False
