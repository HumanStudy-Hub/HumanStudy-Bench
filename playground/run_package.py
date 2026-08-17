#!/usr/bin/env python3
"""Run a buffer (agent-built) study package by delegating to Claude Code.

The package is self-contained (`task/adapter.py` + `evaluation/evaluation.py`).
Rather than hard-coding each package's interface, this runner hands the package
and the run request to Claude Code, which reads the adapter, injects the
researcher's model as the participant, runs the study, and evaluates it. This
handles any package interface the Build Study pipeline produces.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from playground.run_key import decrypt_api_key


def _clone_package(run: Dict[str, Any], jobs_repo: str) -> Path:
    job_id = str(run.get("jobId") or "").strip()
    slug = str(run.get("packageSlug") or "").strip()
    token = os.environ.get("PIPELINE_PROGRESS_TOKEN", "")
    if not job_id or not slug or not jobs_repo or not token:
        raise SystemExit("Buffer run is missing jobId/packageSlug/jobs repo/token; cannot locate the package.")
    destination = Path(tempfile.mkdtemp(prefix="hs-buffer-"))
    remote = f"https://x-access-token:{token}@github.com/{jobs_repo}.git"
    subprocess.run(
        ["git", "clone", "--quiet", "--depth", "1", "--branch", f"jobs/{job_id}", remote, str(destination)],
        check=True,
        timeout=300,
    )
    package = destination / "jobs" / job_id / "package" / slug
    if not package.is_dir():
        raise SystemExit(f"Buffer package not found: jobs/{job_id}/package/{slug}")
    return package


def _build_prompt(contract: Path, run_dir: Path, package: Path, run: Dict[str, Any]) -> str:
    return (
        f"{contract.read_text()}\n\n"
        "## This run\n\n"
        f"- Package directory: `{package}`\n"
        f"- Run directory: `{run_dir}`\n"
        f"- Participant model: `{run.get('model')}`\n"
        f"- Seed: `{run.get('seed')}`\n"
        f"- Participants per scenario: `{run.get('participantsPerScenario')}`\n"
        f"- Temperature: `{run.get('temperature')}`\n\n"
        "Complete the run now: drive the package's adapter with the participant "
        "model, evaluate, and write all outputs to `<run>/output/`. Then update "
        "`<run>/run.json` to complete."
    )


def _stream_output(process: subprocess.Popen, log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)


def _write_empty_outputs(run_dir: Path, note: str) -> None:
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    empty_summary = {
        "totalTests": 0, "scoredTests": 0, "replicatedTests": 0,
        "replicationRate": None, "directionMatchRate": None,
        "meanAbsoluteEffectGap": None, "meanHumanEffect": None,
        "meanAgentEffect": None, "effectCorrelation": None, "studyScore": None,
    }
    (output_dir / "evaluation.json").write_text(json.dumps({"status": "not_ready", "missing_requirement": note}, indent=2) + "\n")
    (output_dir / "sessions.json").write_text(json.dumps([], indent=2) + "\n")
    (output_dir / "analysis.json").write_text(json.dumps({"summary": empty_summary, "tests": []}, indent=2) + "\n")
    (output_dir / "charts.json").write_text(json.dumps({"charts": [], "source": "default"}, indent=2) + "\n")
    (output_dir / "transcript_sample.json").write_text(json.dumps([], indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path, help="Run directory containing run.json")
    parser.add_argument("--package-path", type=Path, help="Buffer package directory (cloned when omitted)")
    parser.add_argument("--progress-repo")
    parser.add_argument("--progress-branch")
    parser.add_argument("--progress-path")
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-5"))
    parser.add_argument("--timeout-minutes", type=float, default=25)
    args = parser.parse_args()

    run_dir = args.run.resolve()
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    package = args.package_path.resolve() if args.package_path else _clone_package(run, args.progress_repo or "")

    output_dir = run_dir / "output"
    logs_dir = run_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    own_key = decrypt_api_key(run.get("sealedApiKey"))
    shared_key = os.environ.get("OPENROUTER_API_KEY", "")
    participant_key = own_key or shared_key
    if not participant_key:
        raise SystemExit("No OpenRouter API key is available for this run.")

    contract = REPO_ROOT / "playground" / "package_runner.md"
    prompt = _build_prompt(contract, run_dir, package, run)

    agent_env = os.environ.copy()
    for secret in ("PIPELINE_PROGRESS_TOKEN", "HUMANSTUDY_PIPELINE_TOKEN", "GITHUB_TOKEN", "PLAYGROUND_KEY_SECRET"):
        agent_env.pop(secret, None)
    # Claude Code itself runs through OpenRouter; the participant model inside
    # the agent's Python script also reads OPENROUTER_API_KEY.
    agent_env["ANTHROPIC_BASE_URL"] = "https://openrouter.ai/api"
    agent_env["ANTHROPIC_AUTH_TOKEN"] = shared_key
    agent_env["ANTHROPIC_API_KEY"] = ""
    agent_env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = args.model
    agent_env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = args.model
    agent_env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = args.model
    agent_env["CLAUDE_CODE_SUBAGENT_MODEL"] = args.model
    agent_env["OPENROUTER_MODEL"] = args.model
    agent_env["OPENROUTER_API_KEY"] = participant_key

    process = subprocess.Popen(
        [
            "claude",
            "--print",
            "--model", args.model,
            "--add-dir", str(run_dir),
            "--add-dir", str(package),
            "--dangerously-skip-permissions",
            prompt,
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=agent_env,
    )
    output_thread = threading.Thread(target=_stream_output, args=(process, logs_dir / "run.log"), daemon=True)
    output_thread.start()
    deadline = time.monotonic() + args.timeout_minutes * 60

    try:
        while process.poll() is None:
            if time.monotonic() >= deadline:
                print("[package] the run agent ran out of time", flush=True)
                _stop_process(process)
                break
            time.sleep(5)
    finally:
        _stop_process(process)
        output_thread.join(timeout=5)

    if not (output_dir / "evaluation.json").exists():
        _write_empty_outputs(run_dir, "The run agent did not produce an evaluation result.")

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if run.get("status") != "complete":
        run.update({
            "status": "complete",
            "message": "The run finished and the results are ready",
            "resultsReady": True,
        })
        (run_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    print(json.dumps({"status": "complete"}, default=str))


if __name__ == "__main__":
    main()
