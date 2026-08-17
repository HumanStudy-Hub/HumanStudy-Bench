#!/usr/bin/env python3
"""Run a buffer (agent-built) study package with a chosen model.

Buffer packages expose the standard harness interface:

- `run_sessions(llm, seed) -> list[dict]`, where `llm(prompt: str) -> str` is the
  injected model call;
- `evaluate(sessions) -> dict`.

New packages put `run_sessions` in `task/adapter.py` and `evaluate` in
`evaluation/evaluation.py`. Legacy packages expose them in `task/run_sessions.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from playground import settings
from playground.run_key import decrypt_api_key


def _import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    for directory in (path.parent, path.parent.parent):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    spec.loader.exec_module(module)
    return module


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


def _make_llm(model: str, api_key: str, temperature: float) -> Callable[[str], str]:
    from openai import OpenAI
    client = OpenAI(base_url=settings.OPENROUTER_API_BASE, api_key=api_key)

    def llm(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=800,
        )
        return (response.choices[0].message.content or "").strip()
    return llm


def _write_outputs(run_dir: Path, result: Any, sessions: Any) -> None:
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    (output_dir / "sessions.json").write_text(json.dumps(sessions, indent=2, default=str) + "\n")
    empty_summary = {
        "totalTests": 0, "scoredTests": 0, "replicatedTests": 0,
        "replicationRate": None, "directionMatchRate": None,
        "meanAbsoluteEffectGap": None, "meanHumanEffect": None,
        "meanAgentEffect": None, "effectCorrelation": None, "studyScore": None,
    }
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
    args = parser.parse_args()

    run_dir = args.run.resolve()
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    package = args.package_path.resolve() if args.package_path else _clone_package(run, args.progress_repo or "")

    adapter = _import_module(package / "task" / "adapter.py", "buffer_adapter")
    evaluation = _import_module(package / "evaluation" / "evaluation.py", "buffer_evaluation")

    own_key = decrypt_api_key(run.get("sealedApiKey"))
    api_key = own_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("No OpenRouter API key is available for this run.")

    model = str(run.get("model") or settings.DEFAULT_MODEL)
    temperature = float(run.get("temperature") or 1.0)
    seed = int(run.get("seed") or 42)
    llm = _make_llm(model, api_key, temperature)

    n = int(run.get("participantsPerScenario") or 8)
    if hasattr(adapter, "run_sessions") and hasattr(evaluation, "evaluate"):
        sessions = adapter.run_sessions(llm, seed, n)
        result = evaluation.evaluate(sessions)
    else:
        shim = _import_module(package / "task" / "run_sessions.py", "buffer_run_sessions")
        sessions = shim.run_sessions(llm, seed, n)
        result = shim.evaluate(sessions)

    _write_outputs(run_dir, result, sessions)
    run.update({
        "status": "complete",
        "message": "The run finished and the results are ready",
        "resultsReady": True,
    })
    (run_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    print(json.dumps({"status": "complete", "evaluation": result}, default=str))


if __name__ == "__main__":
    main()
