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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from playground import settings
from playground.progress import ProgressWriter
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


def _load_cache(run_dir: Path) -> Dict[str, str]:
    path = run_dir / "output" / "llm_cache.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_cache(run_dir: Path, cache: Dict[str, str]) -> None:
    (run_dir / "output").mkdir(parents=True, exist_ok=True)
    (run_dir / "output" / "llm_cache.json").write_text(json.dumps(cache), encoding="utf-8")


def _make_llm(model: str, api_key: str, temperature: float, on_step=None, cache: Optional[Dict[str, str]] = None, save_cache=None) -> Callable[..., str]:
    import threading
    from openai import OpenAI
    client = OpenAI(base_url=settings.OPENROUTER_API_BASE, api_key=api_key)
    counter = [0]
    lock = threading.Lock()

    def llm(prompt: str, key: Optional[str] = None) -> str:
        with lock:
            counter[0] += 1
            cache_key = key if key is not None else f"idx:{counter[0]}"
            step = counter[0]
        if on_step:
            on_step(step)
        if cache is not None and cache_key in cache:
            return cache[cache_key]
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=800,
        )
        result = (response.choices[0].message.content or "").strip()
        if cache is not None:
            with lock:
                cache[cache_key] = result
            if save_cache:
                save_cache(cache)
        return result
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

    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = (logs_dir / "run.log").open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    adapter = _import_module(package / "task" / "adapter.py", "buffer_adapter")
    evaluation = _import_module(package / "evaluation" / "evaluation.py", "buffer_evaluation")

    own_key = decrypt_api_key(run.get("sealedApiKey"))
    api_key = own_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("No OpenRouter API key is available for this run.")

    model = str(run.get("model") or settings.DEFAULT_MODEL)
    temperature = float(run.get("temperature") or 1.0)
    seed = int(run.get("seed") or 42)

    log(f"Running {package.name} with {model} (n={run.get('participantsPerScenario') or 8})")
    progress = ProgressWriter(run_dir, args.progress_repo, args.progress_branch, args.progress_path)
    progress.write({"phase": "preparing", "completedTrials": 0, "totalTrials": 0, "message": "Loading the study"}, force=True)

    def on_step(count: int) -> None:
        log(f"{count} model calls so far")
        progress.write({"phase": "running_participants", "completedTrials": count, "totalTrials": count, "message": f"{count} model calls so far"})

    cache = _load_cache(run_dir)
    llm = _make_llm(model, api_key, temperature, on_step, cache, lambda c: _save_cache(run_dir, c))

    n = int(run.get("participantsPerScenario") or 8)
    if hasattr(adapter, "run_sessions") and hasattr(evaluation, "evaluate"):
        sessions = adapter.run_sessions(llm, seed, n)
        log("Scoring against the published findings")
        progress.write({"phase": "scoring", "completedTrials": 0, "totalTrials": 0, "message": "Scoring against the published findings"}, force=True)
        result = evaluation.evaluate(sessions)
    else:
        shim = _import_module(package / "task" / "run_sessions.py", "buffer_run_sessions")
        sessions = shim.run_sessions(llm, seed, n)
        log("Scoring against the published findings")
        progress.write({"phase": "scoring", "completedTrials": 0, "totalTrials": 0, "message": "Scoring against the published findings"}, force=True)
        result = shim.evaluate(sessions)

    _write_outputs(run_dir, result, sessions)
    run.update({
        "status": "complete",
        "message": "The run finished and the results are ready",
        "resultsReady": True,
    })
    (run_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    log("Run complete")

    print(json.dumps({"status": "complete", "evaluation": result}, default=str))


if __name__ == "__main__":
    main()
