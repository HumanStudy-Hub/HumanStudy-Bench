#!/usr/bin/env python3
"""Run a buffer (agent-built) study package with a chosen model.

Buffer packages expose the standard harness interface:

- `run_sessions(llm, seed, n, arms=None, on_session=None) -> list[dict]`, where
  `llm(prompt: str, key=None) -> str` is the injected model call;
- `evaluate(sessions) -> dict`.

New packages put `run_sessions` in `task/adapter.py` and `evaluate` in
`evaluation/evaluation.py`. Legacy packages expose them in `task/run_sessions.py`.

The runner saves sessions as they complete, so a run that is stopped early (the
researcher cancels it, or the Actions time limit hits) still finalizes the
sessions produced so far into `evaluation.json`, `analysis.json`, and
`charts.json` — a quick-iteration preview without re-running anything.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import inspect
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
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


def _cache_path(run_dir: Path, selection: Dict[str, Any]) -> Path:
    """A scoped run keeps its own cache so its call-order keys never collide with
    a whole run (or another arm) sharing the same run directory."""
    material_id = str(selection.get("materialId") or "").strip() if isinstance(selection, dict) else ""
    if material_id:
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", material_id)[:60] or "scoped"
        return run_dir / "output" / f"llm_cache_{safe_id}.json"
    return run_dir / "output" / "llm_cache.json"


def _load_cache(run_dir: Path, selection: Dict[str, Any]) -> Dict[str, str]:
    path = _cache_path(run_dir, selection)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_cache(run_dir: Path, cache: Dict[str, str], selection: Dict[str, Any]) -> None:
    path = _cache_path(run_dir, selection)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache), encoding="utf-8")


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


def _count_numbers(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_count_numbers(child) for child in value.values())
    if isinstance(value, list):
        return sum(_count_numbers(child) for child in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 1
    return 0


def build_buffer_analysis(sessions: Any, evaluation: Any) -> Dict[str, Any]:
    numeric = _count_numbers(evaluation)
    summary = {
        "totalTests": numeric,
        "scoredTests": numeric,
        "replicatedTests": 0,
        "replicationRate": None,
        "directionMatchRate": None,
        "meanAbsoluteEffectGap": None,
        "meanHumanEffect": None,
        "meanAgentEffect": None,
        "effectCorrelation": None,
        "studyScore": None,
    }
    return {"summary": summary, "tests": [], "sessions": len(sessions) if isinstance(sessions, list) else 0}


def _coverage(sessions: Any) -> Dict[str, int]:
    """Count completed sessions per condition/arm, so the UI can show which
    conditions a (possibly early-stopped) whole run actually covered."""
    coverage: Dict[str, int] = {}
    for entry in (sessions if isinstance(sessions, list) else []):
        if not isinstance(entry, dict):
            continue
        for field in ("arm", "condition", "condition_id", "study_id", "culture"):
            value = entry.get(field)
            if value is not None:
                key = str(value)
                coverage[key] = coverage.get(key, 0) + 1
                break
    return coverage


def _bar_chart(chart_id: str, title: str, description: str, buckets: Dict[str, Any]) -> Dict[str, Any]:
    """A grouped bar chart of the numeric scalar fields across a dict of buckets."""
    numeric_fields: List[str] = []
    seen = set()
    for _key, value in buckets.items():
        if not isinstance(value, dict):
            continue
        for field, entry in value.items():
            if isinstance(entry, (int, float)) and not isinstance(entry, bool) and field not in seen:
                seen.add(field)
                numeric_fields.append(field)
    numeric_fields = numeric_fields[:6]
    keys = sorted(buckets.keys())
    traces = []
    for field in numeric_fields:
        traces.append({
            "type": "bar",
            "name": field,
            "x": keys,
            "y": [buckets.get(key, {}).get(field) if isinstance(buckets.get(key), dict) else None for key in keys],
            "hovertemplate": f"{field}: %{{y}}<extra></extra>",
        })
    return {
        "id": chart_id,
        "title": title,
        "description": description,
        "plotly": {"data": traces, "layout": {"barmode": "group"}},
    }


def build_buffer_charts(evaluation: Any) -> Dict[str, Any]:
    """Render the package's own evaluate() output into plottable charts.

    This is deliberately generic: it charts whatever numeric scalar fields the
    evaluator returned, grouped by arm and by cross-arm comparison, so a buffer
    run gets a readable preview even though its evaluator shape is package-
    specific.
    """
    charts: List[Dict[str, Any]] = []
    if not isinstance(evaluation, dict):
        return {"charts": [], "source": "default"}
    if "not_ready" in evaluation:
        raw = evaluation.get("not_ready")
        reason = str(raw.get("reason", "not ready")) if isinstance(raw, dict) else "not ready"
        charts.append({
            "id": "not_ready", "title": "Not scoreable", "description": reason,
            "plotly": {"data": [], "layout": {"title": "Not scoreable"}},
        })
        return {"charts": charts, "source": "default"}
    by_arm = evaluation.get("by_arm")
    if isinstance(by_arm, dict) and by_arm:
        charts.append(_bar_chart("by_arm", "By condition", "Numeric evaluator metrics per condition.", by_arm))
    comparisons = evaluation.get("cross_arm_comparisons")
    if isinstance(comparisons, dict) and comparisons:
        charts.append(_bar_chart("comparisons", "Cross-condition comparisons", "Paired evaluator values across conditions.", comparisons))
    if not charts:
        charts.append({
            "id": "empty", "title": "No numeric results", "description": "The evaluator returned no numeric metrics to chart.",
            "plotly": {"data": [], "layout": {"title": "No numeric results"}},
        })
    return {"charts": charts[:6], "source": "default"}


def _write_results(run_dir: Path, sessions: Any, evaluate_fn: Callable[[Any], Any]) -> Any:
    """Run the evaluator and write the result files, without touching run.json."""
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_fn(sessions)
    (output_dir / "evaluation.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    (output_dir / "sessions.json").write_text(json.dumps(sessions, indent=2, default=str) + "\n")
    (output_dir / "coverage.json").write_text(json.dumps(_coverage(sessions), indent=2) + "\n")
    (output_dir / "analysis.json").write_text(json.dumps(build_buffer_analysis(sessions, result), indent=2, default=str) + "\n")
    (output_dir / "charts.json").write_text(json.dumps(build_buffer_charts(result), indent=2, default=str) + "\n")
    (output_dir / "transcript_sample.json").write_text(json.dumps([], indent=2) + "\n")
    return result


class OutputPublisher:
    """Mirror result files to the jobs repository through the GitHub contents API.

    The workflow's final git commit can race with a stopped run, so results are
    also written here directly (sha-tracked, throttled) — the same mechanism the
    progress writer already uses. This is what makes "stop and view partial
    results" survive a cancelled Actions job.
    """

    def __init__(self, repo: str, branch: str, token: str, run_dir: Path, min_interval: float = 8.0) -> None:
        self.repo = repo
        self.branch = branch
        self.token = token
        self.run_dir = run_dir
        self.min_interval = min_interval
        self.shas: Dict[str, str] = {}
        self.last: Dict[str, float] = {}

    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "HumanStudy-Hub-Playground",
        }

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        url = f"https://api.github.com/repos/{self.repo}/contents/{urllib.parse.quote(path, safe='/')}"
        if method == "GET":
            url += "?ref=" + urllib.parse.quote(self.branch, safe="")
        request = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, method=method, headers=self._headers())
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read())

    def publish(self, rel_path: str, content: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last.get(rel_path, 0.0) < self.min_interval:
            return
        self.last[rel_path] = now
        path = f"{self.run_dir.name}/{rel_path}"
        sha = self.shas.get(path)
        if sha is None:
            try:
                sha = self._request("GET", path).get("sha")
            except Exception:
                sha = None
        body = {"message": f"playground: output {rel_path}", "branch": self.branch, "content": base64.b64encode(content.encode()).decode()}
        if sha:
            body["sha"] = sha
        try:
            response = self._request("PUT", path, body)
            self.shas[path] = response["content"]["sha"]
        except urllib.error.HTTPError as exc:
            if exc.code == 409:
                self.shas.pop(path, None)
            print(f"[output] could not publish {rel_path}: {exc}", flush=True)
        except Exception as exc:
            print(f"[output] could not publish {rel_path}: {exc}", flush=True)

    def publish_results(self, force: bool = False) -> None:
        output_dir = self.run_dir / "output"
        for name in ("evaluation.json", "sessions.json", "coverage.json", "analysis.json", "charts.json"):
            path = output_dir / name
            if path.exists():
                self.publish(f"output/{name}", path.read_text(encoding="utf-8"), force=force)


def _finalize(run_dir: Path, sessions: Any, evaluate_fn: Callable[[Any], Any], run: Dict[str, Any], partial: bool, log) -> Any:
    result = _write_results(run_dir, sessions, evaluate_fn)
    if partial:
        run.update({
            "status": "complete",
            "message": "The run was stopped early; partial results are shown",
            "resultsReady": True,
            "partial": True,
            "participants": len(sessions) if isinstance(sessions, list) else 0,
        })
    else:
        run.update({
            "status": "complete",
            "message": "The run finished and the results are ready",
            "resultsReady": True,
            "participants": len(sessions) if isinstance(sessions, list) else 0,
        })
    (run_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    log(f"Finalized {'partial' if partial else 'full'} results from {len(sessions) if isinstance(sessions, list) else 0} sessions")
    return result


def _call_run_sessions(fn: Callable[..., Any], llm: Callable[..., str], seed: int, n: int, arms: Optional[List[str]], on_session: Optional[Callable[[Any], None]]) -> List[dict]:
    """Run the package's run_sessions, honouring optional arms and on_session."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs: Dict[str, Any] = {}
    if arms is not None:
        if "arms" not in params:
            raise SystemExit("This study package does not support arm-scoped runs yet. Run the whole study instead.")
        kwargs["arms"] = arms
    if on_session is not None and "on_session" in params:
        kwargs["on_session"] = on_session
    return fn(llm, seed, n, **kwargs)


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
    selection = run.get("selection") if isinstance(run.get("selection"), dict) else {}
    arms = None
    if selection.get("mode") == "material" and selection.get("materialId"):
        arms = [str(selection["materialId"])]
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
    if arms:
        log(f"Scoped run to arm(s): {', '.join(arms)}")
    progress = ProgressWriter(run_dir, args.progress_repo, args.progress_branch, args.progress_path)
    progress.write({"phase": "preparing", "completedTrials": 0, "totalTrials": 0, "message": "Loading the study"}, force=True)

    last_step_log = [0.0]

    def on_step(count: int) -> None:
        # A cache-hit resume can complete calls many times a second; throttling
        # here keeps the log and the progress API from being flooded.
        now = time.monotonic()
        if now - last_step_log[0] < 2.0:
            return
        last_step_log[0] = now
        log(f"{count} model calls so far")
        progress.write({"phase": "running_participants", "completedTrials": count, "totalTrials": count, "message": f"{count} model calls so far"})

    cache = _load_cache(run_dir, selection)
    llm = _make_llm(model, api_key, temperature, on_step, cache, lambda c: _save_cache(run_dir, c, selection))

    n = int(run.get("participantsPerScenario") or 8)
    if hasattr(adapter, "run_sessions") and hasattr(evaluation, "evaluate"):
        run_sessions_fn = adapter.run_sessions
        evaluate_fn = evaluation.evaluate
    else:
        shim = _import_module(package / "task" / "run_sessions.py", "buffer_run_sessions")
        run_sessions_fn = shim.run_sessions
        evaluate_fn = shim.evaluate

    sessions_so_far: List[dict] = []
    publisher: Optional[OutputPublisher] = None
    progress_token = os.environ.get("PIPELINE_PROGRESS_TOKEN", "")
    if progress_token and args.progress_repo and args.progress_branch:
        publisher = OutputPublisher(args.progress_repo, args.progress_branch, progress_token, run_dir)

    last_result = [0.0]

    def on_session(session: Any) -> None:
        sessions_so_far.append(session)
        (run_dir / "output").mkdir(parents=True, exist_ok=True)
        (run_dir / "output" / "sessions.json").write_text(json.dumps(sessions_so_far, indent=2, default=str) + "\n")
        (run_dir / "output" / "coverage.json").write_text(json.dumps(_coverage(sessions_so_far), indent=2) + "\n")
        progress.write({"phase": "running_participants", "completedTrials": len(sessions_so_far), "totalTrials": len(sessions_so_far), "message": f"{len(sessions_so_far)} sessions complete"})
        # Keep results fresh on disk and in the repo so a hard stop still shows
        # a plot: evaluate the sessions so far (throttled) and publish them.
        now = time.monotonic()
        if now - last_result[0] >= 8.0:
            last_result[0] = now
            try:
                _write_results(run_dir, sessions_so_far, evaluate_fn)
                if publisher:
                    publisher.publish_results()
            except Exception as exc:
                log(f"Could not refresh partial results: {exc}")

    # A cancelled run (researcher stop, or the Actions time limit) finalizes the
    # sessions completed so far instead of throwing everything away.
    def stop_handler(_signum, _frame) -> None:
        log("Stop requested; finalizing partial results")
        try:
            _finalize(run_dir, sessions_so_far, evaluate_fn, run, partial=True, log=log)
            if publisher:
                publisher.publish_results(force=True)
        except Exception as exc:
            log(f"Could not finalize partial results: {exc}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    sessions = _call_run_sessions(run_sessions_fn, llm, seed, n, arms, on_session)
    progress.write({"phase": "scoring", "completedTrials": len(sessions), "totalTrials": len(sessions), "message": "Scoring against the published findings"}, force=True)
    result = _finalize(run_dir, sessions, evaluate_fn, run, partial=False, log=log)
    if publisher:
        publisher.publish_results(force=True)

    print(json.dumps({"status": "complete", "evaluation": result}, default=str))


if __name__ == "__main__":
    main()
