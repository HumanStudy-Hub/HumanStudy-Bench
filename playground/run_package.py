#!/usr/bin/env python3
"""Run a buffer (agent-built) study package with a chosen model.

Buffer packages are self-contained: `task/adapter.py` exposes the runnable task
(typically a `run_session(...)` that accepts an `agent_fn`), and
`evaluation/evaluation.py` exposes the paper's checks. This runner injects the
researcher's model as the agent and scores with the package's own evaluator —
the package format is untouched.

Each package documents its own agent contract in `task/task.json`; the small
`RUN_SPECS` table below is the per-package wiring that plugs a generic LLM agent
into that contract. It is wiring only, not a scoring implementation.
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
from typing import Any, Callable, Dict, List, Optional

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


def _llm_text(model: str, api_key: str, messages: List[Dict[str, str]], temperature: float) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=settings.OPENROUTER_API_BASE, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=500,
    )
    return (response.choices[0].message.content or "").strip()


def _simulated_reply(spec: Dict[str, Any]) -> str:
    """Deterministic mock used by --simulate so the wiring runs with no API."""
    return spec.get("simulate_reply", "A")


def build_llm_agent(
    spec: Dict[str, Any],
    model: str,
    api_key: str,
    temperature: float,
    simulate: bool,
) -> Callable[..., Any]:
    """Return an `agent_fn` compatible with the package's own contract.

    `spec` describes how to render the package's input into a prompt and how to
    parse the model's reply back into the return value the adapter expects.
    """
    def agent(*args: Any, **kwargs: Any) -> Any:
        prompt = spec["prompt"](args, kwargs)
        if simulate:
            reply = _simulated_reply(spec)
        else:
            reply = _llm_text(model, api_key, [{"role": "user", "content": prompt}], temperature)
        return spec["parse"](reply, args, kwargs)
    return agent


# ---------------------------------------------------------------------------
# Per-package wiring. `detect` returns True when an adapter matches; `run_spec`
# and `agent_spec` tell the runner how to drive it and how to build the LLM
# agent. This is the only place package-specific knowledge lives.
# ---------------------------------------------------------------------------

def _cascade_prompt(args: tuple, kwargs: dict) -> str:
    state = args[0] if args else {}
    instructions = state.get("instructions_text", "")
    private = state.get("private_draw", "")
    history = json.dumps(state.get("decisions_so_far_this_period", []))
    return (
        f"{instructions}\n\n"
        f"This period, your private draw is: {private}.\n"
        f"Decisions made so far this period (earliest first): {history}\n\n"
        "Based on the rules above and everything you have observed, which urn "
        "do you now believe is more likely? Answer with exactly one letter: A or B."
    )


def _cascade_parse(reply: str, args: tuple, kwargs: dict) -> dict:
    text = (reply or "").strip().upper()
    decision = "A" if ("A" in text and "B" not in text) or text.startswith("A") else "B"
    return {"decision": decision}


def _standard_prompt(args: tuple, kwargs: dict) -> str:
    state = args[0] if args else {}
    return (
        json.dumps(state, indent=2, ensure_ascii=False)
        + "\n\nRespond with a single JSON object containing your action."
    )


def _standard_parse(reply: str, args: tuple, kwargs: dict) -> dict:
    text = (reply or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
        return {}


RUN_SPECS: List[Dict[str, Any]] = [
    {
        "name": "anderson_holt_1997_information_cascades",
        "detect": lambda adapter: hasattr(adapter, "run_session") and hasattr(adapter, "CONDITIONS") and hasattr(adapter, "_baseline_follow_private_signal"),
        "conditions": lambda adapter: list(adapter.CONDITIONS.keys()),
        "run": lambda adapter, agent, study, condition, seed, participants: adapter.run_session(
            agent, study, condition, n_periods=participants, seed=seed, session_id=f"{study}:{condition}"
        ),
        "evaluate": lambda evaluation, runs: evaluation.evaluate(runs),
        "agent_spec": {
            "prompt": _cascade_prompt,
            "parse": _cascade_parse,
            "simulate_reply": "A",
        },
    },
]


def _find_run_spec(adapter: Any) -> Optional[Dict[str, Any]]:
    for spec in RUN_SPECS:
        if spec["detect"](adapter):
            return spec
    return None


def _clone_package(run: Dict[str, Any], jobs_repo: str) -> Path:
    """Clone the buffer package's job branch from the private jobs repository."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path, help="Run directory containing run.json")
    parser.add_argument("--package-path", type=Path, help="Buffer package directory (cloned from the job branch when omitted)")
    parser.add_argument("--progress-repo")
    parser.add_argument("--progress-branch")
    parser.add_argument("--progress-path")
    parser.add_argument("--simulate", action="store_true", help="Use deterministic mock replies instead of the LLM")
    args = parser.parse_args()

    run_dir = args.run.resolve()
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    package = args.package_path.resolve() if args.package_path else _clone_package(run, args.progress_repo or "")

    adapter = _import_module(package / "task" / "adapter.py", "buffer_adapter")
    evaluation = _import_module(package / "evaluation" / "evaluation.py", "buffer_evaluation")

    api_key = decrypt_api_key(run.get("sealedApiKey"))
    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not args.simulate:
        raise SystemExit("No OpenRouter API key is available for this run.")

    model = str(run.get("model") or settings.DEFAULT_MODEL)
    temperature = float(run.get("temperature") or 1.0)
    participants = int(run.get("participantsPerScenario") or 8)
    seed = int(run.get("seed") or 42)

    # Standard contract (current builds): adapter.run_sessions(agent_fn, seed)
    # -> sessions, evaluation.evaluate(sessions) -> result. Needs no wiring.
    if hasattr(adapter, "run_sessions") and hasattr(evaluation, "evaluate"):
        agent_spec = {"prompt": _standard_prompt, "parse": _standard_parse, "simulate_reply": "{}"}
        agent = build_llm_agent(agent_spec, model, api_key, temperature, args.simulate)
        runs = adapter.run_sessions(agent, seed)
        result = evaluation.evaluate(runs)
    else:
        # Legacy packages: drive them through their own per-package wiring.
        spec = _find_run_spec(adapter)
        if spec is None:
            raise SystemExit(
                f"Package {package.name} does not match a known run contract. "
                "Only agent_fn-style packages are supported; this package needs a wiring entry in RUN_SPECS."
            )
        agent = build_llm_agent(spec["agent_spec"], model, api_key, temperature, args.simulate)
        runs = []
        for study, condition in spec["conditions"](adapter):
            session = spec["run"](adapter, agent, study, condition, seed, participants)
            runs.append(session)
        result = spec["evaluate"](evaluation, runs)

    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    (output_dir / "sessions.json").write_text(json.dumps(runs, indent=2, default=str) + "\n")

    run.update({
        "status": "complete",
        "message": "The run finished and the results are ready",
        "resultsReady": True,
    })
    (run_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    print(json.dumps({"status": "complete", "evaluation": result}, default=str))


if __name__ == "__main__":
    main()
