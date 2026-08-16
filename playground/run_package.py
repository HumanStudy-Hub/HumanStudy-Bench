#!/usr/bin/env python3
"""Run a buffer (agent-built) study package with a chosen model.

Buffer packages are self-contained: `task/adapter.py` exposes the runnable task
and `evaluation/evaluation.py` exposes the paper's checks. This runner injects
the researcher's model as the participant and scores with the package's own
evaluator — the package format is untouched.

Two contracts are supported:

- *Standard* (current builds): `adapter.run_sessions(agent_fn, seed)` where
  `agent_fn(input: dict) -> dict`, and `evaluation.evaluate(sessions)`. No
  per-package wiring needed.
- *Legacy* (older builds): each package defined its own agent interface, so a
  small driver in `DRIVERS` adapts the model to that interface. This is wiring
  only, not scoring.
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


def _llm_call(model: str, api_key: str, messages: List[Dict[str, str]], temperature: float, simulate: bool, mock: str = "mock") -> str:
    if simulate:
        return mock
    from openai import OpenAI
    client = OpenAI(base_url=settings.OPENROUTER_API_BASE, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=800,
    )
    return (response.choices[0].message.content or "").strip()


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


# ---------------------------------------------------------------------------
# Standard contract
# ---------------------------------------------------------------------------

def _standard_agent(model: str, api_key: str, temperature: float, simulate: bool) -> Callable[[dict], dict]:
    def agent(input_state: dict) -> dict:
        prompt = json.dumps(input_state, indent=2, ensure_ascii=False) + "\n\nRespond with a single JSON object containing your action."
        reply = _llm_call(model, api_key, [{"role": "user", "content": prompt}], temperature, simulate, mock="{}")
        return _parse_json(reply)
    return agent


# ---------------------------------------------------------------------------
# Legacy drivers (one per older package contract)
# ---------------------------------------------------------------------------

def _cascades_driver(adapter: Any, evaluation: Any, model: str, api_key: str, temperature: float, seed: int, participants: int, simulate: bool):
    def agent(round_input: dict) -> dict:
        instructions = round_input.get("instructions_text", "")
        private = round_input.get("private_draw", "")
        history = json.dumps(round_input.get("decisions_so_far_this_period", []))
        prompt = (
            f"{instructions}\n\n"
            f"This period, your private draw is: {private}.\n"
            f"Decisions made so far this period: {history}\n\n"
            "Which urn do you now believe is more likely? Answer with exactly one letter: A or B."
        )
        reply = _llm_call(model, api_key, [{"role": "user", "content": prompt}], temperature, simulate, mock="A")
        text = reply.strip().upper()
        decision = "A" if ("A" in text and "B" not in text) or text.startswith("A") else "B"
        return {"decision": decision}

    sessions = []
    for study, condition in adapter.CONDITIONS.keys():
        sessions.append(adapter.run_session(agent, study, condition, n_periods=participants, seed=seed, session_id=f"{study}:{condition}"))
    return sessions, evaluation.evaluate(sessions)


def _gsp_driver(adapter: Any, evaluation: Any, model: str, api_key: str, temperature: float, seed: int, participants: int, simulate: bool):
    def agent(state: dict, rng: Any) -> float:
        prompt = json.dumps(state, indent=2) + "\n\nOutput only your bid as a single number."
        reply = _llm_call(model, api_key, [{"role": "user", "content": prompt}], temperature, simulate, mock="5.0")
        match = re.search(r"-?\d+(?:\.\d+)?", reply or "")
        return float(match.group(0)) if match else 0.0

    conditions = []
    for cost in (0.0, 0.1):
        for alpha in (0.2, 0.5, 0.8):
            conditions.append(adapter.Condition(study="human_main", cost=cost, alpha=alpha))
    for cost in (0.0, 0.1):
        for info_cost in (0.0, 0.5):
            conditions.append(adapter.Condition(study="human_mechanism", cost=cost, alpha=0.5, info_cost=info_cost))

    co_player = adapter.policy_softmax_explorer
    runs = [adapter.run_session(condition, participants, agent, co_player, seed) for condition in conditions]
    return runs, evaluation.evaluate(runs)


def _false_consensus_driver(adapter: Any, evaluation: Any, model: str, api_key: str, temperature: float, seed: int, participants: int, simulate: bool):
    materials = adapter.load_json(adapter.MATERIALS_PATH)
    cells = (
        [("study1", c) for c in ("supermarket", "term_paper", "traffic_ticket", "space_program")]
        + [("study2", c) for c in ("self_first", "peer_first")]
        + [("study3", c) for c in ("eat_at_joes", "repent")]
        + [("study4", c) for c in ("generic",)]
    )
    records = []
    for study, condition in cells:
        prompt = adapter.build_prompt(study, condition, materials)
        for _ in range(participants):
            if simulate:
                response = adapter.synthetic_response(study, condition, materials)
            else:
                reply = _llm_call(model, api_key, [{"role": "user", "content": json.dumps(prompt, indent=2) + "\n\nRespond with a single JSON object."}], temperature, simulate, mock="{}")
                response = _parse_json(reply)
            ok, _errors = adapter.validate_response(study, condition, response, materials)
            if ok:
                records.append({"study_id": study, "condition_id": condition, **response})
    return records, evaluation.run_evaluation(records)


def _comap_driver(adapter: Any, evaluation: Any, model: str, api_key: str, temperature: float, seed: int, participants: int, simulate: bool):
    def agent(role: str, system_prompt: str, messages: list) -> str:
        full = [{"role": "system", "content": system_prompt}] if system_prompt else []
        full.extend(messages)
        return _llm_call(model, api_key, full, temperature, simulate, mock="mock")

    materials = adapter.load_materials()
    transcript = adapter.run_dyad_session(1, "A", 1, "B", 2, agent, materials, baseline_system_prompt=None)
    # The package's evaluator expects a pre-aggregated comap-vs-baseline data file
    # (paired_outcomes / ancova_input / graph_logs / chat_logs), which is not
    # produced by a single dyad session. Save the transcript and report the gap.
    result = {"status": "not_ready", "missing_requirement": "paired comap-vs-baseline aggregation is not wired; transcript saved."}
    return [transcript], result


DRIVERS: List[Dict[str, Any]] = [
    {
        "name": "anderson_holt_1997_information_cascades",
        "detect": lambda a: hasattr(a, "run_session") and hasattr(a, "CONDITIONS") and hasattr(a, "_baseline_follow_private_signal"),
        "run": _cascades_driver,
    },
    {
        "name": "kannan_pamuru_rosokha_2023_frictions_gsp",
        "detect": lambda a: hasattr(a, "run_session") and hasattr(a, "Condition") and hasattr(a, "policy_softmax_explorer"),
        "run": _gsp_driver,
    },
    {
        "name": "ross_greene_house_1977_false_consensus",
        "detect": lambda a: hasattr(a, "build_prompt") and hasattr(a, "validate_response") and hasattr(a, "synthetic_response"),
        "run": _false_consensus_driver,
    },
    {
        "name": "comap_shared_visual_workspace_pbl",
        "detect": lambda a: hasattr(a, "run_dyad_session") and hasattr(a, "load_materials"),
        "run": _comap_driver,
    },
]


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

    if hasattr(adapter, "run_sessions") and hasattr(evaluation, "evaluate"):
        agent = _standard_agent(model, api_key, temperature, args.simulate)
        sessions = adapter.run_sessions(agent, seed)
        result = evaluation.evaluate(sessions)
    else:
        driver = next((d for d in DRIVERS if d["detect"](adapter)), None)
        if driver is None:
            raise SystemExit(
                f"Package {package.name} does not match a known run contract. "
                "It needs a driver entry in playground/run_package.py."
            )
        sessions, result = driver["run"](adapter, evaluation, model, api_key, temperature, seed, participants, args.simulate)

    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    (output_dir / "sessions.json").write_text(json.dumps(sessions, indent=2, default=str) + "\n")

    run.update({
        "status": "complete",
        "message": "The run finished and the results are ready",
        "resultsReady": True,
    })
    (run_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")

    print(json.dumps({"status": "complete", "evaluation": result}, default=str))


if __name__ == "__main__":
    main()
