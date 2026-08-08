#!/usr/bin/env python3
"""Replay one benchmark study with a chosen model and score it against the paper.

This is the engine behind the HumanStudy-Hub playground. It reuses the study's
own trial builder and evaluator, so a playground run is scored exactly like a
benchmark run — the only things the researcher changes are the model, the
participant prompt, and how many participants take part.

    python playground/run_playground.py --run <run-dir>

The run directory holds `run.json` (written by the web app) and receives
`progress.json`, `output/`, and `logs/`.
"""

import argparse
import json
import os
import random
import re
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from playground import settings
from playground.analysis import build_analysis
from playground.personas import PersonaError, sample_profiles
from playground.progress import ProgressWriter
from playground.run_key import decrypt_api_key
from playground.study_loader import StudyNotRunnable, load_metadata, load_study


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_run(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "run.json"
    if not path.exists():
        raise SystemExit(f"No run.json in {run_dir}")
    return json.loads(path.read_text())


def write_run(run_dir: Path, run: Dict[str, Any]) -> None:
    run["updatedAt"] = now()
    (run_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")


def resolve_prompt(run: Dict[str, Any]) -> tuple[str, Optional[str]]:
    """Return (preset name passed to the registry, explicit prompt override)."""
    preset = str(run.get("preset") or settings.DEFAULT_PRESET)
    custom = (run.get("systemPrompt") or "").strip()
    if preset == settings.CUSTOM_PRESET:
        if not custom:
            raise SystemExit("This run selected a custom prompt but did not include one.")
        return settings.DEFAULT_PRESET, custom
    if preset not in settings.PROMPT_PRESETS:
        raise SystemExit(f"Unknown prompt preset: {preset}")
    return preset, custom or None


def render_prompt(template: str, profile: Dict[str, Any]) -> str:
    """Fill {{field}} placeholders in a hand-written prompt from one agent's profile.

    Without this a custom prompt is one fixed string for the whole run, so every
    agent would be the same person no matter which personas were designed.
    """
    def replace(match: "re.Match[str]") -> str:
        value = profile.get(match.group(1).strip())
        return "" if value is None else str(value)

    filled = re.sub(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}", replace, template)
    # Collapse the gaps left where a profile had nothing for a placeholder.
    return re.sub(r"[ \t]{2,}", " ", filled).strip()


def apply_demographics(profile: Dict[str, Any], demographics: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay the researcher's demographic choices on a generated profile."""
    merged = dict(profile)
    for field in settings.DEMOGRAPHIC_FIELDS:
        value = demographics.get(field)
        if value not in (None, "", []):
            merged[field] = value
    return merged


def cap_trials(study_config: Any, run: Dict[str, Any], has_own_key: bool, log) -> List[Dict[str, Any]]:
    """Build trials at the requested size, then hold the run inside its budget."""
    max_total, max_per_scenario = settings.trial_limits(has_own_key)
    requested = int(run.get("participantsPerScenario") or settings.DEFAULT_PER_SCENARIO)
    per_scenario = max(1, min(requested, max_per_scenario))
    if per_scenario != requested:
        log(f"Requested {requested} participants per condition; capped to {per_scenario} for this run budget.")

    try:
        trials = study_config.create_trials(n_trials=per_scenario)
    except TypeError:
        # A few studies take extra keyword arguments and ignore n_trials.
        trials = study_config.create_trials()
    trials = [trial for trial in trials if isinstance(trial, dict)]
    if not trials:
        raise SystemExit("The study produced no trials to run.")

    if len(trials) > max_total:
        # Subsample evenly across conditions so every condition keeps participants.
        rng = random.Random(int(run.get("seed") or 42))
        by_scenario: Dict[str, List[Dict[str, Any]]] = {}
        for trial in trials:
            key = str(trial.get("scenario_id") or trial.get("scenario") or trial.get("sub_study_id") or "all")
            by_scenario.setdefault(key, []).append(trial)
        quota = max(1, max_total // len(by_scenario))
        kept: List[Dict[str, Any]] = []
        for group in by_scenario.values():
            rng.shuffle(group)
            kept.extend(group[:quota])
        trials = kept[:max_total]
        log(f"Run budget allows {max_total} participant sessions; sampled {len(trials)} across {len(by_scenario)} conditions.")
    return trials


def build_profiles(trials: List[Dict[str, Any]], specification: Dict[str, Any], run: Dict[str, Any], study_id: str) -> List[Dict[str, Any]]:
    """Decide who each participant is.

    A persona group defines the population directly and replaces the study's own
    sampling. Otherwise every participant keeps the profile the study drew for
    them, with any demographic the researcher pinned applied on top.
    """
    population = specification.get("participants", {}).get("population")
    recruitment = specification.get("participants", {}).get("recruitment_source")
    defaults = {"study_id": study_id, "population": population, "education": recruitment or "college student"}

    group = run.get("personaGroup")
    if group:
        return sample_profiles(group, len(trials), int(run.get("seed") or 42), defaults)

    demographics = run.get("demographics") or {}
    profiles = []
    for index, trial in enumerate(trials):
        base = {**defaults, **(trial.get("profile") or {})}
        base["participant_id"] = index
        base["study_id"] = study_id
        profiles.append(apply_demographics(base, demographics))
    return profiles


def evaluation_input(pool_results: Dict[str, Any], study_path: Optional[Path] = None) -> Dict[str, Any]:
    """Study evaluators read participants that each carry their own responses.

    The raw API envelope for every call is dropped here: it is many times larger
    than the answers themselves and no evaluator reads it.
    """
    participants = []
    for summary in pool_results.get("participant_summaries", []):
        participants.append({
            "participant_id": summary.get("participant_id"),
            "profile": summary.get("profile"),
            "responses": [
                {
                    "participant_id": response.get("participant_id"),
                    "trial_number": response.get("trial_number"),
                    "response": response.get("response"),
                    "response_text": response.get("response_text"),
                    "is_correct": response.get("is_correct"),
                    "correct_answer": response.get("correct_answer"),
                    "trial_info": response.get("trial_info"),
                }
                for response in summary.get("responses", []) or []
            ],
        })
    payload: Dict[str, Any] = {"individual_data": participants}
    # Some evaluators resolve the study's own ground truth from this field.
    if study_path is not None:
        payload["study_path"] = str(study_path)
    return payload


def evaluate_in_study_dir(evaluator: Any, scored_input: Dict[str, Any], study_path: Path) -> Dict[str, Any]:
    """Score the run from inside the study folder.

    A handful of evaluators open their ground truth through a relative path, so
    they only resolve correctly when the study directory is the working
    directory.
    """
    previous = os.getcwd()
    os.chdir(study_path)
    try:
        return evaluator.evaluate_study(scored_input)
    finally:
        os.chdir(previous)


def response_stats(pool_results: Dict[str, Any]) -> Dict[str, Any]:
    summaries = pool_results.get("participant_summaries", [])
    completed = 0
    answered = 0
    tokens = 0
    for summary in summaries:
        for response in summary.get("responses", []) or []:
            completed += 1
            if (response.get("response_text") or "").strip():
                answered += 1
            usage = response.get("usage") or {}
            tokens += int(usage.get("total_tokens") or 0)
    return {
        "participants": len(summaries),
        "trials": completed,
        "completedTrials": completed,
        "answeredTrials": answered,
        "totalTokens": tokens,
    }


def sample_transcript(pool_results: Dict[str, Any], prompt_builder: Any, limit: int = 6) -> List[Dict[str, Any]]:
    """A few complete participant exchanges, so researchers can read what happened."""
    samples: List[Dict[str, Any]] = []
    for summary in pool_results.get("participant_summaries", []):
        profile = summary.get("profile") or {}
        for response in summary.get("responses", []) or []:
            trial_info = response.get("trial_info") or {}
            try:
                prompt = prompt_builder.build_trial_prompt({**trial_info, "participant_profile": profile})
            except Exception:
                prompt = None
            samples.append({
                "participantId": summary.get("participant_id"),
                "profile": {key: profile.get(key) for key in ("age", "gender", "education", "background", "persona") if profile.get(key)},
                "prompt": prompt,
                "response": response.get("response_text"),
            })
            if len(samples) >= limit:
                return samples
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path, help="Run directory containing run.json")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--progress-repo")
    parser.add_argument("--progress-branch")
    parser.add_argument("--progress-path")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Exercise the run end to end with simulated participants and no API calls.",
    )
    args = parser.parse_args()

    run_dir = args.run.resolve()
    output_dir = run_dir / "output"
    logs_dir = run_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = (logs_dir / "run.log").open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"[{now()}] {message}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    run = read_run(run_dir)
    progress = ProgressWriter(run_dir, args.progress_repo, args.progress_branch, args.progress_path)
    progress.write({"phase": "preparing", "completedTrials": 0, "totalTrials": 0, "message": "Loading the study"}, force=True)

    try:
        api_key = decrypt_api_key(run.get("sealedApiKey"))
        has_own_key = api_key is not None
        if not has_own_key:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key and not args.simulate:
            raise SystemExit("No OpenRouter API key is available for this run.")

        study_path, specification, study_config, evaluator = load_study(args.repo_root, str(run.get("studyId") or ""))
        metadata = load_metadata(study_path)
        run["studyTitle"] = metadata.get("title") or run.get("studyTitle")
        log(f"Study {study_path.name}: {run['studyTitle']}")

        preset, override = resolve_prompt(run)
        trials = cap_trials(study_config, run, has_own_key, log)
        profiles = build_profiles(trials, specification, run, study_path.name)
        log(f"Prepared {len(trials)} participant sessions on {run.get('model')} with prompt {run.get('preset')}.")
        if run.get("personaGroup"):
            mix = Counter(profile.get("persona_label") for profile in profiles)
            log("Participants: " + ", ".join(f"{count} × {label}" for label, count in mix.items()))
        # The resolved cast is written out so a researcher can see exactly who took
        # part, and reproduce or edit it for the next run.
        (output_dir / "profiles.json").write_text(json.dumps(profiles, indent=2, default=str) + "\n")

        from src.agents.llm_participant_agent import ParticipantPool

        prompt_builder = study_config.get_prompt_builder()
        pool = ParticipantPool(
            study_specification=specification,
            n_participants=len(trials),
            use_real_llm=not args.simulate,
            model=str(run.get("model") or settings.DEFAULT_MODEL),
            api_key=api_key,
            # Passing the OpenRouter base explicitly is what routes every model —
            # including Claude and Gemini names — through OpenRouter.
            api_base=settings.OPENROUTER_API_BASE,
            random_seed=int(run.get("seed") or 42),
            num_workers=int(run.get("workers") or settings.DEFAULT_WORKERS),
            profiles=profiles,
            prompt_builder=prompt_builder,
            system_prompt_override=override,
            system_prompt_preset=preset,
            study_id=study_path.name,
            temperature=float(run.get("temperature") or 1.0),
        )

        # A hand-written prompt with placeholders is rendered per agent, so a
        # custom prompt and a persona group can be used together.
        if override and "{{" in override:
            for agent, profile in zip(pool.participants, profiles):
                agent.system_prompt_override = render_prompt(override, profile)
            log("Custom prompt contains placeholders; each agent received its own version.")

        state = {"done": 0}

        def on_trial(*_: Any) -> None:
            state["done"] += 1
            progress.write({
                "phase": "running_participants",
                "completedTrials": state["done"],
                "totalTrials": len(trials),
                "message": f"{state['done']} of {len(trials)} participant sessions complete",
            })

        progress.write({"phase": "running_participants", "completedTrials": 0, "totalTrials": len(trials), "message": "Participants are starting"}, force=True)
        pool.run_experiment(
            trials=trials,
            instructions=study_config.get_instructions(),
            prompt_builder=prompt_builder,
            one_to_one=True,
            save_callback=on_trial,
        )

        pool_results = pool.aggregate_results()
        stats = response_stats(pool_results)
        scored_input = evaluation_input(pool_results, study_path)
        log(f"Collected {stats['answeredTrials']} answers from {stats['participants']} participants.")
        (output_dir / "responses.json").write_text(json.dumps(scored_input, indent=2, default=str) + "\n")
        (output_dir / "transcript_sample.json").write_text(json.dumps(sample_transcript(pool_results, prompt_builder), indent=2, default=str) + "\n")

        progress.write({"phase": "scoring", "completedTrials": stats["completedTrials"], "totalTrials": len(trials), "message": "Scoring against the published findings"}, force=True)
        evaluation = evaluate_in_study_dir(evaluator, scored_input, study_path)
        (output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2, default=str) + "\n")

        analysis = build_analysis(evaluation, study_path, run, stats)
        (output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, default=str) + "\n")
        summary = analysis["summary"]
        log(
            "Replication: "
            f"{summary['replicatedTests']}/{summary['scoredTests']} tests reproduced the published direction and significance."
        )

        run.update({
            "status": "analysing",
            "message": "Building the comparison charts",
            "participants": stats["participants"],
            "completedTrials": stats["completedTrials"],
            "answeredTrials": stats["answeredTrials"],
            "totalTokens": stats["totalTokens"],
            "summary": summary,
        })
        write_run(run_dir, run)
        progress.write({"phase": "charting", "completedTrials": stats["completedTrials"], "totalTrials": len(trials), "message": "Building the comparison charts"}, force=True)

    except PersonaError as error:
        log(f"Persona group rejected: {error}")
        run.update({"status": "failed", "message": "The persona group for this run could not be used", "error": str(error)})
        write_run(run_dir, run)
        progress.write({"phase": "failed", "message": str(error)}, force=True)
        raise SystemExit(1)
    except StudyNotRunnable as error:
        log(f"Study cannot be replayed: {error}")
        run.update({"status": "failed", "message": "This study cannot be replayed in the playground", "error": str(error)})
        write_run(run_dir, run)
        progress.write({"phase": "failed", "message": str(error)}, force=True)
        raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as error:
        log(f"Run failed: {error}\n{traceback.format_exc()}")
        run.update({"status": "failed", "message": "The playground run failed", "error": str(error)})
        write_run(run_dir, run)
        progress.write({"phase": "failed", "message": str(error)}, force=True)
        raise SystemExit(1)
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
