"""Run explicitly opted-in persistent protocols without flattening their sessions."""
from typing import Any

from src.agents.prompt_registry import SystemPromptRegistry
from playground import settings


def run_persistent_protocol(config: Any, trials: list, profiles: list, run: dict,
                            api_key: str | None, simulate: bool, preset: str,
                            override: str | None, on_participant, on_stage) -> dict:
    if getattr(config, "PLAYGROUND_PROTOCOL_VERSION", None) != 1:
        raise ValueError("This persistent study has no supported playground protocol bridge.")
    if len(profiles) != len(trials):
        raise ValueError("Persistent sessions require one participant profile per planned slot.")
    # Import only the formatting helper, not a model client or generic participant pool.
    from playground.run_playground import render_prompt
    model = str(run.get("model") or settings.DEFAULT_MODEL)
    participant_prompts = [render_prompt(override, p) if override else SystemPromptRegistry.get_prompt(preset, p)
                           for p in profiles]
    runtime_settings = {
        "use_real_llm": not simulate, "model": model,
        "feedback_model": str(run.get("feedbackModel") or model),
        "api_key": api_key, "api_base": settings.OPENROUTER_API_BASE,
        "temperature": float(run.get("temperature") if run.get("temperature") is not None else 1.0),
        "participant_prompts": participant_prompts, "on_stage": on_stage,
    }
    result = config.run_playground_protocol(trials, runtime_settings, profiles, on_participant)
    records = result.get("individual_data") if isinstance(result, dict) else None
    if not isinstance(records, list) or len(records) != len(trials):
        raise ValueError("The persistent protocol did not return every planned participant session.")
    expected_kind = "synthetic_developer_fixture" if simulate else "agent_run"
    if any(r.get("execution_kind") != expected_kind for r in records):
        raise ValueError("Persistent protocol returned incorrectly labelled execution evidence.")
    return result


def persistent_stats(result: dict) -> dict:
    records = result["individual_data"]
    tokens = 0
    for record in records:
        calls = list(record.get("participant_calls", []))
        calls.extend(c.get("feedback", {}).get("provider", {}) for c in record.get("cycles", []))
        tokens += sum(int((call.get("usage") or {}).get("total_tokens") or 0) for call in calls)
    return {"participants": len(records), "trials": len(records), "completedTrials": len(records),
            "answeredTrials": len(records), "totalTokens": tokens}


def persistent_transcript(result: dict, study_path) -> list:
    import json
    materials = json.loads((study_path / "source/materials/protocol.json").read_text())
    instructions = {c["part"]: c["revision_instruction"] for c in materials["cycles"]}
    return [{"participantId": r["participant_id"], "profile": r["profile"],
             "prompt": instructions[c["part"]], "response": c["revision"],
             "stage": "revision_" + c["part"]}
            for r in result["individual_data"][:2] for c in r["cycles"][:3]]
