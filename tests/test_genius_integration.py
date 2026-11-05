"""Offline acceptance of GENIUS discovery and the hosted persistent protocol bridge."""
import copy
import json
from pathlib import Path

import pytest

from playground.persistent_study import run_persistent_protocol, persistent_stats
from playground.run_playground import build_profiles, cap_trials
from playground.study_loader import load_study
from scripts.build_studies_index import build_index

ROOT = Path(__file__).resolve().parents[1]


def loaded():
    return load_study(ROOT, "study_016")


def test_genius_is_discoverable_with_original_contributor_credit():
    entry = next(e for e in build_index() if e["study_id"] == "study_016")
    assert "GENIUS" in entry["title"]
    assert entry["contributors"][0]["github"] == "https://github.com/thorbenjansen"


def test_shared_key_limits_model_calls_and_preserves_both_conditions():
    _, spec, config, _ = loaded()
    from src.core.study_config import BaseStudyConfig
    assert isinstance(config, BaseStudyConfig)
    trials = cap_trials(config, {"participantsPerScenario": 801}, False, lambda _: None)
    assert len(trials) == 2
    assert {t["condition"] for t in trials} == {"control", "feedback"}
    assert len(trials) * config.MAX_MODEL_CALLS_PER_PARTICIPANT <= 60
    profiles = build_profiles(trials, spec, {}, "study_016")
    assert all(p["population"] == spec["population"] for p in profiles)
    assert all(p["education"] != "college student" for p in profiles)


def test_offline_protocol_cannot_become_scientific_evidence():
    _, spec, config, evaluator = loaded()
    trials = config.create_trials(2)
    profiles = build_profiles(trials, spec, {}, "study_016")
    saved, stages = [], []
    result = run_persistent_protocol(config, trials, profiles, {}, None, True, "v1_empty", None,
                                     saved.append, lambda i, s: stages.append((i, s)))
    assert len(saved) == 2 and len(stages) == 46
    assert all(len(r["history"]) == 23 for r in saved)
    assert result["execution_kind"] == "synthetic_developer_fixture"
    assert evaluator.evaluate_study(result)["status"] == "not_scored"
    assert persistent_stats(result)["completedTrials"] == 2


def test_hosted_live_contract_preserves_context_and_uses_openrouter(monkeypatch):
    _, spec, config, evaluator = loaded()
    from runtime import ChatBackend
    from run_study import fixture
    calls = []

    def fake_call(self, messages, json_output=True):
        calls.append((id(self), self.model, self.base, copy.deepcopy(messages)))
        self.last_metadata = {"usage": {"total_tokens": 3}, "requested_model": self.model}
        if not json_output:
            return "SYNTHETIC FEEDBACK"
        context = json.loads(messages[-1]["content"][0]["text"].split("\nAntworte ausschließlich", 1)[0])
        return json.dumps(fixture(context))

    monkeypatch.setattr(ChatBackend, "__call__", fake_call)
    trials = config.create_trials(2)
    profiles = build_profiles(trials, spec, {}, "study_016")
    result = run_persistent_protocol(config, trials, profiles,
                                    {"model": "fixture/participant", "feedbackModel": "fixture/feedback", "temperature": 0},
                                    "fake-test-key", False, "v1_empty", None, lambda r: None, lambda i, s: None)
    participants = [c for c in calls if c[1] == "fixture/participant"]
    feedback = [c for c in calls if c[1] == "fixture/feedback"]
    assert len(participants) == 46 and len(feedback) == 5
    assert all(c[2] == "https://openrouter.ai/api/v1" for c in calls)
    assert [len(c[3]) for c in participants] == list(range(2, 48, 2)) * 2
    assert all(len(c[3]) == 2 for c in feedback)
    # Each participant gets a fresh backend, and no evaluator outcomes enter prompts.
    assert participants[0][0] != participants[23][0]
    assert "reference_items" not in json.dumps([c[3] for c in calls])
    assert evaluator.evaluate_study(result)["status"] == "diagnostic_only"
    assert persistent_stats(result)["totalTokens"] == 51 * 3


def test_unsupported_persistent_protocol_fails_before_any_calls():
    class Unsupported:
        PLAYGROUND_PROTOCOL_VERSION = 99
    with pytest.raises(ValueError, match="no supported"):
        run_persistent_protocol(Unsupported(), [], [], {}, None, True, "v1_empty", None,
                                lambda r: None, lambda i, s: None)


def test_budget_checks_actual_returned_sessions():
    class ExpandingConfig:
        MAX_MODEL_CALLS_PER_PARTICIPANT = 28
        def create_trials(self, n_trials=None):
            return [{"condition": "control"}] * 3
    from playground.study_loader import StudyNotRunnable
    with pytest.raises(StudyNotRunnable, match="model-call budget"):
        cap_trials(ExpandingConfig(), {}, False, lambda _: None)


@pytest.mark.parametrize("error", [FileNotFoundError("Rscript"), __import__('subprocess').TimeoutExpired("Rscript", 1)])
def test_unavailable_original_statistics_are_not_a_fabricated_score(monkeypatch, error):
    path, _, _, evaluator = loaded()
    plan = json.loads((path / "source/materials/assignment_blueprint.json").read_text())
    scorer = json.loads((path / "source/scorer/validation.json").read_text())
    import hashlib
    digest = hashlib.sha256((path / "source/materials/feedback_reconstruction.json").read_bytes()).hexdigest()
    records = [{**slot, "reconstruction_mode": "full_surrogate",
                "baseline_score": {"model_sha256": scorer["packaged_model_sha256"]},
                "cycles": [{"feedback": {"reconstruction_sha256": digest}}]} for slot in plan]
    # Isolate external execution availability after allocation/provenance checks.
    # These are contract fixtures, never evidence of a completed model experiment.
    monkeypatch.setattr(evaluator, "records_to_rows", lambda records: [{"fixture_column": 1}])
    monkeypatch.setattr(evaluator, "estimability_issues", lambda rows: [])
    def unavailable(*args, **kwargs):
        raise error
    monkeypatch.setattr(evaluator.subprocess, "run", unavailable)
    outcome = evaluator.evaluate_study({"individual_data": records})
    assert outcome["status"] == "not_scored" and outcome["test_results"] == []
    assert type(error).__name__ in outcome["reason"]
