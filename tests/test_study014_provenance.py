"""study_014 provenance labels must be true (agent_pipeline contract).

`missing` marks a gap that was not filled. A provenance record that keeps a
value is not a gap, so it must carry another label and say how far the value
is verified through its `status`.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from playground.analysis import build_analysis

GT = Path(__file__).resolve().parents[1] / "studies/study_014/source/ground_truth.json"
OVERALL = "/studies/0/findings/0/original_data_points/data/overall/"
LABELS = {"verbatim", "reported", "derived", "missing"}


def _provenance():
    return json.loads(GT.read_text())["value_provenance"]


def test_missing_label_only_for_records_without_a_value():
    for pointer, record in _provenance().items():
        assert record["evidence_label"] in LABELS, pointer
        if record["evidence_label"] == "missing":
            assert record.get("value") is None, f"{pointer} keeps a value but is labelled missing"


def test_retained_legacy_values_name_their_source():
    # Values kept from the legacy expert annotation but not located in either paper.
    legacy = [
        (pointer, record)
        for pointer, record in _provenance().items()
        if record.get("value") is not None and not record["printed_in"]
        and record["evidence_label"] != "derived"
    ]
    assert len(legacy) == 12
    for pointer, record in legacy:
        assert record["evidence_label"] == "reported", pointer
        assert record["reported_from"] == "hsb_legacy_expert_annotation", pointer
        assert record["status"] in {"unverified_legacy", "differs_from_printed"}, pointer


def test_n_subjects_is_printed_in_feng_figure_1a():
    record = _provenance()[OVERALL + "n_subjects"]
    assert record["value"] == 40
    assert record["evidence_label"] == "reported"
    assert record["printed_in"] == ["feng2025"]
    assert record["location"] == "Figure 1A, p. 7694"
    assert record["status"] == "verified_printed_figure_label"
    assert "120" in record["reason"]  # the 2011 sample, so 40 is not mistaken for it


def test_n_observations_is_derived_from_printed_inputs():
    prov = _provenance()
    record = prov[OVERALL + "n_observations"]
    assert record["evidence_label"] == "derived"
    assert record["value"] == prov[OVERALL + "n_subjects"]["value"] * prov[OVERALL + "n_rounds"]["value"]
    assert record["derivation"]


# --- Replication scoring (owner decision 2026-11-03: one paper) ------------------
# The study follows Davis et al. (2011) only. Human result: Table 2, Model (1)
# (printed p. 182), random-effects regression of reserve price on n - mean(n),
# coefficient 2.98 [SE 0.08], p < 0.01. The agent side is fitted the same way.
# Feng et al. (2025) is a later benchmark kept for reference, not for scoring.

REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "studies/study_014"


def _evaluator():
    spec = importlib.util.spec_from_file_location("study014_scoring_evaluator", STUDY / "scripts/evaluator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sessions(slope: float, agents: int = 8, rounds: int = 24):
    """Agents with their own intercept; reserve price = base + slope * n + noise."""
    bidders = [1, 4, 7, 10]
    data = []
    for a in range(agents):
        responses = []
        for t in range(rounds):
            n = bidders[(t + a) % 4]
            price = 40 + 3 * (a % 5) + slope * n + ((t * 7 + a * 3) % 5) - 2
            responses.append({"response_text": str(int(round(price))), "trial_info": {"num_bidders": n, "round": t + 1}})
        data.append({"responses": responses})
    return {"individual_data": data}


def _evaluate(slope: float, **kw):
    return _evaluator().evaluate_study(_sessions(slope, **kw))


def _finding():
    return json.loads(GT.read_text())["studies"][0]["findings"][0]


def test_scoring_uses_davis_2011_only():
    gt = json.loads(GT.read_text())
    finding = gt["studies"][0]["findings"][0]
    scoring = finding["scoring"]
    tests = finding["statistical_tests"]
    assert [t["source_id"] for t in tests] == ["davis2011"]
    test = tests[scoring["statistical_test_index"]]
    assert scoring["source_id"] == test["source_id"] == gt["source_set"]["scoring_source_id"] == "davis2011"
    # Only printed values: Davis Table 2, n - mean(n) row, Model (1).
    assert test["coefficient"] == 2.98
    assert test["standard_error"] == 0.08
    assert test["p_value_bound"] == 0.01 and test["p_value_status"] == "bound_printed"
    assert "Model (1)" in test["location"] and "printed page 182" in test["location"]
    assert "p_value" not in test  # no exact p-value is printed or derived
    rows = test["model_1_printed_rows"]
    assert rows["constant"]["coefficient"] == 29.45 and rows["period_minus_mean_period"]["coefficient"] == 0.04
    assert rows["cube"]["coefficient"] == 31.78 and rows["r_squared"] == 0.499
    # Feng 2025 is a reference only.
    assert "scoring" not in json.dumps(finding["later_benchmark_see_also"]).replace("used_in_scoring", "")
    assert finding["later_benchmark_see_also"]["used_in_scoring"] is False
    assert "verified_printed_benchmark" not in finding
    assert scoring["agent_model"]["differences_from_davis2011"]


def test_evaluator_reports_davis_coefficient_and_agent_coefficient():
    row = _evaluate(slope=2)["test_results"][0]
    assert row["human_source_id"] == "davis2011"
    assert row["human_coefficient"] == 2.98 and row["human_coefficient_label"] == "printed"
    assert row["human_significant"] is True
    assert row["human_p_value"] is None  # bound only; never a made-up exact value
    assert row["human_p_value_bound"] == 0.01
    assert row["agent_coefficient_label"] == "computed"
    assert row["agent_coefficient"] == pytest.approx(2, abs=0.1)
    assert row["effect_size_metric"] == "re_coefficient_reserve_price_points_per_bidder"
    assert row["human_effect_size"] == 2.98
    assert row["agent_effect_size"] == row["agent_coefficient"]
    assert row["effect_gap"] == pytest.approx(row["agent_coefficient"] - 2.98)
    assert row["n_agents"] == 8 and row["n_agent_observations"] == 8 * 24
    assert row["replication"] is True
    text = json.dumps(row)
    assert "Feng" not in text and "feng2025" not in text and "pearson" not in text.lower()


def test_random_effects_estimator_recovers_slope_with_agent_intercepts():
    module = _evaluator()
    import numpy as np
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 60)
    n = rng.choice([1, 4, 7, 10], size=groups.size).astype(float)
    per = np.tile(np.arange(1, 61), 30).astype(float)
    u = rng.normal(0, 10, 30)[groups]
    y = 25 + 3.0 * (n - n.mean()) + 0.05 * (per - per.mean()) + u + rng.normal(0, 5, groups.size)
    fit = module.random_effects_regression(y, np.column_stack([n - n.mean(), per - per.mean()]), groups)
    assert fit["coef"][1] == pytest.approx(3.0, abs=0.1)
    assert fit["sigma_u"] == pytest.approx(10, rel=0.35)
    assert fit["sigma_e"] == pytest.approx(5, rel=0.1)
    # Too few subjects for the between-subject variance: not estimated.
    assert module.random_effects_regression(y[:120], np.column_stack([n, per])[:120], groups[:120]) is None


def test_too_few_agents_is_not_scored():
    row = _evaluate(slope=2, agents=3)["test_results"][0]
    assert row["agent_coefficient"] is None
    assert row["replication"] is None


def test_wrong_direction_is_scored_as_not_replicated():
    row = _evaluate(slope=-2)["test_results"][0]
    assert row["human_significant"] is True
    assert row["direction_match"] is False
    assert row["replication"] is False


def test_study014_counts_in_playground_replication_rate():
    evaluation = _evaluate(slope=2)
    analysis = build_analysis(evaluation, STUDY, {"id": "r1", "studyId": "study_014"}, {"participants": 8})
    summary = analysis["summary"]
    assert summary["scoredTests"] == 1
    assert summary["replicationRate"] == 1.0
    assert summary["meanHumanEffect"] == 2.98
    row = analysis["tests"][0]
    assert row["agent_effect"] == evaluation["test_results"][0]["agent_coefficient"]
    assert "Davis et al. 2011" in row["reported_statistics"]
    assert "Feng" not in row["reported_statistics"]
