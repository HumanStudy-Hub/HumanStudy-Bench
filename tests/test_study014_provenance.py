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


# --- Replication scoring (owner decision, split sources) ------------------------
# Human significance and direction: Davis et al. (2011) Table 2 regression,
# coefficient 2.98 [SE 0.08], p < .01 (printed p. 182). Effect size: Feng et al.
# (2025) Table 2, Pearson r = .42 with no printed p-value.

REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "studies/study_014"


def _evaluate(slope: int):
    spec = importlib.util.spec_from_file_location("study014_scoring_evaluator", STUDY / "scripts/evaluator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bidders = [1, 4, 7, 10]
    responses = []
    for i in range(40):
        n = bidders[i % 4]
        price = 50 + slope * n + (i % 3)  # strictly monotone in n, slope sets the sign
        responses.append({"response_text": str(price), "trial_info": {"num_bidders": n}})
    return module.evaluate_study({"individual_data": [{"responses": responses}]})


def _finding():
    return json.loads(GT.read_text())["studies"][0]["findings"][0]


def test_scoring_block_names_one_source_per_statistic():
    finding = _finding()
    scoring = finding["scoring"]
    tests = finding["statistical_tests"]
    sig = tests[scoring["significance_and_direction"]["statistical_test_index"]]
    eff = tests[scoring["effect_size"]["statistical_test_index"]]
    assert sig["source_id"] == scoring["significance_and_direction"]["source_id"] == "davis2011"
    assert eff["source_id"] == scoring["effect_size"]["source_id"] == "feng2025"
    # Only printed values: Davis Table 2, n - mean(n) row, Models (1) and (2).
    assert sig["coefficient"] == 2.98
    assert sig["standard_error"] == 0.08
    assert sig["p_value_bound"] == 0.01 and sig["p_value_status"] == "bound_printed"
    assert "printed page 182" in sig["location"]
    assert "p_value" not in sig  # no exact p-value is printed or derived
    # The later Pearson test still has no p-value.
    assert eff["p_value_status"] == "missing"
    assert finding["verified_printed_benchmark"]["pearson_p_value"] is None


def test_evaluator_scores_significance_from_davis_and_effect_from_feng():
    row = _evaluate(slope=2)["test_results"][0]
    assert row["human_significance_source_id"] == "davis2011"
    assert row["human_coefficient"] == 2.98
    assert row["human_significant"] is True
    assert row["human_p_value"] is None  # bound only; never a made-up exact value
    assert row["human_p_value_bound"] == 0.01
    assert row["human_effect_source_id"] == "feng2025"
    assert row["human_pearson_r"] == 0.42
    assert row["human_effect_p_value_status"] == "missing"
    assert row["human_effect_d"] == pytest.approx(2 * 0.42 / (1 - 0.42 ** 2) ** 0.5)
    # The output says that two tests feed the row.
    assert "Davis" in row["scoring_note"] and "Feng" in row["scoring_note"]
    assert "Davis et al. 2011" in row["reported_statistics"]
    assert "Feng et al. 2025" in row["reported_statistics"]
    assert row["replication"] is True


def test_wrong_direction_is_scored_as_not_replicated():
    row = _evaluate(slope=-2)["test_results"][0]
    assert row["human_significant"] is True
    assert row["direction_match"] is False
    assert row["replication"] is False


def test_study014_counts_in_playground_replication_rate():
    evaluation = _evaluate(slope=2)
    analysis = build_analysis(evaluation, STUDY, {"id": "r1", "studyId": "study_014"}, {"participants": 1})
    summary = analysis["summary"]
    assert summary["scoredTests"] == 1
    assert summary["replicationRate"] == 1.0
    assert summary["meanHumanEffect"] == pytest.approx(2 * 0.42 / (1 - 0.42 ** 2) ** 0.5)
    row = analysis["tests"][0]
    assert "Davis et al. 2011" in row["reported_statistics"]
    assert "Feng et al. 2025" in row["reported_statistics"]
