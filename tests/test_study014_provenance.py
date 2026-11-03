"""study_014 provenance labels must be true (agent_pipeline contract).

`missing` marks a gap that was not filled. A provenance record that keeps a
value is not a gap, so it must carry another label and say how far the value
is verified through its `status`.
"""
import json
from pathlib import Path

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
