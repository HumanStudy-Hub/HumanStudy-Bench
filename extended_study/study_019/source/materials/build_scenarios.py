#!/usr/bin/env python3
"""Compile the two public scenario tasks and Figshare raw-data summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import openpyxl


STUDY_1_MD5 = "e6f623799cce4b8331b806b55563b31c"
STUDY_2_MD5 = "c29df321e159c9cd0afbfa3defa73159"

# raw_id, article table scenario, previous decisions, private-favored urn, p(Urn A)
STUDY_1_BASE: Tuple[Tuple[int, int, Tuple[str, ...], str, float], ...] = (
    (1, 1, ("A",), "A", 0.80),
    (2, 2, ("A", "A"), "A", 0.89),
    (3, 3, ("A", "B"), "A", 0.67),
    (4, 4, ("A", "A", "A"), "A", 0.89),
    (5, 5, ("A", "B", "A"), "A", 0.80),
    (6, 11, ("A", "B", "B"), "A", 0.50),
    (7, 10, ("A",), "B", 0.50),
    (8, 6, ("A", "A"), "B", 0.67),
    (9, 7, ("A", "B"), "B", 0.33),
    (10, 8, ("A", "A", "A"), "B", 0.67),
    (11, 12, ("A", "B", "A"), "B", 0.50),
    (12, 9, ("A", "B", "B"), "B", 0.20),
)

# raw_id, article table scenario, [(role, diagnosis)], private disease, p(appendicitis)
STUDY_2_SCENARIOS: Tuple[
    Tuple[int, int, Tuple[Tuple[str, str], ...], str, float], ...
] = (
    (1, 14, (("AS", "AP"),), "AP", 0.80),
    (2, 17, (("MD", "AP"),), "AP", 0.80),
    (3, 26, (("MD", "AP"), ("AS", "AP")), "AP", 0.89),
    (4, 27, (("AS", "AP"), ("MD", "AP")), "AP", 0.89),
    (5, 24, (("AS", "AP"), ("AS", "AP")), "AP", 0.89),
    (6, 5, (("MD", "AP"), ("AS", "SI")), "AP", 0.67),
    (7, 8, (("AS", "AP"), ("MD", "SI")), "AP", 0.67),
    (8, 1, (("AS", "AP"), ("AS", "SI")), "AP", 0.67),
    (9, 28, (("MD", "AP"), ("AS", "AP"), ("AS", "AP")), "AP", 0.89),
    (10, 29, (("AS", "AP"), ("MD", "AP"), ("AS", "AP")), "AP", 0.89),
    (11, 30, (("AS", "AP"), ("AS", "AP"), ("MD", "AP")), "AP", 0.89),
    (12, 25, (("AS", "AP"), ("AS", "AP"), ("AS", "AP")), "AP", 0.89),
    (13, 18, (("MD", "AP"), ("AS", "SI"), ("AS", "AP")), "AP", 0.80),
    (14, 22, (("AS", "AP"), ("MD", "SI"), ("AS", "AP")), "AP", 0.80),
    (15, 19, (("AS", "AP"), ("AS", "SI"), ("MD", "AP")), "AP", 0.80),
    (16, 15, (("AS", "AP"), ("AS", "SI"), ("AS", "AP")), "AP", 0.80),
    (17, 34, (("MD", "AP"), ("AS", "SI"), ("AS", "SI")), "AP", 0.50),
    (18, 37, (("AS", "AP"), ("MD", "SI"), ("AS", "SI")), "AP", 0.50),
    (19, 38, (("AS", "AP"), ("AS", "SI"), ("MD", "SI")), "AP", 0.50),
    (20, 32, (("AS", "AP"), ("AS", "SI"), ("AS", "SI")), "AP", 0.50),
    (21, 31, (("AS", "AP"),), "SI", 0.50),
    (22, 36, (("MD", "AP"),), "SI", 0.50),
    (23, 9, (("MD", "AP"), ("AS", "AP")), "SI", 0.67),
    (24, 10, (("AS", "AP"), ("MD", "AP")), "SI", 0.67),
    (25, 3, (("AS", "AP"), ("AS", "AP")), "SI", 0.67),
    (26, 7, (("MD", "AP"), ("AS", "SI")), "SI", 0.33),
    (27, 6, (("AS", "AP"), ("MD", "SI")), "SI", 0.33),
    (28, 2, (("AS", "AP"), ("AS", "SI")), "SI", 0.33),
    (29, 11, (("MD", "AP"), ("AS", "AP"), ("AS", "AP")), "SI", 0.67),
    (30, 12, (("AS", "AP"), ("MD", "AP"), ("AS", "AP")), "SI", 0.67),
    (31, 13, (("AS", "AP"), ("AS", "AP"), ("MD", "AP")), "SI", 0.67),
    (32, 4, (("AS", "AP"), ("AS", "AP"), ("AS", "AP")), "SI", 0.67),
    (33, 39, (("MD", "AP"), ("AS", "SI"), ("AS", "AP")), "SI", 0.50),
    (34, 35, (("AS", "AP"), ("MD", "SI"), ("AS", "AP")), "SI", 0.50),
    (35, 40, (("AS", "AP"), ("AS", "SI"), ("MD", "AP")), "SI", 0.50),
    (36, 33, (("AS", "AP"), ("AS", "SI"), ("AS", "AP")), "SI", 0.50),
    (37, 23, (("MD", "AP"), ("AS", "SI"), ("AS", "SI")), "SI", 0.20),
    (38, 20, (("AS", "AP"), ("MD", "SI"), ("AS", "SI")), "SI", 0.20),
    (39, 21, (("AS", "AP"), ("AS", "SI"), ("MD", "SI")), "SI", 0.20),
    (40, 16, (("AS", "AP"), ("AS", "SI"), ("AS", "SI")), "SI", 0.20),
)

ROLE_NAMES = {"AS": "assistant physician", "MD": "medical director"}
DISEASE_NAMES = {"AP": "appendicitis", "SI": "sigmoid diverticulitis"}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace(",", "."))


def _workbook_columns(
    path: Path,
    scenario_count: int,
) -> Dict[int, List[Tuple[Optional[int], Optional[int]]]]:
    workbook = openpyxl.load_workbook(path, data_only=True)
    sheet = workbook.active
    rows = list(sheet.iter_rows(values_only=True))[1:]
    output: Dict[int, List[Tuple[Optional[int], Optional[int]]]] = {}
    for scenario_id in range(1, scenario_count + 1):
        values: List[Tuple[Optional[int], Optional[int]]] = []
        for row in rows:
            raw_choice = _number(row[2 * scenario_id - 1])
            raw_confidence = _number(row[2 * scenario_id])
            choice = (
                None
                if raw_choice is None or raw_choice == 9
                else int(raw_choice)
            )
            confidence = (
                None
                if raw_confidence is None or raw_confidence == 9
                else int(raw_confidence)
            )
            if choice is None and confidence is None:
                continue
            values.append((choice, confidence))
        output[scenario_id] = values
    return output


def _flip(value: str) -> str:
    return "B" if value == "A" else "A"


def _bayesian_choice(probability_a: float, a_label: str = "A", b_label: str = "B") -> Optional[str]:
    if probability_a > 0.5:
        return a_label
    if probability_a < 0.5:
        return b_label
    return None


def _mean_or_none(values: Iterable[float]) -> Optional[float]:
    items = list(values)
    return mean(items) if items else None


def _fingerprint(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _human_stats(
    observations: Sequence[Tuple[Optional[str], Optional[int]]],
    *,
    option_1: str,
    private_choice: str,
    bayesian_choice: Optional[str],
    authority_choice: Optional[str] = None,
) -> Dict[str, Any]:
    choices = [choice for choice, _ in observations if choice is not None]
    confidences = [
        confidence
        for _, confidence in observations
        if confidence is not None
    ]
    paired = [
        (choice, confidence)
        for choice, confidence in observations
        if choice is not None and confidence is not None
    ]
    target_probabilities = [
        confidence / 100.0 if choice == bayesian_choice else 1.0 - confidence / 100.0
        for choice, confidence in paired
        if bayesian_choice is not None
    ]
    return {
        "n": len(choices),
        "n_choice": len(choices),
        "n_confidence": len(confidences),
        "n_paired": len(paired),
        "option_1": option_1,
        "option_1_rate": _mean_or_none(choice == option_1 for choice in choices),
        "private_choice_rate": _mean_or_none(
            choice == private_choice for choice in choices
        ),
        "bayesian_choice_rate": (
            _mean_or_none(choice == bayesian_choice for choice in choices)
            if bayesian_choice is not None
            else None
        ),
        "authority_choice_rate": (
            _mean_or_none(choice == authority_choice for choice in choices)
            if authority_choice is not None
            else None
        ),
        "mean_confidence": _mean_or_none(confidences),
        "mean_probability_assigned_to_bayesian_choice": _mean_or_none(
            target_probabilities
        ),
    }


def _compile_study_1(
    raw: Dict[int, List[Tuple[Optional[int], Optional[int]]]],
) -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = []
    for raw_id in range(1, 25):
        base_id = raw_id if raw_id <= 12 else raw_id - 12
        _, article_id, base_previous, base_private, base_probability_a = STUDY_1_BASE[
            base_id - 1
        ]
        mirrored = raw_id > 12
        previous = (
            tuple(_flip(choice) for choice in base_previous)
            if mirrored
            else base_previous
        )
        private_choice = _flip(base_private) if mirrored else base_private
        probability_a = (
            round(1.0 - base_probability_a, 2) if mirrored else base_probability_a
        )
        visible_observations: List[Tuple[Optional[str], Optional[int]]] = []
        for choice_code, confidence in raw[raw_id]:
            visible_choice: Optional[str] = None
            if choice_code is not None:
                coded_choice = "A" if choice_code == 1 else "B"
                visible_choice = _flip(coded_choice) if mirrored else coded_choice
            visible_observations.append((visible_choice, confidence))
        bayesian = _bayesian_choice(probability_a)
        source_signature = {
            "previous_decisions": list(previous),
            "private_ball": "white" if private_choice == "A" else "black",
            "posterior_probability_urn_a": probability_a,
        }
        scenario = {
            "scenario_id": f"study1_raw_{raw_id:02d}",
            "raw_variable_id": raw_id,
            "article_scenario_id": article_id,
            "base_information_structure_id": base_id,
            "mirrored_presentation": mirrored,
            "previous_decisions": list(previous),
            "private_ball": "white" if private_choice == "A" else "black",
            "private_information_favors": private_choice,
            "posterior_probability_urn_a": probability_a,
            "bayesian_choice": bayesian,
            "bayesian_confidence": max(probability_a, 1.0 - probability_a),
            "cascade_scenario": article_id in {6, 8},
            "indifference_scenario": bayesian is None,
            "human_raw_data": _human_stats(
                visible_observations,
                option_1="A",
                private_choice=private_choice,
                bayesian_choice=bayesian,
            ),
            "source_signature": source_signature,
        }
        scenario["material_fingerprint"] = _fingerprint(source_signature)
        scenarios.append(scenario)
    return scenarios


def _authority_condition(
    previous: Sequence[Tuple[str, str]], private_disease: str
) -> str:
    director = next((disease for role, disease in previous if role == "MD"), None)
    if director is None:
        return "baseline"
    if director == private_disease:
        return "medical_director_supports_private"
    return "medical_director_opposes_private"


def _compile_study_2(
    raw: Dict[int, List[Tuple[Optional[int], Optional[int]]]],
) -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = []
    for raw_id, article_id, coded_previous, private_code, probability_ap in STUDY_2_SCENARIOS:
        previous = [
            {
                "position": index,
                "role": ROLE_NAMES[role],
                "diagnosis": DISEASE_NAMES[disease],
            }
            for index, (role, disease) in enumerate(coded_previous, start=1)
        ]
        private_disease = DISEASE_NAMES[private_code]
        observations: List[Tuple[Optional[str], Optional[int]]] = []
        for choice_code, confidence in raw[raw_id]:
            diagnosis = (
                None
                if choice_code is None
                else DISEASE_NAMES["AP"]
                if choice_code == 1
                else DISEASE_NAMES["SI"]
            )
            observations.append((diagnosis, confidence))
        bayesian = _bayesian_choice(
            probability_ap,
            DISEASE_NAMES["AP"],
            DISEASE_NAMES["SI"],
        )
        director_diagnosis = next(
            (
                DISEASE_NAMES[disease]
                for role, disease in coded_previous
                if role == "MD"
            ),
            None,
        )
        condition = _authority_condition(coded_previous, private_code)
        source_signature = {
            "previous_diagnoses": previous,
            "private_symptom": (
                "twinges in the lower left part of the abdomen"
                if private_code == "AP"
                else "regurgitation"
            ),
            "posterior_probability_appendicitis": probability_ap,
        }
        scenario = {
            "scenario_id": f"study2_raw_{raw_id:02d}",
            "raw_variable_id": raw_id,
            "article_scenario_id": article_id,
            "previous_diagnoses": previous,
            "private_symptom": source_signature["private_symptom"],
            "private_information_favors": private_disease,
            "posterior_probability_appendicitis": probability_ap,
            "bayesian_choice": bayesian,
            "bayesian_confidence": max(probability_ap, 1.0 - probability_ap),
            "authority_condition": condition,
            "medical_director_diagnosis": director_diagnosis,
            "cascade_scenario": article_id in {3, 4, 9, 10, 11, 12, 13},
            "indifference_scenario": bayesian is None,
            "human_raw_data": _human_stats(
                observations,
                option_1=DISEASE_NAMES["AP"],
                private_choice=private_disease,
                bayesian_choice=bayesian,
                authority_choice=director_diagnosis,
            ),
            "source_signature": source_signature,
        }
        scenario["material_fingerprint"] = _fingerprint(source_signature)
        scenarios.append(scenario)
    return scenarios


def build(study_1_path: Path, study_2_path: Path) -> Dict[str, Any]:
    if _md5(study_1_path) != STUDY_1_MD5:
        raise ValueError("Study 1 workbook MD5 does not match the Figshare record")
    if _md5(study_2_path) != STUDY_2_MD5:
        raise ValueError("Study 2 workbook MD5 does not match the Figshare record")

    study_1 = _compile_study_1(_workbook_columns(study_1_path, 24))
    study_2 = _compile_study_2(_workbook_columns(study_2_path, 40))
    return {
        "schema_version": 1,
        "source": {
            "figshare_doi": "10.6084/m9.figshare.1597662.v1",
            "study_1_workbook": {
                "file": study_1_path.name,
                "md5": STUDY_1_MD5,
                "sha256": _sha256(study_1_path),
            },
            "study_2_workbook": {
                "file": study_2_path.name,
                "md5": STUDY_2_MD5,
                "sha256": _sha256(study_2_path),
            },
        },
        "study_1": {
            "sub_study_id": "study_1_urn_scenarios",
            "instructions": (
                "Two urns are equally likely. Urn A contains two white balls and one "
                "black ball. Urn B contains one white ball and two black balls. Up to "
                "four people act in sequence. Each person privately observes one ball, "
                "replaces it, sees every urn prediction already announced by earlier "
                "people, and then publicly announces only an urn prediction. Thus, a "
                "later public prediction may already reflect earlier public predictions "
                "and must not be treated as an independent ball draw. You play the final "
                "person shown in each scenario. Use the ordered public predictions and "
                "your own private ball to predict the selected urn. One scenario may be "
                "selected for payment: a correct decision earns CHF 2, and a confidence "
                "judgment within five percentage points of the normative probability "
                "earns an additional CHF 2."
            ),
            "scenarios": study_1,
        },
        "study_2": {
            "sub_study_id": "study_2_medical_authority_scenarios",
            "instructions": (
                "Imagine that you are an assistant physician diagnosing either "
                "appendicitis or sigmoid diverticulitis; the diseases are equally "
                "likely. Regurgitation occurs with probability 0.67 under sigmoid "
                "diverticulitis and 0.33 under appendicitis. Twinges in the lower left "
                "abdomen occur with probability 0.67 under appendicitis and 0.33 under "
                "sigmoid diverticulitis. An assistant physician or medical director is "
                "correct in two of three cases when diagnosing without seeing another "
                "physician's diagnosis. In this "
                "sequential task, each physician sees the diagnoses already entered in "
                "the patient record before adding a diagnosis. Consequently, a later "
                "diagnosis may already reflect earlier diagnoses and is not an independent "
                "medical test. The patient record shows those diagnoses in their original "
                "order, and you make the final diagnosis. One scenario may be selected "
                "for payment: a correct diagnosis earns CHF 2, and a confidence judgment "
                "within five percentage points of the normative probability earns an "
                "additional CHF 2."
            ),
            "scenarios": study_2,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    raw_dir = Path(__file__).parents[1] / "raw_data"
    parser.add_argument(
        "--study-1",
        type=Path,
        default=raw_dir / "Raw Data_Study1.xlsx",
    )
    parser.add_argument(
        "--study-2",
        type=Path,
        default=raw_dir / "RawData_Study2.xlsx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("scenarios.json"),
    )
    args = parser.parse_args()
    payload = build(args.study_1, args.study_2)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(payload['study_1']['scenarios'])} Study 1 and "
        f"{len(payload['study_2']['scenarios'])} Study 2 scenarios to {args.output}"
    )


if __name__ == "__main__":
    main()
