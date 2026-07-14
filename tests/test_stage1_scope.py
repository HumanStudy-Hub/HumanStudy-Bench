import unittest

from generation_pipeline.extractors.study_data_extractor import (
    StudyDataExtractor,
    _retain_stage1_candidates,
)
from generation_pipeline.filters.replicability_filter import (
    ReplicabilityFilter,
    _normalize_experiments,
)
from generation_pipeline.stage1_study_contract import audit_stage1_study_contract
from generation_pipeline.stage1_verifier import build_verifier_prompt


def _experiment(study_id: str, replicable: str, reasons=None):
    return {
        "experiment_id": study_id.replace("_", " ").title(),
        "study_id": study_id,
        "experiment_name": f"Decision task for {study_id}",
        "study_name": f"Decision task for {study_id}",
        "design_type": "between-subjects",
        "conditions_or_factors": ["frame: gain vs loss"],
        "input": "Participants read a framed choice problem.",
        "participant_task": "Choose one of two options.",
        "participants": "N = 100 adults",
        "output": "Choice proportion",
        "candidate_source_hints": [
            {
                "kind": "paper",
                "description": "Method section",
                "expected_fields": ["instructions", "items", "options"],
            }
        ],
        "replicable": replicable,
        "has_self_contained_materials": True,
        "exclusion_reasons": list(reasons or []),
        "missing_materials": "",
    }


class Stage1ScopeTests(unittest.TestCase):
    def test_stage1_prompt_is_topic_independent(self):
        prompt = ReplicabilityFilter(None)._build_prompt("paper.pdf", 12)
        lowered = prompt.lower()
        self.assertIn("topic is unrestricted", lowered)
        self.assertIn("behavioral economics", lowered)
        self.assertNotIn("moral / ethical", lowered)
        self.assertNotIn("outcome not moral", lowered)

    def test_verifier_uses_same_general_scope(self):
        prompt = build_verifier_prompt({"experiments": []}, "paper text")
        self.assertIn("topic-independent", prompt)
        self.assertIn("scientific topic or discipline", prompt)

    def test_contract_rejects_candidate_with_exclusion_reason(self):
        payload = {
            "experiments": [
                _experiment("decision_task", "YES", ["outcome outside old corpus scope"])
            ]
        }
        contract = audit_stage1_study_contract(payload)
        self.assertFalse(contract["studies"]["decision_task"]["ready"])
        self.assertIn(
            "candidate_has_exclusion_reasons",
            contract["studies"]["decision_task"]["blocking_issues"],
        )

    def test_contract_requires_reason_for_no_label(self):
        payload = {"experiments": [_experiment("field_outcome", "NO")]}
        contract = audit_stage1_study_contract(payload)
        self.assertIn(
            "excluded_without_reason",
            contract["studies"]["field_outcome"]["blocking_issues"],
        )

    def test_normalizer_enforces_label_and_overall_consistency(self):
        payload = {
            "experiments": [
                _experiment("eligible_task", "uncertain"),
                {
                    **_experiment("excluded_task", "no"),
                    "exclusion_reasons": "no participant-facing task",
                },
            ],
            "overall_replicable": False,
        }
        _normalize_experiments(payload)
        self.assertEqual(payload["experiments"][0]["replicable"], "UNCERTAIN")
        self.assertEqual(
            payload["experiments"][1]["exclusion_reasons"],
            ["no participant-facing task"],
        )
        self.assertTrue(payload["overall_replicable"])

    def test_stage2_prompt_and_output_keep_only_stage1_candidates(self):
        stage1 = {
            "experiments": [
                _experiment("eligible_task", "YES"),
                _experiment("excluded_task", "NO", ["no participant-facing task"]),
            ]
        }
        prompt = StudyDataExtractor(None)._build_prompt(stage1, "paper.pdf", 10)
        self.assertIn('"study_id": "eligible_task"', prompt)
        self.assertNotIn('"study_id": "excluded_task"', prompt)
        self.assertNotIn("moral / ethical", prompt.lower())

        result = {
            "eligible_studies": [
                {"study": "Eligible Task", "effects": [{"IV": "x", "DV": "y"}]},
                {"study": "Excluded Task", "effects": [{"IV": "a", "DV": "b"}]},
            ]
        }
        _retain_stage1_candidates(result, stage1)
        self.assertEqual(len(result["eligible_studies"]), 1)
        self.assertEqual(result["eligible_studies"][0]["study_id"], "eligible_task")


if __name__ == "__main__":
    unittest.main()
