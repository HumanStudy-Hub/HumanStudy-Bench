import json
from pathlib import Path
import tempfile
import unittest

from generation_pipeline.stage4 import _deterministic_materials, normalize_to_human_extraction


class Stage4CanonicalIdTests(unittest.TestCase):
    def test_stage4_preserves_stage2_canonical_id_across_package_contracts(self):
        payload = {
            "paper_title": "Decision Framing",
            "eligible_studies": [
                {
                    "study_id": "study_problem_1",
                    "experiment_id": "Problem 1",
                    "study": "Asian Disease Framing Task",
                    "sample": {"total_n": 100},
                    "effects": [{"IV": "frame", "DV": "choice", "effecttype": "main"}],
                }
            ],
            "study_materials": {
                "study_asian_disease_framing_task": {
                    "sub_study_id": "study_asian_disease_framing_task",
                    "instructions": "Choose one program.",
                    "items": [{"item_id": "choice", "question": "Which program?", "options": ["A", "B"]}],
                    "readiness": {"ready": True, "blocking_issues": [], "warnings": []},
                }
            },
            "simulation_targets": [
                {
                    "target_id": "study_asian_disease_framing_task__effect_01",
                    "sub_study_id": "study_asian_disease_framing_task",
                    "study_name": "Asian Disease Framing Task",
                    "effect_index": 0,
                }
            ],
        }

        extraction = normalize_to_human_extraction(payload)

        self.assertEqual(list(extraction["study_materials"]), ["study_problem_1"])
        self.assertEqual(extraction["study_materials"]["study_problem_1"]["sub_study_id"], "study_problem_1")
        self.assertEqual(extraction["studies"][0]["study_id"], "study_problem_1")
        self.assertEqual(extraction["studies"][0]["sub_studies"][0]["sub_study_id"], "study_problem_1")
        self.assertEqual(extraction["simulation_targets"][0]["sub_study_id"], "study_problem_1")
        self.assertEqual(extraction["simulation_targets"][0]["target_id"], "study_problem_1__effect_01")

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _deterministic_materials(extraction, Path(temp_dir))
            self.assertEqual([path.name for path in paths], ["study_problem_1.json"])
            material = json.loads(paths[0].read_text(encoding="utf-8"))
            self.assertEqual(material["sub_study_id"], "study_problem_1")
            self.assertEqual(material["metadata"]["target_ids"], ["study_problem_1__effect_01"])


if __name__ == "__main__":
    unittest.main()
