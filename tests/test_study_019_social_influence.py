import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from generation_pipeline.stage5 import Stage5Options, run_stage5
from src.core.study import Study
from src.core.study_config import get_study_config
from src.evaluation.evaluator_runner import find_evaluator_path


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = REPO_ROOT / "extended_study" / "study_019"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SequentialSocialInfluenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = Study.load(STUDY_PATH)
        cls.config = get_study_config(
            "study_019",
            STUDY_PATH,
            cls.study.specification,
        )
        cls.material = cls.config.load_material("scenarios")
        cls.evaluator = load_module(
            "study_019_evaluator",
            STUDY_PATH / "scripts" / "evaluator.py",
        )
        cls.compiler = load_module(
            "study_019_material_compiler",
            STUDY_PATH / "source" / "materials" / "build_scenarios.py",
        )

    def test_package_loads_with_both_complete_scenario_sets(self):
        self.assertTrue(self.study.validate())
        self.assertEqual(
            self.study.metadata["implementation_scope"]["status"],
            "complete_source_grounded_scenario_tasks",
        )
        self.assertEqual(len(self.study.get_validation_criteria()), 4)
        self.assertEqual(self.study.get_pass_threshold(), 0.7)
        self.assertEqual(len(self.material["study_1"]["scenarios"]), 24)
        self.assertEqual(len(self.material["study_2"]["scenarios"]), 40)
        self.assertEqual(
            self.material["source"]["study_1_workbook"]["md5"],
            "e6f623799cce4b8331b806b55563b31c",
        )
        self.assertEqual(
            self.material["source"]["study_2_workbook"]["md5"],
            "c29df321e159c9cd0afbfa3defa73159",
        )

    def test_material_compiler_reproduces_committed_scenarios(self):
        raw_dir = STUDY_PATH / "source" / "raw_data"
        rebuilt = self.compiler.build(
            raw_dir / "Raw Data_Study1.xlsx",
            raw_dir / "RawData_Study2.xlsx",
        )
        self.assertEqual(rebuilt, self.material)

    def test_study_1_mirrored_presentations_restore_visible_orientation(self):
        scenarios = {
            row["scenario_id"]: row
            for row in self.material["study_1"]["scenarios"]
        }
        base = scenarios["study1_raw_01"]
        mirror = scenarios["study1_raw_13"]
        self.assertEqual(base["article_scenario_id"], 1)
        self.assertEqual(mirror["article_scenario_id"], 1)
        self.assertEqual(
            base["source_signature"],
            {
                "previous_decisions": ["A"],
                "private_ball": "white",
                "posterior_probability_urn_a": 0.8,
            },
        )
        self.assertEqual(
            mirror["source_signature"],
            {
                "previous_decisions": ["B"],
                "private_ball": "black",
                "posterior_probability_urn_a": 0.2,
            },
        )
        self.assertEqual(base["human_raw_data"]["option_1_rate"], 0.95)
        self.assertEqual(mirror["human_raw_data"]["option_1_rate"], 0.125)

    def test_study_2_role_order_and_authority_condition_are_preserved(self):
        scenarios = {
            row["scenario_id"]: row
            for row in self.material["study_2"]["scenarios"]
        }
        scenario = scenarios["study2_raw_22"]
        self.assertEqual(scenario["article_scenario_id"], 36)
        self.assertEqual(
            scenario["previous_diagnoses"],
            [
                {
                    "position": 1,
                    "role": "medical director",
                    "diagnosis": "appendicitis",
                }
            ],
        )
        self.assertEqual(scenario["private_symptom"], "regurgitation")
        self.assertEqual(
            scenario["authority_condition"],
            "medical_director_opposes_private",
        )
        self.assertEqual(scenario["posterior_probability_appendicitis"], 0.5)
        self.assertEqual(scenario["human_raw_data"]["option_1_rate"], 0.625)

        cascade = scenarios["study2_raw_25"]
        self.assertEqual(cascade["article_scenario_id"], 3)
        self.assertAlmostEqual(
            cascade["human_raw_data"]["mean_confidence"] / 100.0,
            0.65925,
        )
        self.assertAlmostEqual(
            cascade["human_raw_data"][
                "mean_probability_assigned_to_bayesian_choice"
            ],
            0.60075,
        )

    def test_choice_and_confidence_missingness_are_kept_separate(self):
        study_1 = {
            row["scenario_id"]: row["human_raw_data"]
            for row in self.material["study_1"]["scenarios"]
        }
        self.assertEqual(
            (
                study_1["study1_raw_07"]["n_choice"],
                study_1["study1_raw_07"]["n_confidence"],
                study_1["study1_raw_07"]["n_paired"],
            ),
            (40, 39, 39),
        )
        self.assertEqual(
            (
                study_1["study1_raw_08"]["n_choice"],
                study_1["study1_raw_08"]["n_confidence"],
                study_1["study1_raw_08"]["n_paired"],
            ),
            (40, 39, 39),
        )

        study_2 = {
            row["scenario_id"]: row["human_raw_data"]
            for row in self.material["study_2"]["scenarios"]
        }
        self.assertEqual(
            (
                study_2["study2_raw_23"]["n_choice"],
                study_2["study2_raw_23"]["n_confidence"],
                study_2["study2_raw_23"]["n_paired"],
            ),
            (39, 40, 39),
        )
        self.assertEqual(
            (
                study_2["study2_raw_39"]["n_choice"],
                study_2["study2_raw_39"]["n_confidence"],
                study_2["study2_raw_39"]["n_paired"],
            ),
            (40, 39, 39),
        )

    def test_compiled_raw_data_recovers_the_published_behavioral_rates(self):
        def weighted_rate(rows, key):
            observations = [
                (row["human_raw_data"][key], row["human_raw_data"]["n"])
                for row in rows
                if row["human_raw_data"].get(key) is not None
            ]
            return sum(value * count for value, count in observations) / sum(
                count for _, count in observations
            )

        study_1 = self.material["study_1"]["scenarios"]
        self.assertAlmostEqual(
            weighted_rate(
                [row for row in study_1 if row["bayesian_choice"] is not None],
                "bayesian_choice_rate",
            ),
            0.869,
            delta=0.002,
        )
        self.assertAlmostEqual(
            weighted_rate(
                [row for row in study_1 if row["cascade_scenario"]],
                "bayesian_choice_rate",
            ),
            0.755,
            delta=0.003,
        )
        self.assertAlmostEqual(
            weighted_rate(
                [row for row in study_1 if row["indifference_scenario"]],
                "private_choice_rate",
            ),
            0.799,
            delta=0.004,
        )

        study_2 = self.material["study_2"]["scenarios"]
        self.assertAlmostEqual(
            weighted_rate(
                [row for row in study_2 if row["bayesian_choice"] is not None],
                "bayesian_choice_rate",
            ),
            0.920,
            delta=0.003,
        )
        self.assertAlmostEqual(
            weighted_rate(
                [row for row in study_2 if row["cascade_scenario"]],
                "bayesian_choice_rate",
            ),
            0.821,
            delta=0.004,
        )
        self.assertAlmostEqual(
            weighted_rate(
                [row for row in study_2 if row["indifference_scenario"]],
                "private_choice_rate",
            ),
            0.497,
            delta=0.001,
        )
        self.assertAlmostEqual(
            weighted_rate(
                [
                    row
                    for row in study_2
                    if row["medical_director_diagnosis"] is not None
                ],
                "authority_choice_rate",
            ),
            0.7489,
            delta=0.001,
        )

    def test_response_parsers_and_balanced_participant_assignment(self):
        self.assertEqual(
            self.config.parse_urn_response("CHOICE=B; CONFIDENCE=73"),
            ("B", 73),
        )
        self.assertEqual(
            self.config.parse_medical_response(
                "DIAGNOSIS=SIGMOID_DIVERTICULITIS; CONFIDENCE=64"
            ),
            ("sigmoid diverticulitis", 64),
        )
        self.assertIsNone(
            self.config.parse_urn_response("CHOICE=A; CONFIDENCE=49")
        )
        self.assertIsNone(
            self.config.parse_medical_response(
                "DIAGNOSIS=APPENDICITIS; CONFIDENCE=101"
            )
        )
        trials = self.config.create_trials(n_trials=80)
        self.assertEqual(
            sum(row["sub_study_id"] == "study_1_urn_scenarios" for row in trials),
            40,
        )
        self.assertEqual(
            sum(
                row["sub_study_id"] == "study_2_medical_authority_scenarios"
                for row in trials
            ),
            40,
        )
        with self.assertRaises(ValueError):
            self.config.create_trials(n_trials=3)

    def test_prompts_contain_evidence_without_hidden_source_answers(self):
        builder = self.config.prompt_builder
        study_1_prompt = builder.build_urn_prompt(
            presented_trial_number=4,
            previous_decisions=["A", "B", "A"],
            private_ball="black",
        )
        study_2_prompt = builder.build_medical_prompt(
            presented_trial_number=9,
            previous_diagnoses=[
                {
                    "position": 1,
                    "role": "medical director",
                    "diagnosis": "appendicitis",
                },
                {
                    "position": 2,
                    "role": "assistant physician",
                    "diagnosis": "sigmoid diverticulitis",
                },
            ],
            private_symptom="regurgitation",
        )
        visible = (study_1_prompt + "\n" + study_2_prompt).lower()
        study_1_instructions = self.material["study_1"]["instructions"].lower()
        study_2_instructions = self.material["study_2"]["instructions"].lower()
        self.assertIn("predicted urn a", visible)
        self.assertIn("private ball is black", visible)
        self.assertIn("medical director diagnosed appendicitis", visible)
        self.assertIn("patient has this symptom: regurgitation", visible)
        self.assertIn("sees every urn prediction", study_1_instructions)
        self.assertIn(
            "must not be treated as an independent ball draw",
            study_1_instructions,
        )
        self.assertIn("each physician sees the diagnoses", study_2_instructions)
        self.assertIn("is not an independent medical test", study_2_instructions)
        for hidden in (
            "posterior probability",
            "bayesian choice",
            "human raw data",
            "reported value",
            "material fingerprint",
        ):
            self.assertNotIn(hidden, visible)

    def test_small_mock_stage5_passes_environment_but_not_replication(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_stage5(
                STUDY_PATH,
                runs_dir=Path(temp_dir),
                models=["mock"],
                options=Stage5Options(n_participants=2, mock=True, seed=17),
                data_dir=REPO_ROOT,
            )
            output = json.loads(
                Path(summary["runs"][0]["output_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(len(output["individual_data"]), 2)
            self.assertEqual(
                sum(len(row["responses"]) for row in output["individual_data"]),
                64,
            )
            evaluation = output["evaluation"]
            self.assertTrue(evaluation["environment_passed"])
            self.assertFalse(evaluation["behavioral_evaluable"])
            self.assertFalse(evaluation["behavioral_passed"])
            self.assertFalse(evaluation["passed"])
            self.assertIsNone(evaluation["behavioral_alignment_score"])
            self.assertIsNotNone(
                evaluation["provisional_behavioral_alignment_score"]
            )
            self.assertEqual(evaluation["execution_score"], 1.0)
            self.assertTrue(output["behavioral_diagnostics"])
            self.assertEqual(
                Path(evaluation["evaluator_path"]),
                find_evaluator_path("study_019", STUDY_PATH),
            )

    def test_full_original_mock_sample_produces_all_2560_responses(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_stage5(
                STUDY_PATH,
                runs_dir=Path(temp_dir),
                models=["mock"],
                options=Stage5Options(n_participants=80, mock=True, seed=23),
                data_dir=REPO_ROOT,
            )
            output = json.loads(
                Path(summary["runs"][0]["output_path"]).read_text(encoding="utf-8")
            )
            stats = output["descriptive_statistics"]
            self.assertEqual(stats["participants"], 80)
            self.assertEqual(
                stats["participants_by_sub_study"],
                {
                    "study_1_urn_scenarios": 40,
                    "study_2_medical_authority_scenarios": 40,
                },
            )
            self.assertEqual(stats["responses"], 2560)
            self.assertEqual(
                stats["responses_by_sub_study"],
                {
                    "study_1_urn_scenarios": 960,
                    "study_2_medical_authority_scenarios": 1600,
                },
            )
            evaluation = self.evaluator.evaluate_study(output)
            stage5_evaluation = output["evaluation"]
            self.assertTrue(evaluation["passed"])
            self.assertTrue(stage5_evaluation["passed"])
            self.assertTrue(stage5_evaluation["environment_passed"])
            self.assertTrue(stage5_evaluation["behavioral_passed"])
            self.assertEqual(
                stage5_evaluation["test_results"],
                evaluation["test_results"],
            )
            self.assertEqual(evaluation["execution_score"], 1.0)
            self.assertGreater(evaluation["behavioral_alignment_score"], 0.95)
            authority_cells = output["behavioral_diagnostics"][
                "study_2_medical_authority_scenarios"
            ]["authority_condition_cells"]
            self.assertIn("posterior_0.67", authority_cells)
            self.assertIn(
                "medical_director_opposes_private",
                authority_cells["posterior_0.67"],
            )

            constant = copy.deepcopy(output)
            for participant in constant["individual_data"]:
                for response in participant["responses"]:
                    if (
                        response["trial_info"]["sub_study_id"]
                        == "study_1_urn_scenarios"
                    ):
                        response["response"] = "A"
                    else:
                        response["response"] = "appendicitis"
                    response["trial_info"]["confidence"] = 50
            constant_evaluation = self.evaluator.evaluate_study(constant)
            self.assertTrue(constant_evaluation["environment_passed"])
            self.assertFalse(constant_evaluation["behavioral_passed"])
            self.assertFalse(constant_evaluation["passed"])


if __name__ == "__main__":
    unittest.main()
