import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from generation_pipeline.stage5 import Stage5Options, run_stage5
from src.core.study import Study
from src.core.study_config import get_study_config


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = REPO_ROOT / "studies" / "study_017"


def load_evaluator():
    path = STUDY_PATH / "scripts" / "evaluator.py"
    spec = importlib.util.spec_from_file_location("study_017_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AdvisorChoiceDatesTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = Study.load(STUDY_PATH)
        cls.config = get_study_config("study_017", STUDY_PATH, cls.study.specification)

    def test_package_loads_and_question_bank_is_complete(self):
        self.assertTrue(self.study.validate())
        bank = self.config.load_material("question_bank")
        self.assertEqual(len(bank["questions"]), 74)
        prompts = {question["id"]: question["prompt"] for question in bank["questions"]}
        self.assertEqual(
            prompts["date_007"], "RMS Titanic begins her maiden voyage"
        )
        self.assertIn("Mrs Dalloway", prompts["date_014"])
        self.assertIn("Who Wants to be a Millionaire?", prompts["date_073"])

    def test_condition_cells_and_trial_units(self):
        trials = self.config.create_trials(n_trials=4)
        observed = [
            (
                trial["condition_assignment"]["feedback"],
                trial["condition_assignment"]["advisor_order"],
            )
            for trial in trials
        ]
        self.assertEqual(
            observed,
            [
                (False, "accurate_first"),
                (False, "agreeing_first"),
                (True, "accurate_first"),
                (True, "agreeing_first"),
            ],
        )
        full_trials = self.config.create_trials(n_trials=60)
        self.assertEqual(
            sum(not trial["condition_assignment"]["feedback"] for trial in full_trials),
            29,
        )
        self.assertEqual(
            sum(trial["condition_assignment"]["feedback"] for trial in full_trials),
            31,
        )

    def test_parsers_and_advice_generation(self):
        self.assertEqual(self.config.parse_estimate("YEAR=1954; WIDTH=7"), (1954, 7))
        self.assertIsNone(self.config.parse_estimate("YEAR=2054; WIDTH=7"))
        self.assertEqual(self.config.parse_advisor_choice("ADVISOR=b"), "B")
        module = importlib.import_module(self.config.__class__.__module__)
        rng = __import__("random").Random(3)
        year, mode = module.StudyStudy017Config._sample_advice(
            rng,
            advisor_type="accurate",
            initial_year=1960,
            correct_year=1954,
        )
        self.assertTrue(1893 <= year <= 2007)
        self.assertIn(mode, {"accurate", "reflected_control"})

    def test_agent_visible_prompts_do_not_label_hidden_policies(self):
        builder = self.config.prompt_builder
        initial = builder.build_initial_prompt(
            event_prompt="Roger Bannister runs the first 4-minute mile",
            trial_number=1,
            block_label="familiarisation block 1",
        )
        choice = builder.build_choice_prompt(
            advisor_a_name="Advisor #21", advisor_b_name="Advisor #44"
        )
        final = builder.build_final_prompt(
            event_prompt="Roger Bannister runs the first 4-minute mile",
            initial_year=1960,
            initial_width=13,
            advisor_name="Advisor #21",
            advice_year=1952,
            advice_width=6,
        )
        visible = (initial + choice + final).lower()
        self.assertNotIn("1954", visible)
        self.assertNotIn("accurate advisor", visible)
        self.assertNotIn("agreeing advisor", visible)

    def test_mock_stage5_runs_all_core_trials(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_stage5(
                STUDY_PATH,
                runs_dir=Path(temp_dir),
                models=["mock"],
                options=Stage5Options(n_participants=4, mock=True, seed=17),
                data_dir=REPO_ROOT,
            )
            output_path = Path(summary["runs"][0]["output_path"])
            output = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(output["individual_data"]), 4)
            self.assertEqual(
                sum(len(row["responses"]) for row in output["individual_data"]), 160
            )
            self.assertEqual(output["descriptive_statistics"]["choice_trials"], 40)
            self.assertEqual(output["descriptive_statistics"]["parse_failures"], 0)
            for participant in output["individual_data"]:
                feedback = participant["profile"]["feedback_condition"]
                self.assertTrue(
                    all(
                        response["trial_info"]["correct_year_revealed_after_final"]
                        is feedback
                        for response in participant["responses"]
                    )
                )
                visible = " ".join(
                    str(prompt or "")
                    for response in participant["responses"]
                    for prompt in response["trial_info"]["agent_visible_prompts"].values()
                ).lower()
                self.assertNotIn("accurate advisor", visible)
                self.assertNotIn("agreeing advisor", visible)

            evaluation = load_evaluator().evaluate_study(output)
            self.assertEqual(evaluation["execution_score"], 1.0)
            self.assertTrue(evaluation["passed"])


if __name__ == "__main__":
    unittest.main()
