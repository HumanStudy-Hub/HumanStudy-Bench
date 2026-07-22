import importlib.util
import tempfile
import unittest
from pathlib import Path

from generation_pipeline.stage5 import Stage5Options, run_stage5
from src.core.study import Study
from src.core.study_config import get_study_config


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = REPO_ROOT / "studies" / "study_016"


def load_evaluator():
    path = STUDY_PATH / "scripts" / "evaluator.py"
    spec = importlib.util.spec_from_file_location("study_016_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class InformationCascadeStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = Study.load(STUDY_PATH)
        cls.config = get_study_config("study_016", STUDY_PATH, cls.study.specification)

    def test_package_loads_and_validates(self):
        self.assertTrue(self.study.validate())
        self.assertEqual(self.study.metadata["implementation_scope"]["status"], "core_treatment_complete")

    def test_trial_count_requires_complete_sessions(self):
        self.assertEqual(len(self.config.create_trials(n_trials=6)), 6)
        self.assertEqual(len(self.config.create_trials(n_trials=18)), 18)
        with self.assertRaises(ValueError):
            self.config.create_trials(n_trials=5)
        with self.assertRaises(ValueError):
            self.config.create_trials(n_trials=7)

    def test_public_evidence_and_tie_breaking(self):
        module = importlib.import_module(self.config.__class__.__module__)
        self.assertEqual(module.bayesian_choice("D", ["A", "A"]), "A")
        self.assertEqual(module.bayesian_choice("L", ["B", "B"]), "B")
        self.assertEqual(module.bayesian_choice("D", ["A"]), "B")
        self.assertEqual(module.bayesian_choice("L", ["B"]), "A")

    def test_prompt_contains_only_decision_time_information(self):
        prompt = self.config.prompt_builder.build_decision_prompt(
            period_number=1,
            decision_position=3,
            private_signal="D",
            prior_decisions=["A", "B"],
            cumulative_earnings=0,
            previous_period=None,
        )
        self.assertIn("Your private draw: Dark", prompt)
        self.assertIn("Earlier public decisions in this period: A, B", prompt)
        self.assertNotIn("Urn used:", prompt)

    def test_mock_stage5_runs_one_complete_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_stage5(
                STUDY_PATH,
                runs_dir=Path(temp_dir),
                models=["mock"],
                options=Stage5Options(n_participants=6, mock=True, seed=17),
                data_dir=REPO_ROOT,
            )
            output_path = Path(summary["runs"][0]["output_path"])
            self.assertTrue(output_path.exists())

            import json

            output = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(output["individual_data"]), 6)
            self.assertEqual(sum(len(row["responses"]) for row in output["individual_data"]), 90)
            self.assertEqual(output["descriptive_statistics"]["periods"], 15)
            self.assertEqual(output["descriptive_statistics"]["parse_failures"], 0)

            evaluation = load_evaluator().evaluate_study(output)
            self.assertEqual(evaluation["execution_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
