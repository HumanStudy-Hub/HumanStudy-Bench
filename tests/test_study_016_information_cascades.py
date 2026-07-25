import importlib.util
import tempfile
import unittest
from pathlib import Path

from generation_pipeline.stage5 import Stage5Options, run_stage5
from src.core.study import Study
from src.core.study_config import get_study_config


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = REPO_ROOT / "extended_study" / "study_016"


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
        self.assertEqual(
            self.study.metadata["implementation_scope"]["status"],
            "evidence_complete_treatments_implemented",
        )
        self.assertEqual(self.study.specification["participants"]["n"], 66)

    def test_trial_count_requires_complete_sessions(self):
        self.assertEqual(len(self.config.create_trials()), 66)
        self.assertEqual(len(self.config.create_trials(n_trials=6)), 6)
        self.assertEqual(len(self.config.create_trials(n_trials=18)), 18)
        first_thirty = self.config.create_trials(n_trials=30)
        self.assertEqual(
            [first_thirty[index]["sub_study_id"] for index in (0, 18, 24)],
            [
                "symmetric_baseline",
                "symmetric_public_draw_after_position_4",
                "symmetric_public_draw_after_position_4",
            ],
        )
        self.assertEqual(
            self.config.create_trials(n_trials=66)[-1]["sub_study_id"],
            "asymmetric_baseline",
        )
        with self.assertRaises(ValueError):
            self.config.create_trials(n_trials=5)
        with self.assertRaises(ValueError):
            self.config.create_trials(n_trials=7)
        with self.assertRaises(ValueError):
            self.config.create_trials(n_trials=72)

    def test_public_evidence_and_tie_breaking(self):
        module = importlib.import_module(self.config.__class__.__module__)
        self.assertEqual(module.bayesian_choice("D", ["A", "A"]), "A")
        self.assertEqual(module.bayesian_choice("L", ["B", "B"]), "B")
        self.assertEqual(module.bayesian_choice("D", ["A"]), "B")
        self.assertEqual(module.bayesian_choice("L", ["B"]), "A")

    def test_probability_updating_supports_asymmetric_urns(self):
        module = importlib.import_module(self.config.__class__.__module__)
        urns = {
            "A": {"light": 6, "dark": 1},
            "B": {"light": 5, "dark": 2},
        }
        prior_a = 0.5
        for _ in range(3):
            prior_a = module._posterior_after_signal(prior_a, "L", urns)
        choice, posterior_a = module._choice_from_private_posterior(prior_a, "D", urns)
        self.assertEqual(choice, "B")
        self.assertAlmostEqual(posterior_a, 0.46, places=2)

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
        self.assertIn("Earlier public predictions this period: A, B", prompt)
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
            self.assertTrue(evaluation["passed"])
            practice = output["individual_data"][0]["profile"]["practice_periods"]
            self.assertEqual({period["true_urn"] for period in practice}, {"A", "B"})
            self.assertTrue(all(period["decisions_required"] is False for period in practice))

    def test_full_mock_run_covers_all_evidence_complete_treatments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_stage5(
                STUDY_PATH,
                runs_dir=Path(temp_dir),
                models=["mock"],
                options=Stage5Options(n_participants=66, mock=True, seed=23),
                data_dir=REPO_ROOT,
            )

            import json

            output = json.loads(
                Path(summary["runs"][0]["output_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(len(output["individual_data"]), 66)
            self.assertEqual(
                sum(len(row["responses"]) for row in output["individual_data"]),
                990,
            )
            self.assertEqual(output["descriptive_statistics"]["sessions"], 11)
            self.assertEqual(output["descriptive_statistics"]["periods"], 165)
            self.assertEqual(
                set(output["descriptive_statistics"]["by_treatment"]),
                {
                    "symmetric_baseline",
                    "symmetric_public_draw_after_position_4",
                    "asymmetric_baseline",
                },
            )

            public_session = [
                response
                for row in output["individual_data"]
                for response in row["responses"]
                if response["trial_info"]["session_index"] == 3
                and response["trial_info"]["period_number"] == 1
            ]
            public_session.sort(key=lambda row: row["trial_info"]["decision_position"])
            self.assertTrue(
                all(
                    response["trial_info"]["public_draws_visible"] == []
                    for response in public_session[:4]
                )
            )
            self.assertTrue(
                all(
                    "Additional public draws now visible:" not in response["prompt"]
                    for response in public_session[:4]
                )
            )
            self.assertEqual(
                len(public_session[4]["trial_info"]["public_draws_visible"]),
                2,
            )
            self.assertTrue(
                all(
                    "Additional public draws now visible:" in response["prompt"]
                    for response in public_session[4:]
                )
            )
            self.assertEqual(
                public_session[4]["trial_info"]["public_draws_visible"],
                public_session[5]["trial_info"]["public_draws_visible"],
            )

            asymmetric = next(
                response
                for row in output["individual_data"]
                for response in row["responses"]
                if response["trial_info"]["session_index"] == 5
            )
            self.assertEqual(
                asymmetric["trial_info"]["urns"],
                {
                    "A": {"light": 6, "dark": 1},
                    "B": {"light": 5, "dark": 2},
                },
            )

            evaluation = load_evaluator().evaluate_study(output)
            self.assertEqual(evaluation["execution_score"], 1.0)
            self.assertTrue(evaluation["passed"])


if __name__ == "__main__":
    unittest.main()
