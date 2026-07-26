from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

from effect_algebra.build_datasets import build_tree
from effect_algebra.compare_results import result_row
from effect_algebra.datasets import (
    a_normative_choice,
    build_a_rows,
    build_b_control_rows,
    build_b_rows,
    build_c_rows,
    load_jsonl,
)
from effect_algebra.evaluate_choices import summarize_scored_rows
from effect_algebra.modeling import validate_adapter_compatibility
from effect_algebra.validate_datasets import validate_dataset_tree, validate_rows


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_016_CONFIG = REPO_ROOT / "extended_study" / "study_016" / "scripts" / "config.py"
STUDY_019_SCENARIOS = (
    REPO_ROOT
    / "extended_study"
    / "study_019"
    / "source"
    / "materials"
    / "scenarios.json"
)


def load_study_016_config():
    scripts_dir = str(STUDY_016_CONFIG.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "effect_algebra_study_016_config",
        STUDY_016_CONFIG,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EffectAlgebraDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study_016 = load_study_016_config()

    def test_a_solver_matches_study_016_runtime_for_all_states(self):
        for history_length in range(6):
            for history in itertools.product(("A", "B"), repeat=history_length):
                for private_signal in ("A", "B"):
                    actual_choice, actual_posterior, actual_public = a_normative_choice(
                        history,
                        private_signal,
                    )
                    public_prior = 0.5
                    for choice in history:
                        public_prior = self.study_016._posterior_after_public_decision(
                            public_prior,
                            choice,
                            self.study_016.SYMMETRIC_URNS,
                        )
                    expected_choice, expected_posterior = (
                        self.study_016._choice_from_private_posterior(
                            public_prior,
                            "L" if private_signal == "A" else "D",
                            self.study_016.SYMMETRIC_URNS,
                        )
                    )
                    self.assertEqual(actual_choice, expected_choice)
                    self.assertAlmostEqual(actual_public, public_prior)
                    self.assertAlmostEqual(actual_posterior, expected_posterior)

    def test_a_rows_are_unique_balanced_and_valid(self):
        rows = build_a_rows("train", 128, seed=91)
        self.assertEqual(len({json.dumps(row["prompt"], sort_keys=True) for row in rows}), 128)
        self.assertEqual(
            {code: sum(row["target_code"] == code for row in rows) for code in ("X", "Y")},
            {"X": 64, "Y": 64},
        )
        template_counts = {
            family: sum(row["metadata"]["template_family"] == family for row in rows)
            for family in {"urn", "factory", "archive"}
        }
        self.assertEqual(set(template_counts.values()), {42, 44})
        for first, second in zip(rows[::2], rows[1::2]):
            self.assertEqual(
                first["metadata"]["state_hash"],
                second["metadata"]["state_hash"],
            )
            self.assertNotEqual(first["target_code"], second["target_code"])
        with tempfile.TemporaryDirectory() as temp_dir:
            report = validate_rows(Path(temp_dir) / "A_train.jsonl", rows)
        self.assertTrue(report["valid"], report["errors"])

    def test_a_generator_rejects_duplicate_capacity(self):
        with self.assertRaisesRegex(ValueError, "unique prompt/label mappings"):
            build_a_rows("dev", 253, seed=91)

    def test_b_rows_preserve_complete_feedback_episode(self):
        rows = build_b_rows(
            "train",
            12,
            seed=123,
            rounds_per_advisor=5,
        )
        for row in rows:
            metadata = row["metadata"]
            self.assertEqual(len(metadata["ledger"]), 10)
            self.assertTrue(
                all(record["correct_year"] is not None for record in metadata["ledger"])
            )
            target_advisor = metadata["code_to_advisor"][row["target_code"]]
            self.assertEqual(
                metadata["advisor_name_to_type"][target_advisor],
                "accurate",
            )
            self.assertGreaterEqual(metadata["accuracy_margin"], 5.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            report = validate_rows(Path(temp_dir) / "B_train.jsonl", rows)
        self.assertTrue(report["valid"], report["errors"])

    def test_b_no_feedback_control_has_no_label_or_hidden_answer(self):
        rows = build_b_control_rows(6, seed=456, rounds_per_advisor=4)
        for row in rows:
            self.assertIsNone(row["target_code"])
            self.assertNotIn("chosen", row)
            self.assertNotIn("rejected", row)
            self.assertTrue(
                all(
                    record["correct_year"] is None
                    for record in row["metadata"]["ledger"]
                )
            )

    def test_c_uses_all_source_scenarios_and_preserves_indifference(self):
        rows = build_c_rows(STUDY_019_SCENARIOS)
        self.assertEqual(len(rows), 40)
        self.assertEqual(sum(row["target_code"] is None for row in rows), 10)
        self.assertEqual(
            len({row["metadata"]["source_material_fingerprint"] for row in rows}),
            40,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = validate_rows(Path(temp_dir) / "C_test.jsonl", rows)
        self.assertTrue(report["valid"], report["errors"])

    def test_full_tree_is_valid_and_c_never_enters_training_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "data"
            args = argparse.Namespace(
                output_dir=output_dir,
                repo_root=REPO_ROOT,
                a_train=64,
                a_dev=16,
                a_test=16,
                b_train=24,
                b_dev=8,
                b_test=8,
                b_control=8,
                rounds_per_advisor=4,
                seed=777,
                skip_validation=False,
            )
            manifest = build_tree(args)
            report = validate_dataset_tree(output_dir)
            self.assertTrue(report["valid"], report["errors"])
            self.assertFalse(manifest["training_boundary"]["C_used_for_training"])
            for path in (output_dir / "dpo").glob("*.jsonl"):
                self.assertNotIn(
                    "C",
                    {row["effect"] for row in load_jsonl(path)},
                )

    def test_validator_rejects_tampered_a_label(self):
        rows = build_a_rows("train", 4, seed=12)
        rows[0]["target_code"] = "Y" if rows[0]["target_code"] == "X" else "X"
        with tempfile.TemporaryDirectory() as temp_dir:
            report = validate_rows(Path(temp_dir) / "A_train.jsonl", rows)
        self.assertFalse(report["valid"])


class EffectAlgebraModelTests(unittest.TestCase):
    def test_adapter_compatibility_rejects_different_rank(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [root / "a", root / "b"]
            for rank, path in zip((16, 8), paths):
                path.mkdir()
                (path / "adapter_config.json").write_text(
                    json.dumps(
                        {
                            "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
                            "peft_type": "LORA",
                            "task_type": "CAUSAL_LM",
                            "r": rank,
                            "lora_alpha": 32,
                            "target_modules": ["q_proj", "v_proj"],
                        }
                    ),
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                validate_adapter_compatibility(paths)

    def test_metric_summary_separates_normative_human_and_authority(self):
        rows = [
            {
                "effect": "C",
                "target_code": "X",
                "predicted_code": "X",
                "probability_by_code": {"X": 0.8, "Y": 0.2},
                "log_probability_by_code": {"X": -0.2, "Y": -1.5},
                "category": None,
                "authority_condition": "medical_director_supports_private",
                "medical_director_code": "X",
                "agreeing_advisor_code": None,
                "human_probability_by_code": {"X": 0.75, "Y": 0.25},
                "human_n": 40,
            },
            {
                "effect": "C",
                "target_code": None,
                "predicted_code": "Y",
                "probability_by_code": {"X": 0.4, "Y": 0.6},
                "log_probability_by_code": {"X": -0.9, "Y": -0.5},
                "category": None,
                "authority_condition": "medical_director_opposes_private",
                "medical_director_code": "Y",
                "agreeing_advisor_code": None,
                "human_probability_by_code": {"X": 0.3, "Y": 0.7},
                "human_n": 40,
            },
        ]
        summary = summarize_scored_rows(rows)
        self.assertEqual(summary["overall"]["labeled_rows"], 1)
        self.assertEqual(summary["overall"]["accuracy"], 1.0)
        self.assertEqual(summary["authority"]["hard_alignment_rate"], 1.0)
        self.assertAlmostEqual(
            summary["human_distribution"]["weighted_probability_mae"],
            0.075,
        )

    def test_control_metric_and_result_table_report_advisor_agreement(self):
        scored_rows = [
            {
                "effect": "B_control",
                "target_code": None,
                "predicted_code": "Y",
                "probability_by_code": {"X": 0.25, "Y": 0.75},
                "log_probability_by_code": {"X": -1.4, "Y": -0.3},
                "category": None,
                "authority_condition": None,
                "medical_director_code": None,
                "agreeing_advisor_code": "Y",
                "human_probability_by_code": None,
                "human_n": None,
            }
        ]
        summary = summarize_scored_rows(scored_rows)
        self.assertEqual(summary["advisor_agreement"]["hard_agreeing_choice_rate"], 1.0)
        self.assertEqual(
            summary["advisor_agreement"]["mean_agreeing_choice_probability"],
            0.75,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "B_control.json"
            result_path.write_text(
                json.dumps(
                    {
                        "model_label": "base",
                        "dataset": "/tmp/B_control.jsonl",
                        "summary": summary,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(result_row(result_path)["advisor_agreement"], 0.75)


if __name__ == "__main__":
    unittest.main()
