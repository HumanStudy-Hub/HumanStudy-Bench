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
    a_bucket_coverage,
    a_normative_choice,
    a_state_bucket,
    build_a_rows,
    build_b_control_rows,
    build_b_rows,
    build_c_rows,
    build_c_training_rows,
    build_d_rows,
    build_d_training_rows,
    c_stratified_folds,
    load_c_scenarios,
    load_jsonl,
    proportional_label_flags,
)
from effect_algebra.human_priors import (
    A_PRIORS,
    b_prior,
    binomial_noise_floor,
    dpo_optimal_beta,
    dpo_reachable,
    trivial_baselines,
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
        # Response codes stay balanced even though a share of labels is flipped
        # to express the human proportion.
        self.assertEqual(
            {code: sum(row["target_code"] == code for row in rows) for code in ("X", "Y")},
            {"X": 64, "Y": 64},
        )
        for first, second in zip(rows[::2], rows[1::2]):
            self.assertEqual(
                first["metadata"]["state_hash"],
                second["metadata"]["state_hash"],
            )
            self.assertNotEqual(
                first["metadata"]["reference_code"],
                second["metadata"]["reference_code"],
            )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = validate_rows(
                Path(temp_dir) / "A_train.jsonl", rows, role="trainable"
            )
        self.assertTrue(report["valid"], report["errors"])

    def test_a_labels_reproduce_published_bucket_proportions(self):
        rows = build_a_rows("train", 512, seed=91)
        by_bucket = {}
        for row in rows:
            by_bucket.setdefault(row["metadata"]["bucket"], []).append(row)
        self.assertIn("cascade", by_bucket)
        for bucket, group in by_bucket.items():
            prior = A_PRIORS[bucket]
            self.assertIsNotNone(prior.probability)
            observed = sum(
                1 for row in group
                if row["target_code"] == row["metadata"]["reference_code"]
            ) / len(group)
            # Rounding a ratio onto a finite group leaves at most half a row
            # of slack in each of the two response-code halves.
            self.assertAlmostEqual(
                observed,
                prior.probability,
                delta=max(0.03, 1.0 / len(group)),
            )
            for row in group:
                self.assertAlmostEqual(
                    row["human_probability_by_code"][row["metadata"]["reference_code"]],
                    prior.probability,
                )

    def test_a_excludes_uncalibrated_states_and_reports_coverage(self):
        coverage = a_bucket_coverage()
        self.assertEqual(coverage["total_states"], 126)
        self.assertEqual(coverage["uncalibrated_buckets"], ["tie_no_conflict"])
        self.assertAlmostEqual(coverage["coverage"], 114 / 126)
        rows = build_a_rows("train", 256, seed=5)
        self.assertNotIn(
            "tie_no_conflict",
            {row["metadata"]["bucket"] for row in rows},
        )

    def test_a_bucketing_has_no_unreachable_branch(self):
        buckets = {
            a_state_bucket(list(history), signal)
            for length in range(6)
            for history in itertools.product(("A", "B"), repeat=length)
            for signal in ("A", "B")
        }
        self.assertEqual(buckets, set(A_PRIORS))

    def test_a_generator_rejects_duplicate_capacity(self):
        with self.assertRaisesRegex(ValueError, "unique calibrated prompt/label"):
            build_a_rows("dev", 1000, seed=91)

    def test_proportional_labels_avoid_duplicating_prompts(self):
        flags = proportional_label_flags(20, 0.7)
        self.assertEqual(sum(flags), 14)
        self.assertEqual(len(flags), 20)

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
            reference_advisor = metadata["code_to_advisor"][metadata["reference_code"]]
            self.assertEqual(
                metadata["advisor_name_to_type"][reference_advisor],
                "accurate",
            )
            self.assertGreaterEqual(metadata["accuracy_margin"], 5.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            report = validate_rows(
                Path(temp_dir) / "B_train.jsonl", rows, role="trainable"
            )
        self.assertTrue(report["valid"], report["errors"])

    def test_b_labels_follow_the_published_pick_rate(self):
        rows = build_b_rows("train", 200, seed=7, rounds_per_advisor=3)
        prior = b_prior("3C")
        self.assertAlmostEqual(prior.probability, 0.83)
        observed = sum(
            1 for row in rows
            if row["target_code"] == row["metadata"]["reference_code"]
        ) / len(rows)
        self.assertAlmostEqual(observed, prior.probability, delta=0.02)

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
        for row in rows:
            self.assertFalse(row["trainable"])
            self.assertNotIn("chosen", row)
            self.assertNotIn("rejected", row)
        with tempfile.TemporaryDirectory() as temp_dir:
            report = validate_rows(Path(temp_dir) / "C_test.jsonl", rows, role="eval")
        self.assertTrue(report["valid"], report["errors"])

    def test_c_training_rows_match_per_scenario_human_rates(self):
        scenarios = load_c_scenarios(STUDY_019_SCENARIOS)
        chosen = [str(scenario["scenario_id"]) for scenario in scenarios[:4]]
        rows = build_c_training_rows(
            STUDY_019_SCENARIOS, chosen, replicas=20, seed=3
        )
        self.assertEqual(len(rows), 80)
        by_scenario = {}
        for row in rows:
            by_scenario.setdefault(row["metadata"]["scenario_id"], []).append(row)
        for scenario_id, group in by_scenario.items():
            expected = float(group[0]["metadata"]["human_option_1_rate"])
            observed = sum(
                1 for row in group
                if row["target_code"] == row["metadata"]["reference_code"]
            ) / len(group)
            self.assertAlmostEqual(observed, expected, delta=0.06)
        with tempfile.TemporaryDirectory() as temp_dir:
            report = validate_rows(
                Path(temp_dir) / "C_fold0_train.jsonl", rows, role="cv_train"
            )
        self.assertTrue(report["valid"], report["errors"])

    def test_c_folds_cover_every_scenario_exactly_once(self):
        scenarios = load_c_scenarios(STUDY_019_SCENARIOS)
        folds = c_stratified_folds(scenarios, folds=5, seed=1)
        self.assertEqual(len(folds), 5)
        flattened = [scenario_id for fold in folds for scenario_id in fold]
        self.assertEqual(len(flattened), 40)
        self.assertEqual(len(set(flattened)), 40)
        for fold in folds:
            self.assertGreaterEqual(len(fold), 7)

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
                b_probe_rounds="2,4",
                b_probe_count=8,
                d_replicas=20,
                d_holdout=6,
                c_folds=5,
                c_replicas=20,
                seed=777,
                skip_validation=False,
            )
            manifest = build_tree(args)
            report = validate_dataset_tree(output_dir)
            self.assertTrue(report["valid"], report["errors"])
            self.assertFalse(
                manifest["training_boundary"]["c_test_used_for_training"]
            )
            for path in (output_dir / "dpo").glob("*.jsonl"):
                self.assertNotIn(
                    "C",
                    {row["effect"] for row in load_jsonl(path)},
                )
            # The C evaluation set carries no preference pair at all, so a
            # training entry point cannot consume it even by mistake.
            for row in load_jsonl(output_dir / "eval" / "C_test.jsonl"):
                self.assertFalse(row["trainable"])
                self.assertNotIn("chosen", row)
            for fold in report["summary"]["cv_folds"].values():
                self.assertEqual(fold["overlap"], 0)
            # The transfer claim needs D's source disjoint from both its own
            # held-out scenarios and from the C evaluation set.
            self.assertEqual(report["summary"]["d_split"]["overlap"], 0)
            self.assertEqual(report["summary"]["d_split"]["source_scenarios"], 18)
            c_scenarios = {
                row["metadata"]["scenario_id"]
                for row in load_jsonl(output_dir / "eval" / "C_test.jsonl")
            }
            for path in (output_dir / "dpo").glob("*.jsonl"):
                training = {
                    row["metadata"]["scenario_id"]
                    for row in load_jsonl(path)
                    if "scenario_id" in row.get("metadata", {})
                }
                self.assertEqual(training & c_scenarios, set(), str(path))

    def test_training_entry_point_refuses_evaluation_rows(self):
        from effect_algebra.train_dpo import _training_records

        rows = build_c_rows(STUDY_019_SCENARIOS)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "C_test.jsonl"
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not trainable"):
                _training_records(path)

    def test_validator_rejects_tampered_a_reference(self):
        rows = build_a_rows("train", 4, seed=12)
        metadata = rows[0]["metadata"]
        metadata["reference_code"] = (
            "Y" if metadata["reference_code"] == "X" else "X"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = validate_rows(
                Path(temp_dir) / "A_train.jsonl", rows, role="trainable"
            )
        self.assertFalse(report["valid"])

    def test_validator_rejects_a_human_proportion_that_is_not_published(self):
        rows = build_a_rows("train", 40, seed=12)
        rows[0]["metadata"]["human_probability"] = 0.5
        rows[0]["human_probability_by_code"] = {"X": 0.5, "Y": 0.5}
        with tempfile.TemporaryDirectory() as temp_dir:
            report = validate_rows(
                Path(temp_dir) / "A_train.jsonl", rows, role="trainable"
            )
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
                "bucket": None,
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
                "bucket": None,
                "authority_condition": "medical_director_opposes_private",
                "medical_director_code": "Y",
                "agreeing_advisor_code": None,
                "human_probability_by_code": {"X": 0.3, "Y": 0.7},
                "human_n": 40,
            },
        ]
        summary = summarize_scored_rows(rows)
        self.assertEqual(summary["normative"]["labeled_rows"], 1)
        self.assertEqual(summary["normative"]["accuracy"], 1.0)
        self.assertEqual(summary["authority"]["hard_alignment_rate"], 1.0)
        # Calibration is the primary metric and is not weighted by n here:
        # |0.80 - 0.75| and |0.40 - 0.30| average to 0.075.
        self.assertAlmostEqual(summary["calibration"]["mae"], 0.075)
        self.assertAlmostEqual(
            summary["calibration"]["by_authority_condition"][
                "medical_director_opposes_private"
            ]["mae"],
            0.1,
        )

    def test_control_metric_and_result_table_report_advisor_agreement(self):
        scored_rows = [
            {
                "effect": "B_control",
                "target_code": None,
                "predicted_code": "Y",
                "probability_by_code": {"X": 0.25, "Y": 0.75},
                "log_probability_by_code": {"X": -1.4, "Y": -0.3},
                "bucket": None,
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


class EffectDAndFlywheelTests(unittest.TestCase):
    def test_d_rows_come_from_study_1_with_per_scenario_rates(self):
        rows = build_d_rows(STUDY_019_SCENARIOS)
        self.assertEqual(len(rows), 24)
        for row in rows:
            self.assertFalse(row["trainable"])
            self.assertNotIn("chosen", row)
            # Study 1 has a few missing choices, so denominators are per
            # scenario rather than a flat 40.
            self.assertIn(row["metadata"]["human_denominator"], (39, 40))
        self.assertEqual(
            len({row["metadata"]["source_material_fingerprint"] for row in rows}),
            24,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = validate_rows(Path(temp_dir) / "D_test.jsonl", rows, role="eval")
        self.assertTrue(report["valid"], report["errors"])

    def test_d_training_rows_match_per_scenario_human_rates(self):
        rows = build_d_training_rows(STUDY_019_SCENARIOS, replicas=20, seed=2)
        self.assertEqual(len(rows), 480)
        by_scenario = {}
        for row in rows:
            by_scenario.setdefault(row["metadata"]["scenario_id"], []).append(row)
        self.assertEqual(len(by_scenario), 24)
        for group in by_scenario.values():
            expected = float(group[0]["metadata"]["human_option_1_rate"])
            observed = sum(
                1 for row in group
                if row["target_code"] == row["metadata"]["reference_code"]
            ) / len(group)
            self.assertAlmostEqual(observed, expected, delta=0.06)

    def test_flywheel_subsets_hold_proportions_and_code_balance(self):
        from effect_algebra.flywheel import (
            build_conditions,
            build_pool,
            subsample_preserving_proportions,
        )

        pool = build_pool(
            REPO_ROOT,
            a_count=128,
            b_count=64,
            d_replicas=10,
            rounds_per_advisor=3,
            seed=11,
        )
        built = build_conditions(
            pool,
            row_budget=120,
            conditions=["A_only", "A_plus_B", "A_plus_B_plus_D"],
            seed=11,
        )
        for name, payload in built.items():
            rows = payload["rows"]
            # Matched budget: diversity is compared at equal data volume, so an
            # improvement cannot be attributed to simply having more rows.
            self.assertEqual(len(rows), 120, name)
            codes = [row["target_code"] for row in rows]
            self.assertAlmostEqual(
                codes.count("X") / len(codes), 0.5, delta=0.02, msg=name
            )
            for effect in payload["effects"]:
                group = [row for row in rows if row["effect"] == effect]
                observed = sum(
                    1 for row in group
                    if row["target_code"] == row["metadata"]["reference_code"]
                ) / len(group)
                expected = sum(
                    row["human_probability_by_code"][row["metadata"]["reference_code"]]
                    for row in group
                ) / len(group)
                self.assertAlmostEqual(observed, expected, delta=0.08, msg=name)

    def test_subsample_is_deterministic_for_a_fixed_seed(self):
        from effect_algebra.flywheel import subsample_preserving_proportions

        rows = build_a_rows("train", 128, seed=3)
        first = subsample_preserving_proportions(rows, 40, seed=9)
        second = subsample_preserving_proportions(rows, 40, seed=9)
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertEqual(len(first), 40)


class ReferenceModelTests(unittest.TestCase):
    def test_bayesian_reference_is_far_from_humans_but_beats_chance(self):
        from effect_algebra.reference_models import score_reference_model

        rows = build_c_rows(STUDY_019_SCENARIOS)
        bayesian = score_reference_model(rows, "bayesian_hard")
        uniform = score_reference_model(rows, "uniform_half")
        oracle = score_reference_model(rows, "human_oracle")
        # A perfectly rational agent is closer to humans than chance, but still
        # measurably apart: that gap is what the calibration study is about.
        self.assertLess(
            bayesian["summary"]["calibration"]["mae"],
            uniform["summary"]["calibration"]["mae"],
        )
        self.assertGreater(bayesian["summary"]["calibration"]["mae"], 0.05)
        self.assertAlmostEqual(oracle["summary"]["calibration"]["mae"], 0.0)

    def test_oracles_are_labelled_so_they_cannot_be_reported_as_models(self):
        from effect_algebra.reference_models import REFERENCE_MODELS

        self.assertTrue(REFERENCE_MODELS["human_oracle"]["oracle"])
        self.assertTrue(REFERENCE_MODELS["condition_mean_oracle"]["oracle"])
        self.assertFalse(REFERENCE_MODELS["bayesian_hard"]["oracle"])
        self.assertFalse(REFERENCE_MODELS["uniform_half"]["oracle"])


try:  # torch is only needed by the training modules, not the data layer.
    import torch as _torch
except ImportError:  # pragma: no cover - exercised on data-only environments
    _torch = None


@unittest.skipIf(_torch is None, "torch is not installed in this environment")
class SoftLabelObjectiveTests(unittest.TestCase):
    """The main method's core claim, checked numerically rather than asserted."""

    def _batch(self, target_x):
        from effect_algebra.train_soft import SoftLabelCollator

        collator = SoftLabelCollator(pad_token_id=0)
        return collator(
            [
                {
                    "input_ids": [5, 6, 7],
                    "token_x": 11,
                    "token_y": 12,
                    "target_x": target_x,
                    "target_y": 1.0 - target_x,
                }
            ]
        )

    def _model(self, gap):
        import torch

        class Model(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None):
                batch, width = input_ids.shape
                logits = torch.zeros((batch, width, 50))
                logits[0, 2, 11] = gap
                logits[0, 2, 12] = 0.0
                return type("Output", (object,), {"logits": logits})()

        return Model()

    def test_collator_points_at_each_sequence_own_final_token(self):
        from effect_algebra.train_soft import SoftLabelCollator

        collator = SoftLabelCollator(pad_token_id=0)
        batch = collator(
            [
                {"input_ids": [5, 6, 7], "token_x": 11, "token_y": 12,
                 "target_x": 0.7, "target_y": 0.3},
                {"input_ids": [1, 2, 3, 4, 9], "token_x": 21, "token_y": 22,
                 "target_x": 0.2, "target_y": 0.8},
            ]
        )
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 5))
        self.assertEqual(batch["last_index"].tolist(), [2, 4])
        self.assertEqual(batch["attention_mask"].sum(1).tolist(), [3, 5])
        self.assertEqual(batch["input_ids"][0, 3:].tolist(), [0, 0])

    def test_loss_minimum_is_exactly_the_human_proportion(self):
        import math

        from effect_algebra.train_soft import soft_label_loss

        for target in (0.3, 0.7, 0.9):
            batch = self._batch(target)
            expected_gap = math.log(target / (1.0 - target))
            candidates = [expected_gap + offset for offset in (-1.0, -0.3, 0.0, 0.3, 1.0)]
            losses = [
                float(soft_label_loss(self._model(gap), batch).detach())
                for gap in candidates
            ]
            self.assertEqual(min(range(len(losses)), key=losses.__getitem__), 2, target)

    def test_overconfidence_is_penalised_unlike_proportional_dpo(self):
        from effect_algebra.train_soft import soft_label_loss

        batch = self._batch(0.7)
        at_target = float(soft_label_loss(self._model(0.847), batch).detach())
        overshot = float(soft_label_loss(self._model(6.0), batch).detach())
        # Pushing past the human proportion costs more, which is the property
        # proportional DPO structurally cannot have: its optimum shift always
        # carries the sign of the majority option, so it can only sharpen.
        self.assertGreater(overshot, at_target + 1.0)


class LetterBiasTests(unittest.TestCase):
    """Forced-choice answers carry a preference for the letter itself."""

    def _pair(self, p_x_first, p_x_mirror):
        # Same item under both letter assignments. In the first row the
        # reference option is X; in the mirror it is Y.
        return [
            {
                "id": "row", "effect": "C", "split": "test", "pair_id": "state",
                "reference_code": "X", "target_code": "X",
                "predicted_code": "X" if p_x_first >= 0.5 else "Y",
                "probability_by_code": {"X": p_x_first, "Y": 1 - p_x_first},
                "log_probability_by_code": {"X": 0.0, "Y": 0.0},
                "human_probability_by_code": {"X": 0.7, "Y": 0.3}, "human_n": 40,
            },
            {
                "id": "row_m", "effect": "C", "split": "test", "pair_id": "state",
                "reference_code": "Y", "target_code": "Y",
                "predicted_code": "X" if p_x_mirror >= 0.5 else "Y",
                "probability_by_code": {"X": p_x_mirror, "Y": 1 - p_x_mirror},
                "log_probability_by_code": {"X": 0.0, "Y": 0.0},
                "human_probability_by_code": {"X": 0.3, "Y": 0.7}, "human_n": 40,
            },
        ]

    def test_pure_letter_bias_cancels_to_chance(self):
        from effect_algebra.evaluate_choices import merge_mirror_pairs

        # A model that always answers Y regardless of content.
        merged, report = merge_mirror_pairs(self._pair(0.02, 0.02))
        self.assertEqual(report["paired_items"], 1)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0]["probability_by_code"]["X"], 0.5, places=6)

    def test_content_signal_survives_symmetrization(self):
        from effect_algebra.evaluate_choices import merge_mirror_pairs

        # Reference option favoured in both frames: 0.8 as X, 0.8 as Y.
        merged, _ = merge_mirror_pairs(self._pair(0.8, 0.2))
        self.assertAlmostEqual(merged[0]["probability_by_code"]["X"], 0.8, places=6)

    def test_symmetrization_lowers_mae_against_a_biased_model(self):
        from effect_algebra.evaluate_choices import summarize_scored_rows

        rows = self._pair(0.02, 0.02)
        biased = summarize_scored_rows(rows, symmetrize=False)["calibration"]["mae"]
        fixed = summarize_scored_rows(rows, symmetrize=True)["calibration"]["mae"]
        # |0.02 - 0.7| and |0.02 - 0.3| average to 0.48.
        self.assertAlmostEqual(biased, 0.48, places=6)
        self.assertAlmostEqual(fixed, 0.2, places=6)
        self.assertLess(fixed, biased)

    def test_a_no_information_model_scores_chance_not_perfect(self):
        from effect_algebra.evaluate_choices import summarize_scored_rows

        # Both orders identical at 0.5: the model carries no information.
        summary = summarize_scored_rows(self._pair(0.5, 0.5))["normative"]
        self.assertAlmostEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["tied_rows"], 1)
        # A model that does identify the target still scores 1.0.
        confident = summarize_scored_rows(self._pair(0.9, 0.1))["normative"]
        self.assertAlmostEqual(confident["accuracy"], 1.0)
        self.assertEqual(confident["tied_rows"], 0)

    def test_symmetrization_survives_a_saturated_letter_bias(self):
        import math

        from effect_algebra.evaluate_choices import merge_mirror_pairs

        sigmoid = lambda z: 1.0 / (1.0 + math.exp(-z))

        def saturated(bias, content):
            # One frame scores bias + content, the mirror scores content - bias.
            # Log probabilities are what the scorer actually records, so supply
            # them the way a real run would.
            first, second = bias + content, -bias + content
            return [
                {
                    "id": "a", "effect": "C", "split": "test", "pair_id": "s",
                    "reference_code": "X", "target_code": "X", "predicted_code": "X",
                    "probability_by_code": {"X": sigmoid(first), "Y": 1 - sigmoid(first)},
                    "log_probability_by_code": {"X": first, "Y": 0.0},
                    "human_probability_by_code": {"X": 0.8, "Y": 0.2}, "human_n": 40,
                },
                {
                    "id": "b", "effect": "C", "split": "test", "pair_id": "s",
                    "reference_code": "Y", "target_code": "Y", "predicted_code": "X",
                    "probability_by_code": {"X": 1 - sigmoid(second), "Y": sigmoid(second)},
                    "log_probability_by_code": {"X": 0.0, "Y": second},
                    "human_probability_by_code": {"X": 0.2, "Y": 0.8}, "human_n": 40,
                },
            ]

        # Averaging probabilities would collapse the large-bias cases to 0.5,
        # because both frames read as near-certainty. Averaging log odds is
        # exact at every bias magnitude.
        for bias in (1.0, 3.5, 8.19, 15.0, 30.0):
            merged, _ = merge_mirror_pairs(saturated(bias, 2.0))
            self.assertAlmostEqual(
                merged[0]["probability_by_code"]["X"],
                sigmoid(2.0),
                places=6,
                msg="bias={}".format(bias),
            )

    def test_unpaired_rows_pass_through(self):
        from effect_algebra.evaluate_choices import merge_mirror_pairs

        rows = self._pair(0.8, 0.2)
        del rows[1]
        merged, report = merge_mirror_pairs(rows)
        self.assertEqual(len(merged), 1)
        self.assertEqual(report["paired_items"], 0)

    def test_evaluation_sets_carry_both_letter_assignments(self):
        from collections import Counter

        rows = build_c_rows(STUDY_019_SCENARIOS) + build_c_rows(
            STUDY_019_SCENARIOS, mirror=True
        )
        self.assertEqual(len(rows), 80)
        groups = Counter(row["metadata"]["state_hash"] for row in rows)
        self.assertEqual(set(groups.values()), {2})
        codes = Counter(
            (row["metadata"]["state_hash"], row["metadata"]["reference_code"])
            for row in rows
        )
        self.assertEqual(set(codes.values()), {1})
        self.assertEqual(len({row["id"] for row in rows}), 80)


class CalibrationScaleTests(unittest.TestCase):
    def test_noise_floor_is_positive_and_below_trivial_baselines(self):
        scenarios = load_c_scenarios(STUDY_019_SCENARIOS)
        rates = [
            float(scenario["human_raw_data"]["option_1_rate"]) for scenario in scenarios
        ]
        counts = [
            int(scenario["human_raw_data"]["n_choice"]) for scenario in scenarios
        ]
        floor = binomial_noise_floor(rates, counts, samples=400, seed=1)
        baselines = trivial_baselines(rates)
        # A perfect model still misses by the sampling error of an n=40 estimate.
        self.assertGreater(floor["mean"], 0.02)
        self.assertLess(floor["mean"], 0.06)
        self.assertLess(floor["mean"], baselines["always_half"])
        self.assertGreater(baselines["always_half"], 0.3)

    def test_dpo_cannot_soften_an_overshooting_base_model(self):
        # Base closer to 0.5 than the humans: reachable.
        self.assertTrue(dpo_reachable(0.75, 0.60))
        self.assertIsNotNone(dpo_optimal_beta(0.75, 0.60))
        # Base already more extreme in the same direction: no positive beta
        # exists, and every positive beta pushes it further out.
        self.assertFalse(dpo_reachable(0.75, 0.95))
        self.assertIsNone(dpo_optimal_beta(0.75, 0.95))
        self.assertFalse(dpo_reachable(0.95, 0.99))
        # Crossing the midpoint is fine, because the majority option flips.
        self.assertTrue(dpo_reachable(0.43, 0.90))


if __name__ == "__main__":
    unittest.main()
