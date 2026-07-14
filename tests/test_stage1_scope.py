import unittest

from generation_pipeline.pdf.models import DocumentBlock
from generation_pipeline.extractors.study_data_extractor import (
    StudyDataExtractor,
    _retain_stage1_candidates,
)
from generation_pipeline.filters.replicability_filter import (
    ReplicabilityFilter,
    _normalize_experiments,
)
from generation_pipeline.stage1_study_contract import (
    apply_stage1_study_contract,
    audit_stage1_study_contract,
    normalize_stage1_semantic_fields,
)
from generation_pipeline.stage1_verifier import (
    _compact_stage1,
    _filter_policy_inconsistent_feedback,
    _filter_existing_missing_units,
    _filter_structured_boundary_corrections,
    _expand_field_correction,
    _filter_ungrounded_group_corrections,
    _is_grounded_missing_unit,
    _normalize_study_audit_report,
    _normalize_window_report,
    _numeric_consistency_challenges,
    _validate_study_audit_payload,
    build_study_verifier_prompt,
    build_verifier_prompt,
)
from generation_pipeline.pipeline import (
    _apply_stage1_eligibility_corrections,
    _apply_stage1_field_corrections,
    _stage1_has_targeted_feedback,
    _stage1_refinement_improves,
    _stage1_result_score,
)
from generation_pipeline.stage1_compiler import _apply_simulation_barrier_gate


def _experiment(study_id: str, replicable: str, reasons=None):
    return {
        "experiment_id": study_id.replace("_", " ").title(),
        "study_id": study_id,
        "experiment_name": f"Decision task for {study_id}",
        "study_name": f"Decision task for {study_id}",
        "design_type": "between-subjects",
        "conditions_or_factors": ["frame: gain vs loss"],
        "material_variants": [],
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
        "unit_provenance": "current_paper",
        "is_distinct_empirical_unit": True,
        "unit_provenance_evidence": "The paper reports this sample and task.",
        "empirical_support": {
            "own_sample_or_assignment": "yes",
            "participant_facing_task": "yes",
            "quantitative_result": "yes",
        },
        "simulation_barriers": [],
        "evidence_refs": ["p001_text"],
    }


class Stage1ScopeTests(unittest.TestCase):
    def test_refinement_score_rejects_more_grounded_issues(self):
        def result(issue_count):
            return {
                "stage1_evidence": {
                    "extraction_complete": True,
                    "rejected_comparison_relation_count": 0,
                },
                "stage1_study_contract": {"blocking_issue_count": 0},
                "stage1_quality": {"issues": []},
                "stage1_verification": {
                    "status": "ok",
                    "overall": "needs_review" if issue_count else "pass",
                    "regeneration_instructions": {
                        "missing_studies": [f"missing-{index}" for index in range(issue_count)],
                        "split_merge_corrections": [],
                        "comparison_group_corrections": [],
                        "study_field_corrections": [],
                        "eligibility_corrections": [],
                    },
                },
            }

        self.assertLess(_stage1_result_score(result(1)), _stage1_result_score(result(3)))

    def test_refinement_requires_material_improvement_on_large_feedback_sets(self):
        self.assertFalse(
            _stage1_refinement_improves((0, 0, 0, 0, 35), (0, 0, 0, 0, 34))
        )
        self.assertTrue(
            _stage1_refinement_improves((0, 0, 0, 0, 10), (0, 0, 0, 0, 8))
        )
        self.assertTrue(
            _stage1_refinement_improves((0, 0, 0, 0, 3), (0, 0, 0, 0, 2))
        )

    def test_targeted_field_repair_preserves_inventory_and_extends_provenance(self):
        original = {
            "experiments": [_experiment("study_1", "YES")],
            "comparison_groups": [{"comparison_group_id": "group_1"}],
            "stage1_verification": {"overall": "needs_review"},
            "stage1_quality": {"issues": []},
        }
        candidate, applied = _apply_stage1_field_corrections(
            original,
            [
                {
                    "study": "study_1",
                    "field": "input",
                    "expected_value": "Participants read the complete corrected problem.",
                    "correction_basis": "direct_contradiction",
                    "evidence_block_ids": ["p002_text"],
                }
            ],
        )

        self.assertEqual(original["experiments"][0]["input"], "Participants read a framed choice problem.")
        self.assertEqual(
            candidate["experiments"][0]["input"],
            "Participants read the complete corrected problem.",
        )
        self.assertEqual(candidate["comparison_groups"], original["comparison_groups"])
        self.assertIn("p002_text", candidate["experiments"][0]["evidence_refs"])
        self.assertEqual(
            candidate["experiments"][0]["field_evidence"]["input"],
            ["p002_text"],
        )
        self.assertNotIn("stage1_verification", candidate)
        self.assertNotIn("stage1_quality", candidate)
        self.assertEqual(applied[0]["field"], "input")

    def test_targeted_repair_handles_fields_and_eligibility_without_boundaries(self):
        field_feedback = {
            "missing_studies": [],
            "split_merge_corrections": [],
            "comparison_group_corrections": [],
            "study_field_corrections": [{"study": "study_1", "field": "input"}],
            "eligibility_corrections": [],
        }
        self.assertTrue(_stage1_has_targeted_feedback(field_feedback))
        field_feedback["missing_studies"] = [{"study": "Study 2"}]
        self.assertFalse(_stage1_has_targeted_feedback(field_feedback))
        field_feedback["missing_studies"] = []
        field_feedback["eligibility_corrections"] = [{"study": "study_1"}]
        self.assertTrue(_stage1_has_targeted_feedback(field_feedback))

    def test_targeted_eligibility_repair_updates_label_and_provenance(self):
        original = {
            "experiments": [_experiment("study_1", "NO")],
            "stage1_verification": {"overall": "needs_review"},
            "stage1_quality": {"issues": []},
        }
        candidate, applied = _apply_stage1_eligibility_corrections(
            original,
            [
                {
                    "study": "study_1",
                    "expected_label": "YES",
                    "correction_basis": "policy_misclassification",
                    "evidence_block_ids": ["p002_text"],
                }
            ],
        )

        self.assertEqual(original["experiments"][0]["replicable"], "NO")
        self.assertEqual(candidate["experiments"][0]["replicable"], "YES")
        self.assertEqual(
            candidate["experiments"][0]["field_evidence"]["replicable"],
            ["p002_text"],
        )
        self.assertEqual(applied[0]["field"], "replicable")

    def test_stage1_prompt_is_topic_independent(self):
        prompt = ReplicabilityFilter(None)._build_prompt("paper.pdf", 12)
        lowered = prompt.lower()
        self.assertIn("topic is unrestricted", lowered)
        self.assertIn("behavioral economics", lowered)
        self.assertNotIn("moral / ethical", lowered)
        self.assertNotIn("outcome not moral", lowered)

    def test_verifier_uses_same_general_scope(self):
        boundary_prompt = build_verifier_prompt({"experiments": []}, "paper text")
        study_prompt = build_study_verifier_prompt(
            _experiment("study_1", "YES"),
            "paper evidence",
            valid_block_ids=["p001_text"],
            audited_evidence_refs=["p001_text"],
            numeric_challenges=[],
        )
        self.assertIn("social-science paper", boundary_prompt)
        self.assertNotIn("moral / ethical", boundary_prompt.lower())
        self.assertIn("topic-independent", study_prompt)
        self.assertIn("scientific topic or discipline", study_prompt.lower())

    def test_study_audit_rejects_prose_instead_of_typed_field_replacement(self):
        experiment = _experiment("study_1", "YES")
        report = {
            "study_id": "study_1",
            "confidence": 0.9,
            "study_field_corrections": [
                {
                    "field": "material_variants",
                    "expected_value": "remove unsupported variant details",
                    "correction_basis": "direct_contradiction",
                    "reason": "The result table differs.",
                    "evidence": "Direct table evidence.",
                    "evidence_block_ids": ["p001_text"],
                }
            ],
            "eligibility_corrections": [],
        }
        with self.assertRaisesRegex(ValueError, "no field with a usable typed replacement"):
            _validate_study_audit_payload(
                report,
                experiment=experiment,
                valid_block_ids={"p001_text"},
                numeric_challenges=[],
            )

    def test_study_audit_normalization_preserves_valid_sibling_and_records_bad_item(self):
        experiment = _experiment("study_1", "YES")
        report = {
            "study_id": "study_1",
            "confidence": 0.9,
            "study_field_corrections": [
                {
                    "field": "input",
                    "expected_value": "Participants read the corrected source problem.",
                    "correction_basis": "direct_contradiction",
                    "reason": "The source wording differs.",
                    "evidence": "Direct source wording.",
                    "evidence_block_ids": ["p001_text"],
                },
                {
                    "field": "material_variants",
                    "expected_value": "remove unsupported details",
                    "correction_basis": "direct_contradiction",
                    "reason": "The variants are unsupported.",
                    "evidence": "Direct source wording.",
                    "evidence_block_ids": ["p001_text"],
                },
            ],
            "eligibility_corrections": [],
            "numeric_challenge_results": [],
        }

        normalized = _normalize_study_audit_report(
            report,
            experiment=experiment,
            valid_block_ids={"p001_text"},
            numeric_challenges=[],
        )

        self.assertEqual(
            [item["field"] for item in normalized["study_field_corrections"]],
            ["input"],
        )
        self.assertEqual(len(normalized["validation_diagnostics"]), 1)
        self.assertIn(
            "no usable typed replacement",
            normalized["validation_diagnostics"][0]["error"],
        )

    def test_boundary_normalization_drops_duplicate_group_as_noop(self):
        first = _experiment("study_1", "YES")
        second = _experiment("study_2", "YES")
        stage1 = {
            "experiments": [first, second],
            "comparison_groups": [
                {
                    "comparison_group_id": "group_1",
                    "member_study_ids": ["study_1", "study_2"],
                }
            ],
        }
        report = {
            "confidence": 0.8,
            "missing_studies": [],
            "split_merge_corrections": [],
            "comparison_group_corrections": [
                {
                    "member_study_ids": ["study_1", "study_2"],
                    "verdict": "missing_group",
                    "current_group_id": None,
                    "expected_member_study_ids": [],
                    "relationship_kind": "paired_contrast",
                    "evidence_kind": "explicit_cross_unit_contrast",
                    "comparison_target": "choice",
                    "reason": "The units are compared.",
                    "evidence": "Direct contrast.",
                    "evidence_block_ids": ["p001_text"],
                }
            ],
        }

        normalized = _normalize_window_report(
            report,
            valid_block_ids={"p001_text"},
            stage1_json=stage1,
        )

        self.assertEqual(normalized["comparison_group_corrections"], [])
        self.assertEqual(normalized["validation_diagnostics"], [])
        self.assertEqual(normalized["dropped_noop_corrections"], 1)

    def test_boundary_normalization_keeps_malformed_correction_as_diagnostic(self):
        stage1 = {"experiments": [_experiment("study_1", "YES")], "comparison_groups": []}
        report = {
            "confidence": 0.8,
            "missing_studies": [],
            "split_merge_corrections": [
                {
                    "verdict": "split_issue",
                    "current_study_ids": ["study_1"],
                    "proposed_study_ids": ["study_1a", "study_1b"],
                    "proposed_source_labels": ["Study 1a"],
                    "boundary_basis": "distinct_source_labels",
                    "reason": "Two labels are visible.",
                    "evidence": "Source headings.",
                    "evidence_block_ids": ["p001_text"],
                }
            ],
            "comparison_group_corrections": [],
        }

        normalized = _normalize_window_report(
            report,
            valid_block_ids={"p001_text"},
            stage1_json=stage1,
        )

        self.assertEqual(normalized["split_merge_corrections"], [])
        self.assertEqual(len(normalized["validation_diagnostics"]), 1)
        self.assertIn("one source label", normalized["validation_diagnostics"][0]["error"])

    def test_composite_field_correction_is_split_by_typed_values(self):
        corrections = _expand_field_correction(
            {
                "field": "conditions_or_factors|input",
                "expected_value": {
                    "conditions_or_factors": ["frame: gain vs loss"],
                    "input": "Participants read one framed problem.",
                },
            }
        )
        self.assertEqual(
            [(item["field"], item["expected_value"]) for item in corrections],
            [
                ("conditions_or_factors", ["frame: gain vs loss"]),
                ("input", "Participants read one framed problem."),
            ],
        )

    def test_numeric_consistency_challenge_surfaces_power_of_ten_ocr_conflict(self):
        experiment = _experiment("study_1", "YES")
        experiment["conditions_or_factors"] = ["sure $30 vs 8% chance to win $45"]
        experiment["input"] = "Choose a sure $30 or an 8 percent chance to win $45."
        block = DocumentBlock(
            block_id="p002_text",
            order=1,
            page_start=2,
            page_end=2,
            block_type="text",
            text="The second prospect has probability .25 x .80 = .20 to win $45.",
        )

        challenges = _numeric_consistency_challenges(experiment, [block])

        self.assertEqual(len(challenges), 1)
        self.assertEqual(challenges[0]["current_percent_value"], 8.0)
        self.assertEqual(challenges[0]["equation_percent_value"], 80.0)
        self.assertEqual(
            challenges[0]["current_fields"],
            ["conditions_or_factors", "input"],
        )
        self.assertEqual(challenges[0]["equation_evidence_block_id"], "p002_text")

    def test_design_type_explanations_normalize_to_controlled_enum(self):
        corrections = _expand_field_correction(
            {
                "field": "design_type",
                "expected_value": "mixed (between- and within-subjects)",
            }
        )
        self.assertEqual(corrections[0]["expected_value"], "mixed")

    def test_numeric_correction_must_cover_every_affected_field(self):
        experiment = _experiment("study_1", "YES")
        experiment["conditions_or_factors"] = ["sure $30 vs 8% chance to win $45"]
        experiment["input"] = "Choose a sure $30 or an 8% chance to win $45."
        challenges = _numeric_consistency_challenges(
            experiment,
            [
                DocumentBlock(
                    block_id="p002_text",
                    order=1,
                    page_start=2,
                    page_end=2,
                    block_type="text",
                    text="The prospect has probability .25 x .80 = .20 to win $45.",
                )
            ],
        )
        report = {
            "study_id": "study_1",
            "confidence": 0.9,
            "study_field_corrections": [
                {
                    "field": "conditions_or_factors",
                    "expected_value": ["sure $30 vs 80% chance to win $45"],
                    "correction_basis": "direct_contradiction",
                    "reason": "The equation establishes the conditional probability.",
                    "evidence": ".25 x .80 = .20",
                    "evidence_block_ids": ["p002_text"],
                }
            ],
            "eligibility_corrections": [],
            "numeric_challenge_results": [
                {
                    "challenge_id": challenges[0]["challenge_id"],
                    "verdict": "correction_required",
                    "reason": "The nearby 8% token lost a zero in OCR.",
                    "evidence_block_ids": ["p002_text"],
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "requires corrections for fields: input"):
            _validate_study_audit_payload(
                report,
                experiment=experiment,
                valid_block_ids={"p001_text", "p002_text"},
                numeric_challenges=challenges,
            )

        report["study_field_corrections"].append(
            {
                "field": "input",
                "expected_value": "Choose a sure $30 or an 80% chance to win $45.",
                "correction_basis": "direct_contradiction",
                "reason": "The equation establishes the conditional probability.",
                "evidence": ".25 x .80 = .20",
                "evidence_block_ids": ["p002_text"],
            }
        )
        _validate_study_audit_payload(
            report,
            experiment=experiment,
            valid_block_ids={"p001_text", "p002_text"},
            numeric_challenges=challenges,
        )

    def test_primary_target_execution_barrier_forces_no_without_keyword_rules(self):
        experiment = _experiment("field_action", "YES")
        experiment["simulation_barriers"] = [
            {
                "kind": "physical_action",
                "description": "The primary outcome is an observed real-world action.",
                "affects_primary_target": True,
                "evidence_refs": ["p001_text"],
            }
        ]
        _apply_simulation_barrier_gate(experiment)
        self.assertEqual(experiment["replicable"], "NO")
        self.assertIn("unsupported execution mode", experiment["exclusion_reasons"][0])

    def test_non_primary_barrier_does_not_exclude_static_response_target(self):
        experiment = _experiment("survey_after_field_context", "YES")
        experiment["simulation_barriers"] = [
            {
                "kind": "physical_action",
                "description": "Background context includes an action not used as the target.",
                "affects_primary_target": False,
                "evidence_refs": ["p001_text"],
            }
        ]
        _apply_simulation_barrier_gate(experiment)
        self.assertEqual(experiment["replicable"], "YES")
        self.assertEqual(experiment["exclusion_reasons"], [])

    def test_consequential_commitment_is_a_valid_primary_target_barrier(self):
        experiment = _experiment("committed_choice", "YES")
        experiment["simulation_barriers"] = [
            {
                "kind": "consequential_commitment",
                "description": "The recorded choice created enforced real-world follow-through.",
                "affects_primary_target": True,
                "evidence_refs": ["p001_text"],
            }
        ]
        _apply_simulation_barrier_gate(experiment)
        contract = audit_stage1_study_contract({"experiments": [experiment]})
        self.assertEqual(experiment["replicable"], "NO")
        self.assertNotIn(
            "invalid_simulation_barrier_kind",
            contract["studies"]["committed_choice"]["blocking_issues"],
        )

    def test_verifier_drops_policy_conflict_and_already_covered_variants(self):
        experiment = _experiment("field_action", "NO", ["unsupported execution"])
        experiment["material_variants"] = [
            {
                "variant_id": "message_a",
                "label": "Message A (display sign)",
                "role": "stimulus",
                "assignment": "One of two display messages used in the study",
                "evidence_refs": ["p001_text"],
            }
        ]
        experiment["simulation_barriers"] = [
            {
                "kind": "physical_action",
                "description": "Observed action is the primary outcome.",
                "affects_primary_target": True,
                "evidence_refs": ["p001_text"],
            }
        ]
        fields, eligibility = _filter_policy_inconsistent_feedback(
            {"experiments": [experiment]},
            [
                {
                    "study": "field_action",
                    "field": "simulation_barriers",
                    "expected_value": ["physical action is required"],
                },
                {
                    "study": "field_action",
                    "field": "material_variants",
                    "expected_value": [
                        {
                            "label": "Message A",
                            "role": "stimulus",
                            "assignment": "one of two display messages used in study",
                        }
                    ],
                },
                {
                    "study": "field_action",
                    "field": "design_type",
                    "expected_value": "field",
                },
                {
                    "study": "field_action",
                    "field": "participant_task",
                    "expected_value": "Replace the observed action with a questionnaire proxy.",
                },
                {
                    "study": "field_action",
                    "field": "other",
                    "expected_value": {"directly_evidenced_in_window": True},
                },
            ],
            [
                {
                    "study": "field_action",
                    "expected_label": "YES",
                    "reason": "has a quantitative result",
                }
            ],
        )
        self.assertEqual([item["field"] for item in fields], ["design_type"])
        self.assertEqual(eligibility, [])

    def test_spurious_group_requires_evidence_from_the_group_context(self):
        stage1 = {
            "comparison_groups": [
                {
                    "comparison_group_id": "group_1",
                    "member_study_ids": ["study_1", "study_2"],
                    "evidence_refs": ["p010_text"],
                }
            ]
        }
        corrections = [
            {
                "member_study_ids": ["study_1", "study_2"],
                "verdict": "spurious_group",
                "current_group_id": "group_1",
                "evidence_kind": "explicit_cross_unit_contrast",
                "evidence_block_ids": ["p002_text"],
            },
            {
                "member_study_ids": ["study_1", "study_2"],
                "verdict": "wrong_members",
                "current_group_id": "group_1",
                "expected_member_study_ids": ["study_1", "study_3"],
                "evidence_kind": "explicit_cross_unit_contrast",
                "evidence_block_ids": ["p010_text"],
            },
            {
                "member_study_ids": ["study_2", "study_3"],
                "verdict": "missing_group",
                "current_group_id": None,
                "evidence_kind": "narrative_synthesis",
                "evidence_block_ids": ["p020_text"],
            },
        ]
        filtered = _filter_ungrounded_group_corrections(stage1, corrections)
        self.assertEqual(filtered, [corrections[1]])

    def test_boundary_filters_reject_existing_units_and_field_like_split_feedback(self):
        stage1 = {"experiments": [_experiment("study_1", "YES")]}
        stage1["experiments"][0]["material_variants"] = [
            {
                "variant_id": "story_a",
                "label": "Story A",
                "role": "stimulus",
                "evidence_refs": ["p001_text"],
            },
            {
                "variant_id": "story_b",
                "label": "Story B",
                "role": "stimulus",
                "evidence_refs": ["p001_text"],
            },
        ]
        missing = _filter_existing_missing_units(
            stage1,
            [
                {"proposed_study_id": "study_1", "study": "Study 1 again"},
                {"proposed_study_id": "study_2", "study": "Study 2"},
            ],
        )
        self.assertEqual([item["proposed_study_id"] for item in missing], ["study_2"])

        valid_split = {
            "verdict": "split_issue",
            "current_study_ids": ["study_1"],
            "proposed_study_ids": ["study_1a", "study_1b"],
            "proposed_source_labels": ["Study 1a", "Study 1b"],
            "boundary_basis": "distinct_source_labels",
        }
        corrections = _filter_structured_boundary_corrections(
            stage1,
            [
                {
                    "verdict": "split_issue",
                    "study": "study_1",
                    "reason": "Add missing material variants.",
                },
                {
                    "verdict": "split_issue",
                    "current_study_ids": ["study_1"],
                    "proposed_study_ids": ["study_1_story_a", "study_1_story_b"],
                    "proposed_source_labels": ["Story A", "Story B"],
                    "boundary_basis": "independent_recruitment_or_session",
                },
                valid_split,
            ],
        )
        self.assertEqual(corrections, [valid_split])

    def test_empty_barrier_dispute_does_not_unlock_related_policy_changes(self):
        experiment = _experiment("consequential_choice", "NO", ["real consequence"])
        experiment["simulation_barriers"] = [
            {
                "kind": "physical_action",
                "description": "The choice commits the participant to an action.",
                "affects_primary_target": True,
                "evidence_refs": ["p001_text"],
            }
        ]
        barrier_correction = {
            "study": "consequential_choice",
            "field": "simulation_barriers",
            "expected_value": [],
        }
        fields, eligibility = _filter_policy_inconsistent_feedback(
            {"experiments": [experiment]},
            [
                barrier_correction,
                {
                    "study": "consequential_choice",
                    "field": "exclusion_reasons",
                    "expected_value": [],
                },
                {
                    "study": "consequential_choice",
                    "field": "participant_task",
                    "expected_value": "Treat it as a hypothetical choice.",
                },
            ],
            [
                {
                    "study": "consequential_choice",
                    "expected_label": "YES",
                }
            ],
        )
        self.assertEqual(fields, [barrier_correction])
        self.assertEqual(eligibility, [])

    def test_boundary_filter_never_merges_distinct_formal_source_labels(self):
        problem_3 = _experiment("study_problem_3", "YES")
        problem_3["experiment_id"] = "Problem 3"
        problem_4 = _experiment("study_problem_4", "YES")
        problem_4["experiment_id"] = "Problem 4"
        correction = {
            "verdict": "merge_issue",
            "current_study_ids": ["study_problem_3", "study_problem_4"],
            "proposed_study_ids": ["study_problem_3_combined"],
            "proposed_source_labels": ["Problem 3 combined presentation"],
            "boundary_basis": "same_source_unit",
        }

        self.assertEqual(
            _filter_structured_boundary_corrections(
                {"experiments": [problem_3, problem_4]},
                [correction],
            ),
            [],
        )

    def test_boundary_filter_protects_roman_numbered_experiments(self):
        experiment_i = _experiment("study_experiment_i", "YES")
        experiment_i["experiment_id"] = "Experiment I"
        experiment_ii = _experiment("study_experiment_ii", "YES")
        experiment_ii["experiment_id"] = "Experiment II"
        merge = {
            "verdict": "merge_issue",
            "current_study_ids": ["study_experiment_i", "study_experiment_ii"],
            "proposed_study_ids": ["study_experiment_i_ii"],
            "proposed_source_labels": ["Experiments I and II"],
            "boundary_basis": "same_source_unit",
        }
        split = {
            "verdict": "split_issue",
            "current_study_ids": ["study_experiment_i"],
            "proposed_study_ids": ["study_experiment_i_a", "study_experiment_i_b"],
            "proposed_source_labels": [
                "Experiment I (condition A)",
                "Experiment I (condition B)",
            ],
            "boundary_basis": "independent_recruitment_or_session",
        }

        self.assertEqual(
            _filter_structured_boundary_corrections(
                {"experiments": [experiment_i, experiment_ii]},
                [merge, split],
            ),
            [],
        )

    def test_structured_text_replacement_is_preserved_as_schema_valid_text(self):
        experiment = _experiment("study_1", "YES")
        report = _normalize_study_audit_report(
            {
                "study_id": "study_1",
                "confidence": 0.9,
                "study_field_corrections": [
                    {
                        "field": "participants",
                        "expected_value": {"condition_a": 23, "condition_b": 21},
                        "correction_basis": "source_missing_content",
                        "reason": "The table reports both samples.",
                        "evidence": "N values are listed.",
                        "evidence_block_ids": ["p001_text"],
                    }
                ],
                "eligibility_corrections": [],
                "numeric_challenge_results": [],
            },
            experiment=experiment,
            valid_block_ids={"p001_text"},
            numeric_challenges=[],
        )

        self.assertEqual(report["validation_diagnostics"], [])
        self.assertEqual(
            report["study_field_corrections"][0]["expected_value"],
            '{"condition_a": 23, "condition_b": 21}',
        )

    def test_structured_condition_replacement_becomes_schema_valid_string_list(self):
        experiment = _experiment("study_1", "YES")
        report = _normalize_study_audit_report(
            {
                "study_id": "study_1",
                "confidence": 0.9,
                "study_field_corrections": [
                    {
                        "field": "conditions_or_factors",
                        "expected_value": [
                            {"name": "Series", "levels": ["A", "B", "C"]}
                        ],
                        "correction_basis": "source_missing_content",
                        "reason": "The source reports three assigned series.",
                        "evidence": "Series A, B, and C were assigned.",
                        "evidence_block_ids": ["p001_text"],
                    }
                ],
                "eligibility_corrections": [],
                "numeric_challenge_results": [],
            },
            experiment=experiment,
            valid_block_ids={"p001_text"},
            numeric_challenges=[],
        )

        self.assertEqual(report["validation_diagnostics"], [])
        self.assertEqual(
            report["study_field_corrections"][0]["expected_value"],
            ['{"levels": ["A", "B", "C"], "name": "Series"}'],
        )

    def test_missing_subtask_with_existing_formal_label_is_not_a_new_study(self):
        experiment = _experiment("study_experiment_ixa", "YES")
        experiment["experiment_id"] = "Experiment IXa"
        items = [
            {
                "study": "Experiment IXa — written reasons after the checklist",
                "proposed_study_id": "study_experiment_ixa_written_reasons",
                "evidence_block_ids": ["p020_text"],
            }
        ]

        self.assertEqual(
            _filter_existing_missing_units({"experiments": [experiment]}, items),
            [],
        )

    def test_merged_task_components_are_not_reported_as_missing_studies(self):
        experiment = _experiment("study_problem_1", "YES")
        experiment.update(
            {
                "experiment_id": "Problem 1 + Problem 2",
                "candidate_components": [
                    {
                        "study_id": "study_problem_1",
                        "reported_label": "Problem 1",
                        "study_name": "Gain frame",
                        "evidence_block_ids": ["p001_text"],
                    },
                    {
                        "study_id": "study_problem_2",
                        "reported_label": "Problem 2",
                        "study_name": "Loss frame",
                        "evidence_block_ids": ["p002_text"],
                    },
                ],
                "source_task_family_relation_ids": ["relation_1_2"],
            }
        )
        stage1 = {"experiments": [experiment]}

        filtered = _filter_existing_missing_units(
            stage1,
            [
                {
                    "study": "Problem 2: Loss frame",
                    "proposed_study_id": "study_problem_2",
                    "evidence_block_ids": ["p002_text"],
                },
                {
                    "study": "Problem 11: New task",
                    "proposed_study_id": "study_problem_11",
                    "evidence_block_ids": ["p011_text"],
                },
            ],
        )

        self.assertEqual(
            [item["proposed_study_id"] for item in filtered],
            ["study_problem_11"],
        )
        compact = _compact_stage1(stage1)["empirical_units"][0]
        self.assertEqual(
            [item["reported_label"] for item in compact["candidate_components"]],
            ["Problem 1", "Problem 2"],
        )
        self.assertEqual(
            compact["source_task_family_relation_ids"],
            ["relation_1_2"],
        )

    def test_measurement_steps_inside_formal_parent_are_not_missing_studies(self):
        experiment = _experiment("study_1", "YES")
        experiment["experiment_id"] = "Study 1"
        experiment["evidence_refs"] = ["choice", "estimate", "traits"]

        filtered = _filter_existing_missing_units(
            {"experiments": [experiment]},
            [
                {
                    "study": "Percentage-estimation task inside Study 1",
                    "proposed_study_id": "study_1_percentage_estimation",
                    "evidence_block_ids": ["estimate"],
                },
                {
                    "study": "Trait-rating task (Study I)",
                    "proposed_study_id": "study_1_trait_rating",
                    "evidence_block_ids": ["traits"],
                },
                {
                    "study": "Unlabeled independent sorting task",
                    "proposed_study_id": "sorting_task",
                    "evidence_block_ids": ["sorting"],
                },
            ],
        )

        self.assertEqual(
            [item["proposed_study_id"] for item in filtered],
            ["sorting_task"],
        )

    def test_noop_eligibility_correction_is_dropped(self):
        experiment = _experiment("study_1", "YES")
        fields, eligibility = _filter_policy_inconsistent_feedback(
            {"experiments": [experiment]},
            [],
            [{"study": "study_1", "expected_label": "YES"}],
        )
        self.assertEqual(fields, [])
        self.assertEqual(eligibility, [])

    def test_analysis_exclusion_is_not_a_stage1_exclusion_reason_for_yes_study(self):
        experiment = _experiment("study_1", "YES")
        correction = {
            "study": "study_1",
            "field": "exclusion_reasons",
            "expected_value": ["Respondents with incomplete answers were omitted from analysis."],
        }

        fields, eligibility = _filter_policy_inconsistent_feedback(
            {"experiments": [experiment]},
            [correction],
            [],
        )

        self.assertEqual(fields, [])
        self.assertEqual(eligibility, [])

    def test_exclusion_reason_is_retained_with_same_study_eligibility_downgrade(self):
        experiment = _experiment("study_1", "YES")
        correction = {
            "study": "study_1",
            "field": "exclusion_reasons",
            "expected_value": ["The target requires a live contingent interaction."],
        }
        downgrade = {
            "study": "study_1",
            "expected_label": "NO",
        }

        fields, eligibility = _filter_policy_inconsistent_feedback(
            {"experiments": [experiment]},
            [correction],
            [downgrade],
        )

        self.assertEqual(fields, [correction])
        self.assertEqual(eligibility, [downgrade])

    def test_eligibility_cannot_be_relaxed_without_quantitative_support(self):
        experiment = _experiment("study_1", "NO")
        experiment["empirical_support"]["quantitative_result"] = "no"

        fields, eligibility = _filter_policy_inconsistent_feedback(
            {"experiments": [experiment]},
            [],
            [
                {
                    "study": "study_1",
                    "expected_label": "UNCERTAIN",
                    "reason": "The exact target statistic is unavailable.",
                }
            ],
        )

        self.assertEqual(fields, [])
        self.assertEqual(eligibility, [])

    def test_qualitative_majority_cannot_unlock_quantitative_support(self):
        experiment = _experiment("study_1", "NO", ["No exact quantitative result."])
        experiment["output"] = "A significant majority selected the mixed sample."
        experiment["empirical_support"]["quantitative_result"] = "no"
        support_repair = {
            "study": "study_1",
            "field": "empirical_support",
            "expected_value": {
                "own_sample_or_assignment": "yes",
                "participant_facing_task": "yes",
                "quantitative_result": "yes",
            },
        }
        exclusion_repair = {
            "study": "study_1",
            "field": "exclusion_reasons",
            "expected_value": [],
        }

        fields, eligibility = _filter_policy_inconsistent_feedback(
            {"experiments": [experiment]},
            [support_repair, exclusion_repair],
            [{"study": "study_1", "expected_label": "YES"}],
        )

        self.assertEqual(fields, [])
        self.assertEqual(eligibility, [])

    def test_exact_numeric_output_repair_can_unlock_quantitative_support(self):
        experiment = _experiment("study_1", "NO", ["No exact quantitative result."])
        experiment["output"] = "A majority selected option A."
        experiment["empirical_support"]["quantitative_result"] = "no"
        output_repair = {
            "study": "study_1",
            "field": "output",
            "expected_value": "67 of 89 respondents selected option A.",
        }
        support_repair = {
            "study": "study_1",
            "field": "empirical_support",
            "expected_value": {
                "own_sample_or_assignment": "yes",
                "participant_facing_task": "yes",
                "quantitative_result": "yes",
            },
        }
        exclusion_repair = {
            "study": "study_1",
            "field": "exclusion_reasons",
            "expected_value": [],
        }
        eligibility_repair = {"study": "study_1", "expected_label": "YES"}

        fields, eligibility = _filter_policy_inconsistent_feedback(
            {"experiments": [experiment]},
            [output_repair, support_repair, exclusion_repair],
            [eligibility_repair],
        )

        self.assertEqual(fields, [output_repair, support_repair, exclusion_repair])
        self.assertEqual(eligibility, [eligibility_repair])

    def test_unlabeled_narrative_example_requires_exact_response_statistic(self):
        narrative = {
            "source_boundary": "unlabeled_new_collection",
            "current_paper_collection": True,
            "has_exact_response_statistic": False,
            "response_statistic": None,
        }
        self.assertFalse(_is_grounded_missing_unit(narrative))
        narrative["has_exact_response_statistic"] = True
        narrative["response_statistic"] = "63% selected option A"
        self.assertTrue(_is_grounded_missing_unit(narrative))
        self.assertTrue(
            _is_grounded_missing_unit(
                {
                    "source_boundary": "source_labeled",
                    "current_paper_collection": True,
                    "has_exact_response_statistic": False,
                    "response_statistic": None,
                }
            )
        )

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

    def test_contract_removes_inline_block_ids_but_preserves_provenance_arrays(self):
        experiment = _experiment("study_1", "YES")
        experiment["input"] = (
            "Choose option A [74 percent] after reading the scenario "
            "[p001_text, p002_text_00003]."
        )
        experiment["conditions_or_factors"] = [
            "frame: gain vs loss [p003_text_00004]"
        ]
        experiment["evidence_refs"] = ["p001_text", "p002_text_00003"]
        payload = {"experiments": [experiment]}

        removed = normalize_stage1_semantic_fields(payload)

        self.assertEqual(removed, 2)
        self.assertIn("[74 percent]", payload["experiments"][0]["input"])
        self.assertNotIn("p001_text", payload["experiments"][0]["input"])
        self.assertEqual(
            payload["experiments"][0]["conditions_or_factors"],
            ["frame: gain vs loss"],
        )
        self.assertEqual(
            payload["experiments"][0]["evidence_refs"],
            ["p001_text", "p002_text_00003"],
        )

    def test_contract_preserves_cumulative_normalization_audit(self):
        experiment = _experiment("study_1", "YES")
        experiment["input"] += " [p001_text]"
        payload = {"experiments": [experiment]}

        apply_stage1_study_contract(payload)
        apply_stage1_study_contract(payload)

        normalization = payload["stage1_study_contract"]["normalization"]
        self.assertEqual(normalization["inline_evidence_token_groups_removed"], 1)
        self.assertEqual(
            normalization["inline_evidence_token_groups_removed_this_pass"],
            0,
        )

    def test_contract_requires_reason_for_no_label(self):
        payload = {"experiments": [_experiment("field_outcome", "NO")]}
        contract = audit_stage1_study_contract(payload)
        self.assertIn(
            "excluded_without_reason",
            contract["studies"]["field_outcome"]["blocking_issues"],
        )

    def test_contract_rejects_external_or_non_distinct_inventory_entries(self):
        external = {
            **_experiment("external_study", "YES"),
            "unit_provenance": "cited_prior",
        }
        fragment = {
            **_experiment("sample_fragment", "YES"),
            "is_distinct_empirical_unit": False,
        }
        contract = audit_stage1_study_contract({"experiments": [external, fragment]})
        self.assertIn(
            "invalid_or_external_unit_provenance",
            contract["studies"]["external_study"]["blocking_issues"],
        )
        self.assertIn(
            "not_a_distinct_empirical_unit",
            contract["studies"]["sample_fragment"]["blocking_issues"],
        )

    def test_contract_validates_comparison_group_membership(self):
        payload = {
            "experiments": [
                _experiment("problem_1", "YES"),
                _experiment("problem_2", "YES"),
            ],
            "comparison_groups": [
                {
                    "comparison_group_id": "comparison_group_001",
                    "member_study_ids": ["problem_1", "missing_problem"],
                    "relationship_kind": "paired_contrast",
                    "comparison_target": "choice proportion",
                    "evidence_refs": ["p002_text"],
                }
            ],
        }
        contract = audit_stage1_study_contract(payload)
        group = contract["comparison_groups"]["comparison_group_001"]
        self.assertIn("comparison_group_has_unknown_members", group["blocking_issues"])
        self.assertGreater(contract["comparison_group_blocking_issue_count"], 0)

    def test_unclear_provenance_can_be_ineligible_without_contract_conflict(self):
        experiment = {
            **_experiment("qualitative_variant", "NO", ["no exact quantitative target"]),
            "unit_provenance": "unclear",
            "empirical_support": {
                "own_sample_or_assignment": "unclear",
                "participant_facing_task": "yes",
                "quantitative_result": "no",
            },
        }
        contract = audit_stage1_study_contract({"experiments": [experiment]})
        self.assertTrue(contract["studies"]["qualitative_variant"]["ready"])

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
