import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from generation_pipeline.pdf.evidence import PdfEvidenceIndex
from generation_pipeline.pdf.models import DocumentBlock, ParsedPdfDocument
from generation_pipeline.stage1_compiler import (
    BOUNDARY_ADJUDICATION_MAX_CONTEXT_CHARS,
    _apply_boundary_adjudication,
    _attach_discovered_shared_sample_contexts,
    _augment_relation_problem_family_merges,
    _build_bounded_candidate_ledger,
    _partition_empirical_units,
    _prune_invalid_study_evidence_refs,
    _feedback_for_study,
    _explicit_unit_labels,
    _normalize_material_variants,
    _normalize_mention,
    _reconcile_comparison_groups,
    _reconcile_mentions,
    _validate_boundary_adjudication_payload,
    DiscoveryWindow,
    build_discovery_windows,
    cached_json_call,
    compile_stage1_inventory,
)
from generation_pipeline.stage1_verifier import _inventory_for_window, verify_stage1_inventory


def _document(block_texts):
    return ParsedPdfDocument(
        source_file="paper.pdf",
        source_sha256="abc123",
        parser="test_parser",
        parser_version="1",
        page_count=len(block_texts),
        blocks=[
            DocumentBlock(
                block_id=f"p{index:03d}_text",
                order=index - 1,
                page_start=index,
                page_end=index,
                block_type="text",
                text=text,
                section_path=[f"Section {index}"],
            )
            for index, text in enumerate(block_texts, start=1)
        ],
    )


class _CompilerClient:
    model = "fake-model"

    def __init__(self):
        self.prompts = []

    def generate_content(self, prompt, **kwargs):
        del kwargs
        self.prompts.append(prompt)
        if "high-recall discovery pass" in prompt:
            valid_ids = json.loads(
                re.search(r"Valid evidence block IDs: (\[.*?\])", prompt).group(1)
            )
            mentions = []
            if "STUDY_ONE_MARKER" in prompt:
                mentions.append(
                    {
                        "reported_label": "Study 1",
                        "study_name": "Framed choice task",
                        "kind": "study",
                        "participant_task_hint": "Choose between two framed options",
                        "quantitative_target_hint": "Choice proportion",
                        "material_variants": [],
                        "evidence_block_ids": [valid_ids[0]],
                        "evidence_summary": "A distinct sample completed Study 1.",
                        "boundary_confidence": 0.95,
                    }
                )
            if "STUDY_TWO_MARKER" in prompt:
                mentions.append(
                    {
                        "reported_label": "Study 2",
                        "study_name": "Ranking task",
                        "kind": "study",
                        "participant_task_hint": "Rank alternatives",
                        "quantitative_target_hint": "Mean rank",
                        "material_variants": [],
                        "evidence_block_ids": [valid_ids[0]],
                        "evidence_summary": "A new sample completed Study 2.",
                        "boundary_confidence": 0.94,
                    }
                )
            return json.dumps(
                {
                    "paper_metadata": {
                        "paper_title": "General Social Science Paper"
                        if "TITLE_MARKER" in prompt
                        else None,
                        "paper_authors": ["A. Author"] if "TITLE_MARKER" in prompt else [],
                        "paper_abstract": "An abstract." if "TITLE_MARKER" in prompt else None,
                    },
                    "candidate_mentions": mentions,
                    "comparison_relations": [],
                }
            )
        if "boundary-adjudication pass" in prompt:
            return json.dumps(
                {
                    "merge_groups": [],
                    "reject_candidates": [],
                    "notes": "The source labels identify distinct units.",
                }
            )
        if "Extract exactly one empirical unit" in prompt:
            candidate = json.loads(
                prompt.split("Candidate boundary:\n", 1)[1].split(
                    "\n\nHumanStudy-Bench", 1
                )[0]
            )
            refs = json.loads(
                prompt.split("Evidence IDs must come from this valid list:\n", 1)[1].split(
                    "\n\nThe candidate may", 1
                )[0]
            )
            study_id = candidate["study_id"]
            return json.dumps(
                {
                    "experiment_id": candidate["reported_label"],
                    "study_id": study_id,
                    "experiment_name": candidate["study_name"],
                    "study_name": candidate["study_name"],
                    "design_type": "between-subjects",
                    "conditions_or_factors": ["condition: A vs B"],
                    "material_variants": [],
                    "input": "A source-grounded task",
                    "participant_task": candidate["participant_task_hint"],
                    "participants": "N = 100 adults",
                    "output": candidate["quantitative_target_hint"],
                    "candidate_source_hints": [
                        {
                            "kind": "paper",
                            "description": "Method section",
                            "expected_fields": ["instructions", "items", "options"],
                        }
                    ],
                    "replicable": "YES",
                    "has_self_contained_materials": False,
                    "exclusion_reasons": [],
                    "missing_materials": "Exact option wording",
                    "unit_provenance": "current_paper",
                    "is_distinct_empirical_unit": True,
                    "unit_provenance_evidence": "The bounded method block reports this sample and task.",
                    "empirical_support": {
                        "own_sample_or_assignment": "yes",
                        "participant_facing_task": "yes",
                        "quantitative_result": "yes",
                    },
                    "simulation_barriers": [],
                    "evidence_refs": refs[:1],
                    "field_evidence": {"participant_task": refs[:1]},
                    "extraction_confidence": 0.9,
                    "boundary_notes": "One participant sample.",
                }
            )
        if "independently auditing empirical-unit boundaries" in prompt:
            return json.dumps(
                {
                    "confidence": 0.9,
                    "missing_studies": [],
                    "split_merge_corrections": [],
                    "comparison_group_corrections": [],
                    "notes": "No boundary issue in this window.",
                }
            )
        if "independently auditing one empirical unit" in prompt:
            study_id = re.search(r"Study ID: ([^\n]+)", prompt).group(1).strip()
            return json.dumps(
                {
                    "study_id": study_id,
                    "confidence": 0.9,
                    "study_field_corrections": [],
                    "eligibility_corrections": [],
                    "notes": "No field issue in the complete study evidence.",
                }
            )
        raise AssertionError(f"Unexpected prompt: {prompt[:120]}")


class Stage1CompilerTests(unittest.TestCase):
    def test_material_variants_exclude_joint_options_and_keep_alternative_forms(self):
        variants = _normalize_material_variants(
            [
                {
                    "variant_id": "option_a",
                    "label": "Option A shown with Option B",
                    "role": "stimulus",
                    "is_alternative_version": False,
                    "evidence_refs": ["p001"],
                },
                {
                    "variant_id": "form_one",
                    "label": "Questionnaire form one",
                    "role": "form",
                    "is_alternative_version": True,
                    "evidence_refs": ["p002"],
                },
            ],
            valid_refs={"p001", "p002"},
        )
        self.assertEqual([item["variant_id"] for item in variants], ["form_one"])
        self.assertTrue(variants[0]["is_alternative_version"])

    def test_window_inventory_includes_out_of_window_comparison_members(self):
        stage1 = {
            "experiments": [
                {"study_id": "study_1", "experiment_id": "Study 1", "evidence_refs": ["p001"]},
                {"study_id": "study_2", "experiment_id": "Study 2", "evidence_refs": ["p002"]},
                {"study_id": "study_3", "experiment_id": "Study 3", "evidence_refs": ["p003"]},
            ],
            "comparison_groups": [
                {
                    "comparison_group_id": "group_1",
                    "member_study_ids": ["study_1", "study_2"],
                    "evidence_refs": ["p002"],
                }
            ],
        }
        window = DiscoveryWindow(
            window_id="window_002",
            text="comparison evidence",
            block_ids=["p002"],
            pages=[2],
            char_count=19,
        )
        local = _inventory_for_window(stage1, window)
        by_id = {item["study_id"]: item for item in local["experiments"]}
        self.assertEqual(set(by_id), {"study_1", "study_2", "study_3"})
        self.assertFalse(by_id["study_1"]["directly_evidenced_in_window"])
        self.assertEqual(by_id["study_1"]["evidence_refs"], [])
        self.assertTrue(by_id["study_2"]["directly_evidenced_in_window"])
        self.assertFalse(by_id["study_3"]["directly_evidenced_in_window"])

    def test_cache_keeps_multiple_prompt_variants_instead_of_overwriting(self):
        class Client:
            model = "fake"

            def __init__(self):
                self.calls = 0

            def generate_content(self, prompt, **kwargs):
                del kwargs
                self.calls += 1
                return json.dumps({"prompt": prompt})

        client = Client()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "window_001.json"
            first = cached_json_call(
                client,
                "prompt-a",
                cache_path=cache_path,
                prompt_version="test-v1",
                timeout=1,
                max_tokens=100,
                force=False,
            )
            cached_json_call(
                client,
                "prompt-b",
                cache_path=cache_path,
                prompt_version="test-v1",
                timeout=1,
                max_tokens=100,
                force=False,
            )
            repeated = cached_json_call(
                client,
                "prompt-a",
                cache_path=cache_path,
                prompt_version="test-v1",
                timeout=1,
                max_tokens=100,
                force=False,
            )
            cache_files = list(Path(tmpdir).glob("window_001.*.json"))
        self.assertEqual(first, repeated)
        self.assertEqual(client.calls, 2)
        self.assertEqual(len(cache_files), 2)

    def test_invalid_llm_payload_is_persisted_for_audit(self):
        class Client:
            model = "fake"

            def generate_content(self, prompt, **kwargs):
                del prompt, kwargs
                return json.dumps({"invalid": True})

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "candidate_ledger.json"
            with self.assertRaisesRegex(ValueError, "expected payload"):
                cached_json_call(
                    Client(),
                    "prompt",
                    cache_path=cache_path,
                    prompt_version="test-invalid-v1",
                    timeout=1,
                    max_tokens=100,
                    force=False,
                    max_attempts=1,
                    validator=lambda value: (_ for _ in ()).throw(
                        ValueError("expected payload")
                    ),
                )
            invalid_files = list(Path(tmpdir).glob("*.invalid.json"))

        self.assertEqual(len(invalid_files), 1)

    def test_grounded_missing_feedback_reaches_an_existing_candidate(self):
        candidate = {
            "study_id": "study_problem_3_modified",
            "reported_label": "Modified Problem 3",
            "study_name": "Real-payoff variant",
            "aliases": [],
            "evidence_block_ids": ["p003_text_00037"],
        }
        feedback = {
            "missing_studies": [
                {
                    "study": "Problem 3 real-payoff replication",
                    "reason": "Different group and modified task should remain distinct.",
                    "evidence_block_ids": ["p003_text_00037"],
                }
            ]
        }
        selected = _feedback_for_study(feedback, candidate)
        self.assertEqual(selected, feedback)

    def test_reconciliation_deduplicates_labels_without_merging_related_problems(self):
        mentions = [
            {
                "mention_id": "window_001_mention_01",
                "reported_label": "Problem 1",
                "study_name": "Gain frame",
                "kind": "experiment",
                "evidence_block_ids": ["p001"],
                "boundary_confidence": 0.9,
            },
            {
                "mention_id": "window_002_mention_01",
                "reported_label": "Problem 1 (N = 152)",
                "study_name": "Gain frame repeated mention",
                "kind": "experiment",
                "evidence_block_ids": ["p002"],
                "boundary_confidence": 0.8,
            },
            {
                "mention_id": "window_001_mention_02",
                "reported_label": "Problem 2",
                "study_name": "Loss frame",
                "kind": "experiment",
                "evidence_block_ids": ["p003"],
                "boundary_confidence": 0.9,
            },
        ]
        candidates, report = _reconcile_mentions(mentions)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["source_mention_ids"][:2], [
            "window_001_mention_01",
            "window_002_mention_01",
        ])
        self.assertEqual(candidates[1]["reported_label"], "Problem 2")
        self.assertTrue(report["all_mentions_assigned"])

    def test_reconciliation_nests_material_versions_under_one_parent_unit(self):
        mentions = [
            {
                "mention_id": "m1",
                "reported_label": "Problem 10",
                "study_name": "First version",
                "kind": "experiment",
                "evidence_block_ids": ["p1"],
                "material_variants": [
                    {
                        "label": "parentheses values",
                        "role": "form",
                        "evidence_block_ids": ["p1"],
                    }
                ],
            },
            {
                "mention_id": "m2",
                "reported_label": "Problem 10",
                "study_name": "Second version",
                "kind": "experiment",
                "evidence_block_ids": ["p2"],
                "material_variants": [
                    {
                        "label": "bracket values",
                        "role": "form",
                        "evidence_block_ids": ["p2"],
                    }
                ],
            },
            {
                "mention_id": "m3",
                "reported_label": "Problem 3",
                "study_name": "Real payoff problem",
                "kind": "experiment",
                "evidence_block_ids": ["p3"],
                "material_variants": [],
            },
        ]
        candidates, _ = _reconcile_mentions(mentions)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(
            [candidate["study_id"] for candidate in candidates],
            ["study_problem_10", "study_problem_3"],
        )
        self.assertEqual(
            [variant["label"] for variant in candidates[0]["material_variants"]],
            ["parentheses values", "bracket values"],
        )

    def test_reconciliation_merges_qualified_duplicate_with_overlapping_evidence(self):
        mentions = [
            {
                "mention_id": "m1",
                "reported_label": "Allocation distribution problem (20 objects across five people)",
                "study_name": "Allocation-distribution randomness judgment",
                "kind": "experiment",
                "evidence_block_ids": ["p005_text", "p005_table"],
            },
            {
                "mention_id": "m2",
                "reported_label": "Allocation distribution problem",
                "study_name": "Allocation distribution judgment",
                "kind": "experiment",
                "evidence_block_ids": ["p005_table", "p005_result"],
            },
        ]

        candidates, report = _reconcile_mentions(mentions)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["study_id"], "study_allocation_distribution_problem")
        self.assertEqual(set(candidates[0]["source_mention_ids"]), {"m1", "m2"})
        self.assertEqual(report["actions"][0]["action"], "merge_duplicate_mentions")

    def test_reconciliation_attaches_subordinate_group_and_table_to_formal_parent(self):
        mentions = [
            {
                "mention_id": "parent",
                "reported_label": "Experiment VI",
                "study_name": "Order manipulation",
                "kind": "experiment",
                "evidence_block_ids": ["p013_method", "p013_result", "p014_table"],
            },
            {
                "mention_id": "group",
                "reported_label": "Series B new group",
                "study_name": "Within-person order group",
                "kind": "experiment",
                "evidence_block_ids": ["p013_result"],
            },
            {
                "mention_id": "table",
                "reported_label": "Series B checklist table",
                "study_name": "Checklist result rows",
                "kind": "experiment",
                "evidence_block_ids": ["p014_table"],
            },
            {
                "mention_id": "next",
                "reported_label": "Experiment VII",
                "study_name": "New top-level manipulation",
                "kind": "experiment",
                "evidence_block_ids": ["p015_method"],
            },
        ]

        candidates, report = _reconcile_mentions(mentions)

        self.assertEqual([item["study_id"] for item in candidates], ["study_experiment_vi", "study_experiment_vii"])
        self.assertEqual(
            set(candidates[0]["source_mention_ids"]),
            {"parent", "group", "table"},
        )
        self.assertEqual(
            [action["action"] for action in report["actions"]],
            ["attach_subordinate_mentions", "attach_subordinate_mentions"],
        )

    def test_reconciliation_does_not_attach_child_shared_by_multiple_parents(self):
        mentions = [
            {
                "mention_id": "first",
                "reported_label": "Experiment I",
                "study_name": "First experiment",
                "kind": "experiment",
                "evidence_block_ids": ["shared_table"],
            },
            {
                "mention_id": "second",
                "reported_label": "Experiment II",
                "study_name": "Second experiment",
                "kind": "experiment",
                "evidence_block_ids": ["shared_table"],
            },
            {
                "mention_id": "row",
                "reported_label": "Positive response row",
                "study_name": "Shared table row",
                "kind": "other",
                "evidence_block_ids": ["shared_table"],
            },
        ]

        candidates, _ = _reconcile_mentions(mentions)

        self.assertEqual(len(candidates), 3)

    def test_boundary_adjudication_merges_method_and_result_descriptions(self):
        candidates = [
            {
                "study_id": "study_2_2_experiment",
                "reported_label": "2.2. Experiment",
                "study_name": "Field choice experiment method",
                "source_anchor": True,
                "aliases": [],
                "source_mention_ids": ["method"],
                "evidence_block_ids": ["method_block"],
                "material_variants": [],
                "participant_task_hint": "Choose between alternatives.",
                "quantitative_target_hint": None,
                "boundary_confidence": 0.8,
            },
            {
                "study_id": "study_3_2_experiment",
                "reported_label": "3.2. Experiment",
                "study_name": "Field choice experiment results",
                "source_anchor": True,
                "aliases": [],
                "source_mention_ids": ["results"],
                "evidence_block_ids": ["result_block"],
                "material_variants": [],
                "participant_task_hint": "Choose between alternatives.",
                "quantitative_target_hint": "55% selected the target.",
                "boundary_confidence": 0.9,
            },
        ]
        payload = {
            "merge_groups": [
                {
                    "member_study_ids": [
                        "study_2_2_experiment",
                        "study_3_2_experiment",
                    ],
                    "reason": "Method and result sections describe one collection.",
                    "evidence_block_ids": ["method_block", "result_block"],
                }
            ],
            "reject_candidates": [],
        }

        _validate_boundary_adjudication_payload(payload, candidates=candidates)
        reconciled, actions, rejected = _apply_boundary_adjudication(candidates, payload)

        self.assertEqual(len(reconciled), 1)
        self.assertEqual(
            reconciled[0]["evidence_block_ids"],
            ["method_block", "result_block"],
        )
        self.assertIn("study_3_2_experiment", reconciled[0]["aliases"])
        self.assertIn("55% selected", reconciled[0]["quantitative_target_hint"])
        self.assertEqual(actions[0]["action"], "llm_merge_candidates")
        self.assertEqual(rejected, [])

    def test_shared_questionnaire_context_attaches_without_merging_task_families(self):
        candidates = [
            {
                "study_id": "questionnaire_context",
                "reported_label": "Questionnaire data collection",
                "study_name": "Shared questionnaire sample",
                "source_anchor": False,
                "source_mention_ids": ["context"],
                "evidence_block_ids": ["sample_block"],
            },
            {
                "study_id": "birth_sequence_a",
                "reported_label": "Birth sequence frequency estimate",
                "study_name": "Birth sequence task",
                "source_anchor": False,
                "source_mention_ids": ["birth_a"],
                "evidence_block_ids": ["birth_a_block"],
                "participant_task_hint": "Estimate the frequency of a birth sequence.",
            },
            {
                "study_id": "birth_sequence_b",
                "reported_label": "Birth sequence randomness comparison",
                "study_name": "Birth sequence task",
                "source_anchor": False,
                "source_mention_ids": ["birth_b"],
                "evidence_block_ids": ["birth_b_block"],
                "participant_task_hint": "Estimate the frequency of a birth sequence.",
            },
            {
                "study_id": "program_choice",
                "reported_label": "High-school program choice",
                "study_name": "Program classification task",
                "source_anchor": False,
                "source_mention_ids": ["program"],
                "evidence_block_ids": ["program_block"],
                "participant_task_hint": "Choose which school program generated a class.",
            },
        ]
        payload = {
            "merge_groups": [
                {
                    "member_study_ids": ["birth_sequence_a", "birth_sequence_b"],
                    "canonical_study_id": "birth_sequence_a",
                    "reason": "The two items use one birth-sequence task family.",
                    "evidence_block_ids": ["birth_a_block", "birth_b_block"],
                }
            ],
            "shared_context_links": [
                {
                    "context_study_id": "questionnaire_context",
                    "target_study_ids": [
                        "birth_sequence_a",
                        "birth_sequence_b",
                        "program_choice",
                    ],
                    "reason": "The sample paragraph applies to all questionnaire tasks.",
                    "evidence_block_ids": ["sample_block"],
                }
            ],
            "reject_candidates": [],
        }

        _validate_boundary_adjudication_payload(payload, candidates=candidates)
        reconciled, actions, rejected = _apply_boundary_adjudication(candidates, payload)

        self.assertEqual(
            [item["study_id"] for item in reconciled],
            ["birth_sequence_a", "program_choice"],
        )
        self.assertIn("sample_block", reconciled[0]["evidence_block_ids"])
        self.assertIn("sample_block", reconciled[1]["evidence_block_ids"])
        self.assertEqual(len(reconciled[0]["component_mentions"]), 2)
        self.assertEqual(rejected[0]["study_id"], "questionnaire_context")
        self.assertTrue(
            any(action["action"] == "llm_attach_shared_context" for action in actions)
        )

    def test_disconnected_questionnaire_tasks_cannot_be_merged_as_one_collection(self):
        candidates = [
            {
                "study_id": "birth_task",
                "reported_label": "Birth sequence estimate",
                "study_name": "Birth sequence probability task",
                "participant_task_hint": "Estimate a birth sequence frequency.",
                "evidence_block_ids": ["birth_block"],
            },
            {
                "study_id": "program_task",
                "reported_label": "School program choice",
                "study_name": "School program classification",
                "participant_task_hint": "Choose program A or program B.",
                "evidence_block_ids": ["program_block"],
            },
            {
                "study_id": "marble_task",
                "reported_label": "Marble distribution choice",
                "study_name": "Marble allocation judgment",
                "participant_task_hint": "Choose the more likely marble distribution.",
                "evidence_block_ids": ["marble_block"],
            },
        ]
        payload = {
            "merge_groups": [
                {
                    "member_study_ids": ["birth_task", "program_task", "marble_task"],
                    "reason": "They used one questionnaire sample.",
                    "evidence_block_ids": [
                        "birth_block",
                        "program_block",
                        "marble_block",
                    ],
                }
            ],
            "shared_context_links": [],
            "reject_candidates": [],
        }

        with self.assertRaisesRegex(ValueError, "not a connected task family"):
            _validate_boundary_adjudication_payload(payload, candidates=candidates)

    def test_two_disconnected_unlabeled_tasks_cannot_be_merged(self):
        candidates = [
            {
                "study_id": "probability_task",
                "reported_label": "Probability judgment",
                "study_name": "Birth sequence estimate",
                "participant_task_hint": "Estimate the frequency of a birth sequence.",
                "quantitative_target_hint": "Mean probability estimate.",
                "evidence_block_ids": ["probability"],
            },
            {
                "study_id": "school_task",
                "reported_label": "School choice",
                "study_name": "Academic program classification",
                "participant_task_hint": "Choose which school program produced a class.",
                "quantitative_target_hint": "Program choice proportion.",
                "evidence_block_ids": ["school"],
            },
        ]
        payload = {
            "merge_groups": [
                {
                    "member_study_ids": ["probability_task", "school_task"],
                    "reason": "Both appeared in one questionnaire.",
                    "evidence_block_ids": ["probability", "school"],
                }
            ],
            "shared_context_links": [],
            "reject_candidates": [],
        }

        with self.assertRaisesRegex(ValueError, "not a connected task family"):
            _validate_boundary_adjudication_payload(payload, candidates=candidates)

    def test_repeated_unlabeled_descriptions_form_one_connected_task_family(self):
        candidates = [
            {
                "study_id": "method",
                "reported_label": "The experiment",
                "study_name": "Minimal intergroup allocation experiment",
                "participant_task_hint": "Choose allocations of money to ingroup and outgroup members.",
                "quantitative_target_hint": "Ingroup favouritism and fairness scores.",
                "evidence_block_ids": ["method"],
            },
            {
                "study_id": "results",
                "reported_label": "This experiment",
                "study_name": "Intergroup allocation matrix experiment",
                "participant_task_hint": "Make allocation choices for anonymous ingroup and outgroup members.",
                "quantitative_target_hint": "Allocation matrix ingroup favouritism scores.",
                "evidence_block_ids": ["results"],
            },
            {
                "study_id": "discussion",
                "reported_label": "Present study",
                "study_name": "Minimal group allocation study",
                "participant_task_hint": "Assign money between ingroup and outgroup members.",
                "quantitative_target_hint": "Discrimination in allocations and fairness.",
                "evidence_block_ids": ["discussion"],
            },
        ]
        payload = {
            "merge_groups": [
                {
                    "member_study_ids": ["method", "results", "discussion"],
                    "canonical_study_id": "method",
                    "reason": "Methods, results, and discussion describe one task.",
                    "evidence_block_ids": ["method", "results", "discussion"],
                }
            ],
            "shared_context_links": [],
            "reject_candidates": [],
        }

        _validate_boundary_adjudication_payload(payload, candidates=candidates)

    def test_task_with_its_own_response_target_cannot_be_demoted_to_shared_context(self):
        candidates = [
            {
                "study_id": "sampling_distribution_task",
                "reported_label": "Sampling distributions",
                "participant_task_hint": "Construct a subjective sampling distribution.",
                "quantitative_target_hint": "Median estimates and distribution variances.",
                "evidence_block_ids": ["sampling_block"],
            },
            {
                "study_id": "another_task",
                "reported_label": "Another task",
                "evidence_block_ids": ["another_block"],
            },
        ]
        payload = {
            "merge_groups": [],
            "shared_context_links": [
                {
                    "context_study_id": "sampling_distribution_task",
                    "target_study_ids": ["another_task"],
                    "reason": "Incorrectly treated as context.",
                    "evidence_block_ids": ["sampling_block"],
                }
            ],
            "reject_candidates": [],
        }

        with self.assertRaisesRegex(ValueError, "own response target"):
            _validate_boundary_adjudication_payload(payload, candidates=candidates)

    def test_boundary_adjudication_cannot_merge_distinct_formal_experiments(self):
        candidates = [
            {
                "study_id": "experiment_i",
                "reported_label": "Experiment I",
                "source_anchor": True,
                "evidence_block_ids": ["p1"],
            },
            {
                "study_id": "experiment_ii",
                "reported_label": "Experiment II",
                "source_anchor": True,
                "evidence_block_ids": ["p2"],
            },
        ]
        payload = {
            "merge_groups": [
                {
                    "member_study_ids": ["experiment_i", "experiment_ii"],
                    "reason": "Incorrect merge.",
                    "evidence_block_ids": ["p1", "p2"],
                }
            ],
            "reject_candidates": [],
        }

        with self.assertRaisesRegex(ValueError, "distinct formal source labels"):
            _validate_boundary_adjudication_payload(payload, candidates=candidates)

    def test_roman_and_arabic_labels_resolve_to_the_same_formal_unit(self):
        self.assertEqual(_explicit_unit_labels("Study I"), ["Study I"])
        self.assertEqual(_explicit_unit_labels("Study IIa"), ["Study IIA"])
        self.assertEqual(_explicit_unit_labels("Experiment IXb"), ["Experiment IXB"])

        candidates = [
            {
                "study_id": "study_1",
                "reported_label": "Study 1",
                "source_anchor": True,
                "evidence_block_ids": ["shared", "method"],
            },
            {
                "study_id": "study_i",
                "reported_label": "Study I",
                "source_anchor": True,
                "evidence_block_ids": ["shared", "results"],
            },
        ]
        payload = {
            "merge_groups": [
                {
                    "member_study_ids": ["study_1", "study_i"],
                    "canonical_study_id": "study_1",
                    "reason": "Arabic and Roman labels describe the same source unit.",
                    "evidence_block_ids": ["shared", "method", "results"],
                }
            ],
            "shared_context_links": [],
            "reject_candidates": [],
        }

        _validate_boundary_adjudication_payload(payload, candidates=candidates)

    def test_boundary_adjudication_can_merge_relation_linked_formal_problems(self):
        candidates = [
            {
                "study_id": "problem_1",
                "reported_label": "Problem 1",
                "study_name": "Gain frame",
                "source_anchor": True,
                "evidence_block_ids": ["problem_1_block"],
            },
            {
                "study_id": "problem_2",
                "reported_label": "Problem 2",
                "study_name": "Loss frame",
                "source_anchor": True,
                "evidence_block_ids": ["problem_2_block"],
            },
        ]
        relation_ledger = [
            {
                "relation_mention_id": "relation_1_2",
                "member_study_ids": ["problem_1", "problem_2"],
                "relationship_kind": "paired_contrast",
                "confidence": 0.95,
                "evidence_block_ids": ["relation_block"],
            }
        ]
        payload = {
            "merge_groups": [
                {
                    "member_study_ids": ["problem_1", "problem_2"],
                    "canonical_study_id": "problem_1",
                    "supporting_relation_ids": ["relation_1_2"],
                    "reason": "The paired frames jointly define one simulation target.",
                    "evidence_block_ids": [
                        "problem_1_block",
                        "problem_2_block",
                        "relation_block",
                    ],
                }
            ],
            "shared_context_links": [],
            "reject_candidates": [],
        }

        _validate_boundary_adjudication_payload(
            payload,
            candidates=candidates,
            relation_ledger=relation_ledger,
        )
        reconciled, _, _ = _apply_boundary_adjudication(candidates, payload)

        self.assertEqual(len(reconciled), 1)
        self.assertEqual(reconciled[0]["reported_label"], "Problem 1 + Problem 2")
        self.assertEqual(
            reconciled[0]["source_task_family_relation_ids"],
            ["relation_1_2"],
        )
        self.assertIn("relation_block", reconciled[0]["evidence_block_ids"])

    def test_formal_problem_merge_requires_explicit_relation_support(self):
        candidates = [
            {
                "study_id": "problem_1",
                "reported_label": "Problem 1",
                "evidence_block_ids": ["p1"],
            },
            {
                "study_id": "problem_2",
                "reported_label": "Problem 2",
                "evidence_block_ids": ["p2"],
            },
        ]
        payload = {
            "merge_groups": [
                {
                    "member_study_ids": ["problem_1", "problem_2"],
                    "reason": "Unsupported merge.",
                    "evidence_block_ids": ["p1", "p2"],
                }
            ],
            "reject_candidates": [],
        }

        with self.assertRaisesRegex(ValueError, "distinct formal source labels"):
            _validate_boundary_adjudication_payload(payload, candidates=candidates)

    def test_relation_graph_compiles_connected_problems_into_task_family(self):
        candidates = [
            {
                "study_id": f"problem_{number}",
                "reported_label": f"Problem {number}",
                "evidence_block_ids": [f"problem_{number}_block"],
            }
            for number in (5, 6, 7)
        ]
        candidate_ledger = [
            {
                "study_id": candidate["study_id"],
                "evidence_block_ids": candidate["evidence_block_ids"],
            }
            for candidate in candidates
        ]
        relation_ledger = [
            {
                "relation_mention_id": "relation_5_6",
                "member_study_ids": ["problem_5", "problem_6"],
                "unresolved_member_labels": [],
                "members_are_distinct_empirical_units": True,
                "relationship_kind": "paired_contrast",
                "confidence": 0.9,
                "evidence_block_ids": ["relation_5_6_block"],
            },
            {
                "relation_mention_id": "relation_6_7",
                "member_study_ids": ["problem_6", "problem_7"],
                "unresolved_member_labels": [],
                "members_are_distinct_empirical_units": True,
                "relationship_kind": "multi_unit_comparison",
                "confidence": 0.9,
                "evidence_block_ids": ["relation_6_7_block"],
            },
        ]

        augmented = _augment_relation_problem_family_merges(
            {"merge_groups": [], "shared_context_links": [], "reject_candidates": []},
            candidates=candidates,
            candidate_ledger=candidate_ledger,
            relation_ledger=relation_ledger,
        )

        self.assertEqual(len(augmented["merge_groups"]), 1)
        group = augmented["merge_groups"][0]
        self.assertEqual(
            group["member_study_ids"],
            ["problem_5", "problem_6", "problem_7"],
        )
        self.assertEqual(
            group["supporting_relation_ids"],
            ["relation_5_6", "relation_6_7"],
        )
        self.assertEqual(group["merge_source"], "relation_graph")

    def test_relation_graph_never_merges_formal_experiments(self):
        candidates = [
            {
                "study_id": "experiment_i",
                "reported_label": "Experiment I",
                "evidence_block_ids": ["p1"],
            },
            {
                "study_id": "experiment_ii",
                "reported_label": "Experiment II",
                "evidence_block_ids": ["p2"],
            },
        ]
        relation_ledger = [
            {
                "relation_mention_id": "relation_i_ii",
                "member_study_ids": ["experiment_i", "experiment_ii"],
                "unresolved_member_labels": [],
                "members_are_distinct_empirical_units": True,
                "relationship_kind": "paired_contrast",
                "confidence": 1.0,
                "evidence_block_ids": ["relation_block"],
            }
        ]

        augmented = _augment_relation_problem_family_merges(
            {"merge_groups": [], "shared_context_links": [], "reject_candidates": []},
            candidates=candidates,
            candidate_ledger=candidates,
            relation_ledger=relation_ledger,
        )

        self.assertEqual(augmented["merge_groups"], [])

    def test_boundary_adjudication_can_reject_unreported_generic_experiment_reference(self):
        candidates = [
            {
                "study_id": "unreported_experiments",
                "reported_label": "experiments not reported here",
                "source_anchor": True,
                "evidence_block_ids": ["discussion_block"],
            },
            {
                "study_id": "experiment_i",
                "reported_label": "Experiment I",
                "source_anchor": True,
                "evidence_block_ids": ["experiment_block"],
            },
        ]
        payload = {
            "merge_groups": [],
            "reject_candidates": [
                {
                    "study_id": "unreported_experiments",
                    "reason": "The source explicitly says these experiments are not reported.",
                    "evidence_block_ids": ["discussion_block"],
                }
            ],
        }

        _validate_boundary_adjudication_payload(payload, candidates=candidates)
        reconciled, _, rejected = _apply_boundary_adjudication(candidates, payload)

        self.assertEqual([item["study_id"] for item in reconciled], ["experiment_i"])
        self.assertEqual(rejected[0]["study_id"], "unreported_experiments")

    def test_boundary_merge_uses_formal_parent_even_when_child_is_listed_first(self):
        candidates = [
            {
                "study_id": "experiment_vi",
                "reported_label": "Experiment VI",
                "study_name": "Order experiment",
                "source_anchor": True,
                "aliases": [],
                "source_mention_ids": ["parent"],
                "evidence_block_ids": ["parent_block"],
                "material_variants": [],
            },
            {
                "study_id": "series_b",
                "reported_label": "Series B",
                "study_name": "Within-experiment group",
                "source_anchor": False,
                "aliases": [],
                "source_mention_ids": ["child"],
                "evidence_block_ids": ["child_block"],
                "material_variants": [],
            },
        ]
        payload = {
            "merge_groups": [
                {
                    "member_study_ids": ["series_b", "experiment_vi"],
                    "reason": "Series B is a group inside Experiment VI.",
                    "evidence_block_ids": ["parent_block", "child_block"],
                }
            ],
            "reject_candidates": [],
        }

        _validate_boundary_adjudication_payload(payload, candidates=candidates)
        reconciled, _, _ = _apply_boundary_adjudication(candidates, payload)

        self.assertEqual(reconciled[0]["study_id"], "experiment_vi")
        self.assertEqual(reconciled[0]["reported_label"], "Experiment VI")
        self.assertIn("series_b", reconciled[0]["aliases"])

    def test_formal_parent_labels_absorb_story_and_table_row_mentions(self):
        window_one = DiscoveryWindow(
            window_id="window_001",
            text="Study 1",
            block_ids=["p001"],
            pages=[1],
            char_count=7,
        )
        window_two = DiscoveryWindow(
            window_id="window_002",
            text="Study 1 and Study 2",
            block_ids=["p002"],
            pages=[2],
            char_count=19,
        )
        raw_mentions = [
            (
                {
                    "reported_label": "Study 1 - Supermarket Story",
                    "study_name": "False consensus stories",
                    "kind": "study",
                    "material_variants": [
                        {
                            "label": "Supermarket Story",
                            "role": "stimulus",
                            "evidence_block_ids": ["p001"],
                        }
                    ],
                    "evidence_block_ids": ["p001"],
                },
                window_one,
                1,
            ),
            (
                {
                    "reported_label": "Study 1 - Term Paper Story",
                    "study_name": "False consensus stories",
                    "kind": "study",
                    "material_variants": [
                        {
                            "label": "Term Paper Story",
                            "role": "stimulus",
                            "evidence_block_ids": ["p002"],
                        }
                    ],
                    "evidence_block_ids": ["p002"],
                },
                window_two,
                1,
            ),
            (
                {
                    "reported_label": "College Students in General (Study 2)",
                    "study_name": "Personal characteristics questionnaire",
                    "kind": "study",
                    "material_variants": [],
                    "evidence_block_ids": ["p002"],
                },
                window_two,
                2,
            ),
        ]
        normalized = [
            _normalize_mention(raw, window, position)
            for raw, window, position in raw_mentions
        ]
        candidates, _ = _reconcile_mentions(normalized)
        self.assertEqual([item["study_id"] for item in candidates], ["study_1", "study_2"])
        self.assertEqual(
            [variant["label"] for variant in candidates[0]["material_variants"]],
            ["Supermarket Story", "Term Paper Story"],
        )

    def test_provenance_partition_is_fail_closed_but_preserves_uncertainty(self):
        records = [
            {
                "study_id": "current",
                "experiment_id": "Study 1",
                "unit_provenance": "current_paper",
                "is_distinct_empirical_unit": True,
                "source_anchor": True,
            },
            {
                "study_id": "prior",
                "experiment_id": "Prior study",
                "unit_provenance": "cited_prior",
                "is_distinct_empirical_unit": True,
                "source_anchor": True,
            },
            {
                "study_id": "fragment",
                "experiment_id": "Sample description",
                "unit_provenance": "current_paper",
                "is_distinct_empirical_unit": False,
                "source_anchor": True,
            },
            {
                "study_id": "unclear",
                "experiment_id": "Unlabeled task",
                "unit_provenance": "unclear",
                "is_distinct_empirical_unit": True,
                "replicable": "UNCERTAIN",
                "source_anchor": True,
            },
        ]
        accepted, rejected = _partition_empirical_units(records)
        self.assertEqual([item["study_id"] for item in accepted], ["current", "unclear"])
        self.assertEqual([item["study_id"] for item in rejected], ["prior", "fragment"])

    def test_unlabeled_candidate_without_quantitative_result_is_preserved_for_review(self):
        records = [
            {
                "study_id": "discussion_section",
                "experiment_id": "II",
                "unit_provenance": "current_paper",
                "is_distinct_empirical_unit": True,
                "source_anchor": False,
                "empirical_support": {
                    "own_sample_or_assignment": "unclear",
                    "participant_facing_task": "unclear",
                    "quantitative_result": "no",
                },
                "replicable": "YES",
            }
        ]

        accepted, rejected = _partition_empirical_units(records)

        self.assertEqual([item["study_id"] for item in accepted], ["discussion_section"])
        self.assertEqual(accepted[0]["replicable"], "UNCERTAIN")
        self.assertEqual(rejected, [])

    def test_comparison_relations_link_units_without_merging_them(self):
        experiments = [
            {
                "study_id": "problem_1",
                "experiment_id": "Problem 1",
                "replicable": "YES",
                "candidate_aliases": ["Gain frame problem"],
            },
            {
                "study_id": "problem_2",
                "experiment_id": "Problem 2",
                "replicable": "YES",
                "candidate_aliases": ["Loss frame problem"],
            },
        ]
        relations = [
            {
                "relation_mention_id": "r1",
                "member_labels": ["Problem 1", "Problem 2"],
                "members_are_distinct_empirical_units": True,
                "relationship_kind": "paired_contrast",
                "comparison_target": "choice proportions under gain and loss frames",
                "evidence_block_ids": ["p004_text"],
                "evidence_summary": "The paper explicitly compares the two problems.",
                "confidence": 0.95,
            }
        ]
        groups, rejected, ignored = _reconcile_comparison_groups(relations, experiments)
        self.assertEqual(rejected, [])
        self.assertEqual(ignored, [])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["member_study_ids"], ["problem_1", "problem_2"])
        self.assertEqual(len(experiments), 2)

    def test_shared_sample_relation_is_context_metadata_not_comparison_group(self):
        experiments = [
            {
                "study_id": "preliminary",
                "experiment_id": "2.1 Preliminary questionnaire",
                "candidate_aliases": ["3.1 Preliminary questionnaire", "Preliminary survey"],
            },
            {
                "study_id": "main_experiment",
                "experiment_id": "2.2 Experiment",
                "candidate_aliases": ["3.2 Experiment", "Field experiment"],
            },
        ]
        relations = [
            {
                "relation_mention_id": "r1",
                "member_refs": [
                    {"reported_label": "3.1 Preliminary questionnaire"},
                    {"reported_label": "3.2 Experiment"},
                ],
                "members_are_distinct_empirical_units": True,
                "relationship_kind": "shared_sample",
                "comparison_target": "same participants continued to the main phase",
                "evidence_block_ids": ["p004"],
            }
        ]

        groups, rejected, ignored = _reconcile_comparison_groups(relations, experiments)

        self.assertEqual(rejected, [])
        self.assertEqual(groups, [])
        self.assertEqual(len(ignored), 1)
        self.assertIn("shared_contexts", ignored[0]["ignore_reason"])
        self.assertEqual(
            ignored[0]["resolved_member_study_ids"],
            ["preliminary", "main_experiment"],
        )

    def test_shared_sample_evidence_is_attached_before_study_extraction(self):
        candidates = [
            {
                "study_id": "birth_task",
                "reported_label": "Birth sequence task",
                "study_name": "Birth sequence judgment",
                "evidence_block_ids": ["birth_block"],
            },
            {
                "study_id": "program_task",
                "reported_label": "Program choice task",
                "study_name": "Program classification",
                "evidence_block_ids": ["program_block"],
            },
        ]
        relations = [
            {
                "relation_mention_id": "window_001_relation_01",
                "member_labels": ["Birth sequence task", "Program choice task"],
                "relationship_kind": "shared_sample",
                "evidence_summary": "Both tasks were administered in one questionnaire pool.",
                "evidence_block_ids": ["sample_block", "invalid_block"],
            }
        ]

        actions = _attach_discovered_shared_sample_contexts(
            candidates,
            relations,
            valid_refs={"birth_block", "program_block", "sample_block"},
        )

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["unresolved_member_labels"], [])
        for candidate in candidates:
            self.assertIn("sample_block", candidate["evidence_block_ids"])
            self.assertEqual(len(candidate["shared_contexts"]), 1)
            self.assertEqual(
                candidate["shared_contexts"][0]["evidence_block_ids"],
                ["sample_block"],
            )

    def test_isolated_invalid_study_ref_is_pruned_when_valid_support_remains(self):
        payload = {
            "evidence_refs": ["valid_a", "adjacent_typo"],
            "field_evidence": {
                "output": ["valid_b", "adjacent_typo"],
            },
            "simulation_barriers": [],
            "material_variants": [
                {"evidence_refs": ["valid_b", "adjacent_typo"]},
            ],
        }

        _prune_invalid_study_evidence_refs(
            payload,
            valid_refs={"valid_a", "valid_b"},
        )

        self.assertEqual(payload["evidence_refs"], ["valid_a"])
        self.assertEqual(payload["field_evidence"]["output"], ["valid_b"])
        self.assertEqual(payload["material_variants"][0]["evidence_refs"], ["valid_b"])
        self.assertEqual(len(payload["_evidence_ref_repairs"]), 3)

    def test_invalid_study_ref_cannot_remove_a_claims_only_evidence(self):
        payload = {
            "evidence_refs": ["valid_a"],
            "field_evidence": {"output": ["invalid_only"]},
            "simulation_barriers": [],
            "material_variants": [],
        }

        with self.assertRaisesRegex(ValueError, "only invalid evidence refs"):
            _prune_invalid_study_evidence_refs(
                payload,
                valid_refs={"valid_a"},
            )

    def test_comparison_group_preserves_relation_to_ineligible_inventory_unit(self):
        experiments = [
            {
                "study_id": "study_1",
                "experiment_id": "Study 1",
                "replicable": "YES",
            },
            {
                "study_id": "study_2",
                "experiment_id": "Study 2",
                "replicable": "NO",
            },
        ]
        relations = [
            {
                "relation_mention_id": "r1",
                "member_labels": ["Study 1", "Study 2"],
                "members_are_distinct_empirical_units": True,
                "relationship_kind": "replication_set",
                "comparison_target": "same hypothesis under two procedures",
                "evidence_block_ids": ["p010_text"],
            }
        ]
        groups, rejected, ignored = _reconcile_comparison_groups(relations, experiments)
        self.assertEqual(rejected, [])
        self.assertEqual(ignored, [])
        self.assertEqual(groups[0]["member_study_ids"], ["study_1", "study_2"])

    def test_relation_to_boundary_rejected_nonunit_is_ignored_not_unresolved(self):
        experiments = [
            {
                "study_id": "study_1",
                "experiment_id": "Experiment I",
                "replicable": "YES",
            }
        ]
        rejected = {
            "study_id": "unreported_experiments",
            "experiment_id": "experiments we have not here reported",
            "study_name": "unreported experiments",
        }
        relations = [
            {
                "relation_mention_id": "r1",
                "member_labels": [
                    "Experiment I",
                    "experiments we have not here reported",
                    "present investigation",
                ],
                "members_are_distinct_empirical_units": True,
                "relationship_kind": "multi_unit_comparison",
                "evidence_block_ids": ["p001"],
            }
        ]

        groups, unresolved, ignored = _reconcile_comparison_groups(
            relations,
            experiments,
            all_records=[*experiments, rejected],
        )

        self.assertEqual(groups, [])
        self.assertEqual(unresolved, [])
        self.assertEqual(len(ignored), 1)
        self.assertEqual(ignored[0]["ignored_member_study_ids"], ["unreported_experiments"])
        self.assertEqual(ignored[0]["unresolved_member_labels"], ["present investigation"])

    def test_same_member_relationship_mentions_collapse_to_one_group(self):
        experiments = [
            {"study_id": "study_1", "experiment_id": "Study 1", "replicable": "YES"},
            {"study_id": "study_2", "experiment_id": "Study 2", "replicable": "NO"},
        ]
        relations = [
            {
                "relation_mention_id": "window_001_relation_01",
                "member_labels": ["Study 1", "Study 2"],
                "members_are_distinct_empirical_units": True,
                "relationship_kind": "replication_set",
                "comparison_target": "same hypothesis",
                "evidence_block_ids": ["p001"],
            },
            {
                "relation_mention_id": "window_002_relation_01",
                "member_labels": ["Study 1", "Study 2"],
                "members_are_distinct_empirical_units": True,
                "relationship_kind": "multi_unit_comparison",
                "comparison_target": "contrast procedures",
                "evidence_block_ids": ["p002"],
            },
        ]
        groups, rejected, ignored = _reconcile_comparison_groups(relations, experiments)
        self.assertEqual(rejected, [])
        self.assertEqual(ignored, [])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["relationship_kind"], "multi_unit_comparison")
        self.assertEqual(groups[0]["evidence_refs"], ["p001", "p002"])

    def test_nested_member_sets_remain_distinct_comparison_groups(self):
        experiments = [
            {"study_id": f"study_{number}", "experiment_id": f"Study {number}", "replicable": "YES"}
            for number in (1, 2, 3)
        ]
        relations = [
            {
                "relation_mention_id": "window_001_relation_01",
                "member_labels": ["Study 2", "Study 3"],
                "members_are_distinct_empirical_units": True,
                "relationship_kind": "paired_contrast",
                "comparison_target": "direct pairwise contrast",
                "evidence_block_ids": ["p001"],
            },
            {
                "relation_mention_id": "window_001_relation_02",
                "member_labels": ["Study 1", "Study 2", "Study 3"],
                "members_are_distinct_empirical_units": True,
                "relationship_kind": "multi_unit_comparison",
                "comparison_target": "joint evidence across studies",
                "evidence_block_ids": ["p001"],
            },
        ]
        groups, rejected, ignored = _reconcile_comparison_groups(relations, experiments)
        self.assertEqual(rejected, [])
        self.assertEqual(ignored, [])
        self.assertEqual(
            [group["member_study_ids"] for group in groups],
            [["study_2", "study_3"], ["study_1", "study_2", "study_3"]],
        )

    def test_intra_unit_relation_is_audited_without_forcing_a_split(self):
        experiments = [
            {
                "study_id": "problem_3",
                "experiment_id": "Problem 3",
                "replicable": "YES",
                "candidate_aliases": ["Problem 3 Decision (i)", "Problem 3 Decision (ii)"],
            }
        ]
        relations = [
            {
                "relation_mention_id": "r1",
                "member_labels": ["Problem 3 Decision (i)", "Problem 3 Decision (ii)"],
                "members_are_distinct_empirical_units": False,
                "relationship_kind": "paired_contrast",
                "comparison_target": "two concurrent choices",
                "evidence_block_ids": ["p002_text"],
            }
        ]
        groups, rejected, ignored = _reconcile_comparison_groups(relations, experiments)
        self.assertEqual(groups, [])
        self.assertEqual(rejected, [])
        self.assertEqual(len(ignored), 1)
        self.assertIn("intra-unit", ignored[0]["ignore_reason"])

    def test_relation_consumed_by_task_family_merge_is_not_a_comparison_group(self):
        experiments = [
            {
                "study_id": "problem_1",
                "experiment_id": "Problem 1 + Problem 2",
                "candidate_aliases": ["Problem 1", "Problem 2"],
                "source_task_family_relation_ids": ["relation_1_2"],
            }
        ]
        relations = [
            {
                "relation_mention_id": "relation_1_2",
                "member_labels": ["Problem 1", "Problem 2"],
                "members_are_distinct_empirical_units": True,
                "relationship_kind": "paired_contrast",
                "evidence_block_ids": ["p001"],
            }
        ]

        groups, rejected, ignored = _reconcile_comparison_groups(
            relations,
            experiments,
        )

        self.assertEqual(groups, [])
        self.assertEqual(rejected, [])
        self.assertEqual(len(ignored), 1)
        self.assertIn("task-family merge", ignored[0]["ignore_reason"])

    def test_task_family_component_labels_resolve_consumed_relation(self):
        experiments = [
            {
                "study_id": "study_problem_8",
                "experiment_id": "Problem 8 + Problem 9",
                "candidate_aliases": ["Problem 9"],
                "candidate_components": [
                    {
                        "study_id": "study_problem_8",
                        "reported_label": "Problem 8",
                        "study_name": "Cash loss",
                    },
                    {
                        "study_id": "study_problem_9",
                        "reported_label": "Problem 9",
                        "study_name": "Ticket loss",
                    },
                ],
                "source_task_family_relation_ids": ["relation_8_9"],
            }
        ]
        relations = [
            {
                "relation_mention_id": "relation_8_9",
                "member_refs": [
                    {"reported_label": "Problem 8"},
                    {"reported_label": "Problem 9"},
                ],
                "members_are_distinct_empirical_units": True,
                "relationship_kind": "paired_contrast",
                "evidence_block_ids": ["p005"],
            }
        ]

        groups, rejected, ignored = _reconcile_comparison_groups(
            relations,
            experiments,
        )

        self.assertEqual(groups, [])
        self.assertEqual(rejected, [])
        self.assertEqual(len(ignored), 1)
        self.assertEqual(
            ignored[0]["resolved_member_study_ids"],
            ["study_problem_8"],
        )
        self.assertIn("task-family merge", ignored[0]["ignore_reason"])

    def test_overlapping_same_window_contrasts_form_one_maximal_group(self):
        experiments = [
            {"study_id": f"problem_{number}", "experiment_id": f"Problem {number}", "replicable": "YES"}
            for number in (5, 6, 7)
        ]
        relations = [
            {
                "relation_mention_id": "window_002_relation_01",
                "member_labels": ["Problem 5", "Problem 6"],
                "members_are_distinct_empirical_units": True,
                "relationship_kind": "paired_contrast",
                "comparison_target": "certainty comparison",
                "evidence_block_ids": ["p050", "p051"],
            },
            {
                "relation_mention_id": "window_002_relation_02",
                "member_labels": ["Problem 6", "Problem 7"],
                "members_are_distinct_empirical_units": True,
                "relationship_kind": "paired_contrast",
                "comparison_target": "pseudocertainty comparison",
                "evidence_block_ids": ["p051", "p052"],
            },
        ]
        groups, rejected, ignored = _reconcile_comparison_groups(relations, experiments)
        self.assertEqual(rejected, [])
        self.assertEqual(ignored, [])
        self.assertEqual(len(groups), 1)
        self.assertEqual(
            groups[0]["member_study_ids"],
            ["problem_5", "problem_6", "problem_7"],
        )

    def test_material_variants_do_not_create_a_cross_unit_comparison_group(self):
        experiments = [
            {
                "study_id": "problem_10",
                "experiment_id": "Problem 10",
                "replicable": "YES",
                "material_variants": [
                    {"variant_id": "parentheses", "label": "values in parentheses"},
                    {"variant_id": "brackets", "label": "values in brackets"},
                ],
                "candidate_aliases": [
                    "Problem 10 values in parentheses",
                    "Problem 10 values in brackets",
                ],
            },
        ]
        relations = [
            {
                "relation_mention_id": "window_003_relation_01",
                "member_labels": [
                    "Problem 10 values in parentheses",
                    "Problem 10 values in brackets",
                ],
                "members_are_distinct_empirical_units": False,
                "relationship_kind": "paired_contrast",
                "comparison_target": "willingness to travel",
                "evidence_block_ids": ["p005_text"],
            }
        ]
        groups, rejected, ignored = _reconcile_comparison_groups(relations, experiments)
        self.assertEqual(groups, [])
        self.assertEqual(rejected, [])
        self.assertEqual(len(ignored), 1)
        self.assertIn("intra-unit", ignored[0]["ignore_reason"])

    def test_discovery_windows_cover_every_block_with_bounded_prompts(self):
        document = _document([f"block-{index} " * 500 for index in range(1, 8)])
        windows = build_discovery_windows(document, max_chars=2500, overlap_units=1)
        covered = {block_id for window in windows for block_id in window.block_ids}
        self.assertEqual(covered, {block.block_id for block in document.blocks})
        self.assertGreater(len(windows), 1)
        self.assertTrue(all(window.char_count <= 2500 for window in windows))

    def test_global_candidate_ledger_is_bounded_without_omitting_identities(self):
        document = _document(["Evidence block for all candidate units."])
        index = PdfEvidenceIndex(document)
        candidates = [
            {
                "study_id": f"study_{number}",
                "reported_label": f"Unlabeled empirical collection {number} " + "label " * 80,
                "study_name": f"Candidate {number} " + "description " * 80,
                "kind": "experiment",
                "source_anchor": False,
                "participant_task_hint": "task " * 200,
                "quantitative_target_hint": "result " * 200,
                "evidence_block_ids": ["p001_text"],
            }
            for number in range(80)
        ]

        relations = [
            {
                "relation_mention_id": f"relation_{number}",
                "member_labels": [f"Unlabeled empirical collection {number}", "Unlabeled empirical collection 0"],
                "relationship_kind": "paired_contrast",
                "members_are_distinct_empirical_units": True,
                "comparison_target": "comparison " * 100,
                "evidence_summary": "evidence " * 100,
                "evidence_block_ids": ["p001_text"],
                "confidence": 0.9,
            }
            for number in range(1, 80)
        ]
        ledger, relation_ledger, prompt, strategy = _build_bounded_candidate_ledger(
            candidates,
            relations,
            index,
            pdf_name="paper.pdf",
        )

        self.assertLessEqual(len(prompt), BOUNDARY_ADJUDICATION_MAX_CONTEXT_CHARS)
        self.assertEqual({item["study_id"] for item in ledger}, {f"study_{n}" for n in range(80)})
        self.assertEqual(
            {item["relation_mention_id"] for item in relation_ledger},
            {f"relation_{n}" for n in range(1, 80)},
        )
        self.assertIn(strategy, {"compact", "identity_only", "minimal"})

    def test_evidence_context_prioritizes_exact_citations_before_neighbors(self):
        document = _document(
            [
                "neighbor " * 55,
                "AUTHORITATIVE_TABLE " * 15,
                "following neighbor " * 40,
            ]
        )
        context = PdfEvidenceIndex(document).context_for_study(
            {"study_id": "study_1"},
            gaps=[],
            allow_full_document=False,
            anchor_refs=["p002_text"],
            anchor_radius=1,
            use_facet_retrieval=False,
            max_chars=700,
        )
        self.assertIn("p002_text", context.block_ids)
        self.assertIn("AUTHORITATIVE_TABLE", context.text)

    def test_study_field_audit_combines_citations_from_separate_windows(self):
        filler = "separating filler " * 1400
        document = _document(
            [
                "METHOD_MARKER Study 1 assigned two forms.",
                filler,
                "RESULT_TABLE_MARKER Both forms and their response counts are reported.",
            ]
        )
        stage1 = {
            "experiments": [
                {
                    "study_id": "study_1",
                    "experiment_id": "Study 1",
                    "study_name": "Two-form task",
                    "replicable": "YES",
                    "material_variants": [
                        {
                            "variant_id": "form_a",
                            "label": "Form A",
                            "role": "form",
                            "evidence_refs": ["p001_text", "p003_text"],
                        },
                        {
                            "variant_id": "form_b",
                            "label": "Form B",
                            "role": "form",
                            "evidence_refs": ["p001_text", "p003_text"],
                        },
                    ],
                    "evidence_refs": ["p001_text", "p003_text"],
                    "field_evidence": {
                        "material_variants": ["p001_text", "p003_text"]
                    },
                }
            ]
        }

        class AuditClient:
            model = "fake"

            def __init__(self):
                self.study_prompt = ""
                self.boundary_calls = 0

            def generate_content(self, prompt, **kwargs):
                del kwargs
                if "independently auditing empirical-unit boundaries" in prompt:
                    self.boundary_calls += 1
                    return json.dumps(
                        {
                            "confidence": 0.9,
                            "missing_studies": [],
                            "split_merge_corrections": [],
                            "comparison_group_corrections": [],
                            "notes": "No boundary issue.",
                        }
                    )
                if "independently auditing one empirical unit" in prompt:
                    self.study_prompt = prompt
                    return json.dumps(
                        {
                            "study_id": "study_1",
                            "confidence": 0.9,
                            "study_field_corrections": [],
                            "eligibility_corrections": [],
                            "notes": "Both cited regions support the field.",
                        }
                    )
                raise AssertionError(prompt[:100])

        client = AuditClient()
        with patch(
            "generation_pipeline.stage1_verifier.parse_pdf_document",
            return_value=document,
        ):
            report = verify_stage1_inventory(
                stage1,
                Path("paper.pdf"),
                client,
                workers=2,
            )
        self.assertEqual(report["overall"], "pass")
        self.assertIn("METHOD_MARKER", client.study_prompt)
        self.assertIn("RESULT_TABLE_MARKER", client.study_prompt)
        self.assertEqual(
            report["study_audit"]["studies"][0]["audited_evidence_refs"],
            ["p001_text", "p003_text"],
        )
        initial_boundary_calls = client.boundary_calls
        stage1["experiments"][0]["input"] = "Corrected field-only value."
        with patch(
            "generation_pipeline.stage1_verifier.parse_pdf_document",
            return_value=document,
        ):
            refined_report = verify_stage1_inventory(
                stage1,
                Path("paper.pdf"),
                client,
                workers=2,
                boundary_baseline=report,
            )
        self.assertEqual(client.boundary_calls, initial_boundary_calls)
        self.assertTrue(
            refined_report["window_audit"]["reused_for_field_refinement"]
        )
        self.assertEqual(refined_report["overall"], "pass")

    def test_compiler_discovers_and_extracts_studies_without_full_document_call(self):
        filler = "irrelevant filler " * 550
        document = _document(
            [
                "TITLE_MARKER STUDY_ONE_MARKER Study 1 method and participant task. " + filler,
                filler,
                filler,
                "STUDY_TWO_MARKER Study 2 method and participant ranking task. " + filler,
                filler,
            ]
        )
        client = _CompilerClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "generation_pipeline.stage1_compiler.parse_pdf_document",
                return_value=document,
            ):
                result = compile_stage1_inventory(
                    Path("paper.pdf"),
                    client,
                    artifacts_dir=Path(tmpdir) / "stage1",
                    workers=2,
                )

        self.assertEqual([item["study_id"] for item in result["experiments"]], ["study_1", "study_2"])
        self.assertEqual(result["stage1_evidence"]["full_document_llm_calls"], 0)
        self.assertTrue(result["stage1_evidence"]["all_mentions_assigned"])
        self.assertTrue(result["stage1_evidence"]["extraction_complete"])
        self.assertTrue(result["stage1_evidence"]["all_comparison_relations_resolved"])
        discovery_prompts = [
            prompt for prompt in client.prompts if "high-recall discovery pass" in prompt
        ]
        self.assertGreater(len(discovery_prompts), 1)
        self.assertTrue(all(len(prompt) < document.text_chars for prompt in discovery_prompts))
        self.assertFalse(
            any("STUDY_ONE_MARKER" in prompt and "STUDY_TWO_MARKER" in prompt for prompt in discovery_prompts)
        )

    def test_window_verifier_never_uses_a_full_document_request(self):
        filler = "verification filler " * 600
        document = _document(
            [
                "STUDY_ONE_MARKER Study 1 method. " + filler,
                filler,
                "STUDY_TWO_MARKER Study 2 method. " + filler,
            ]
        )
        stage1 = {
            "experiments": [
                {"study_id": "study_1", "experiment_id": "Study 1", "replicable": "YES"},
                {"study_id": "study_2", "experiment_id": "Study 2", "replicable": "YES"},
            ]
        }
        client = _CompilerClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "generation_pipeline.stage1_verifier.parse_pdf_document",
                return_value=document,
            ):
                report = verify_stage1_inventory(
                    stage1,
                    Path("paper.pdf"),
                    client,
                    artifacts_dir=Path(tmpdir) / "verifier",
                    workers=2,
                )

        self.assertEqual(report["overall"], "pass")
        self.assertEqual(report["window_audit"]["full_document_llm_calls"], 0)
        self.assertEqual(report["study_audit"]["full_document_llm_calls"], 0)
        self.assertEqual(report["study_audit"]["study_count"], 2)
        self.assertTrue(report["study_audit"]["all_cited_evidence_included"])
        verifier_prompts = [
            prompt
            for prompt in client.prompts
            if "independently auditing empirical-unit boundaries" in prompt
        ]
        self.assertGreater(len(verifier_prompts), 1)
        self.assertTrue(all(len(prompt) < document.text_chars for prompt in verifier_prompts))


if __name__ == "__main__":
    unittest.main()
