import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from generation_pipeline.extractors.study_data_extractor import StudyDataExtractor
from generation_pipeline.pdf.models import DocumentBlock, ParsedPdfDocument
from generation_pipeline.stage2_verifier import verify_stage2_findings


def _document():
    texts = [
        "STUDY_ONE_METHOD Study 1 participants chose between two frames.",
        "STUDY_ONE_RESULT Study 1 results reported a significant choice difference.",
        "Neutral transition text without study results.",
        "STUDY_TWO_METHOD Study 2 participants ranked three alternatives.",
        "STUDY_TWO_RESULT Study 2 results reported a significant mean rank difference.",
    ]
    return ParsedPdfDocument(
        source_file="paper.pdf",
        source_sha256="stage2hash",
        parser="test_parser",
        parser_version="1",
        page_count=5,
        blocks=[
            DocumentBlock(
                block_id=f"p{index:03d}_text",
                order=index - 1,
                page_start=index,
                page_end=index,
                block_type="text",
                text=text,
                section_path=["Method" if index in {1, 4} else "Results"],
            )
            for index, text in enumerate(texts, start=1)
        ],
    )


def _stage1():
    experiments = []
    contexts = {}
    for number, refs, task, output in (
        (1, ["p001_text", "p002_text"], "Choose a framed option", "Choice proportion"),
        (2, ["p004_text", "p005_text"], "Rank three alternatives", "Mean rank"),
    ):
        study_id = f"study_{number}"
        experiments.append(
            {
                "experiment_id": f"Study {number}",
                "study_id": study_id,
                "experiment_name": task,
                "study_name": task,
                "design_type": "between-subjects",
                "conditions_or_factors": ["condition: A vs B"],
                "input": task,
                "participant_task": task,
                "participants": "N = 100 adults",
                "output": output,
                "candidate_source_hints": [],
                "replicable": "YES",
                "exclusion_reasons": [],
                "evidence_refs": refs,
            }
        )
        contexts[study_id] = {"block_ids": refs}
    return {
        "paper_id": "paper",
        "paper_title": "Two Study Paper",
        "experiments": experiments,
        "stage1_evidence": {"study_contexts": contexts},
    }


class _Stage2Client:
    model = "fake-model"

    def __init__(self):
        self.prompts = []

    def generate_content(self, prompt, **kwargs):
        del kwargs
        self.prompts.append(prompt)
        if "Extract per-effect records" in prompt:
            study_id = re.search(r'"study_id": "(study_[12])"', prompt).group(1)
            number = int(study_id[-1])
            valid_ids = json.loads(
                prompt.split("VALID BLOCK IDS:\n", 1)[1].split("\n\nEVIDENCE CONTEXT", 1)[0]
            )
            return json.dumps(
                {
                    "paper_title": "Two Study Paper",
                    "paper_metadata": {"authors": ["A. Author"], "year": 2020},
                    "eligible_studies": [
                        {
                            "study": f"Study {number}",
                            "study_id": study_id,
                            "eligibility_rationale": "Stage 1 candidate",
                            "sample": {
                                "total_n": 100,
                                "analyzed_n": None,
                                "mean_age": None,
                                "female_percent": None,
                                "male_percent": None,
                                "platform": "Online",
                                "country": None,
                                "inclusion_criteria": None,
                                "exclusion_criteria": None,
                                "notes": None,
                            },
                            "effects": [
                                {
                                    "platform": "Online",
                                    "effecttype": "main",
                                    "IV": "condition",
                                    "DV": "choice" if number == 1 else "rank",
                                    "size": 100,
                                    "direction": "pos",
                                    "mean_group1": None,
                                    "sd_group1": None,
                                    "mean_group2": None,
                                    "sd_group2": None,
                                    "stats": {
                                        "B": None,
                                        "b": None,
                                        "chi_square": None,
                                        "D": None,
                                        "eta_square": None,
                                        "f": None,
                                        "t": 2.5,
                                        "z": None,
                                        "ci": [None, None],
                                        "p_value": ".02",
                                        "sig": "sig",
                                    },
                                    "materials_notes": "Method section",
                                    "table_or_page_location": f"Study {number} Results",
                                    "evidence_refs": valid_ids[-1:],
                                    "materials": {"status": None, "content": None},
                                    "manipulation": {"status": None, "content": None},
                                    "items": {"status": None, "content": None},
                                }
                            ],
                        }
                    ],
                }
            )
        if "verifying one study extraction" in prompt:
            study_id = re.search(r'"study_id": "(study_[12])"', prompt).group(1)
            valid_ids = json.loads(
                prompt.split("Valid evidence block IDs: ", 1)[1].split("\n", 1)[0]
            )
            return json.dumps(
                {
                    "overall": "pass",
                    "confidence": 0.9,
                    "study_coverage": [
                        {
                            "study": study_id,
                            "verdict": "ok",
                            "issue": "",
                            "evidence": "Method and result are present.",
                            "evidence_block_ids": valid_ids[:1],
                        }
                    ],
                    "finding_checks": [],
                    "regeneration_instructions": {
                        "missing_effects": [],
                        "exact_stats_needed": [],
                        "data_corrections": [],
                    },
                    "notes": "Supported.",
                }
            )
        raise AssertionError(f"Unexpected prompt: {prompt[:120]}")


class Stage2EvidenceTests(unittest.TestCase):
    def test_stage2_extracts_each_candidate_from_its_own_context(self):
        document = _document()
        stage1 = _stage1()
        client = _Stage2Client()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "generation_pipeline.extractors.study_data_extractor.parse_pdf_document",
                return_value=document,
            ):
                result = StudyDataExtractor(client).process(
                    stage1,
                    Path("paper.pdf"),
                    artifacts_dir=Path(tmpdir) / "stage2",
                    extraction_workers=2,
                )

        self.assertEqual(
            [study["study_id"] for study in result["eligible_studies"]],
            ["study_1", "study_2"],
        )
        self.assertEqual(result["stage2_evidence"]["full_document_llm_calls"], 0)
        extraction_prompts = [prompt for prompt in client.prompts if "Extract per-effect" in prompt]
        self.assertEqual(len(extraction_prompts), 2)
        study1_prompt = next(
            prompt
            for prompt in extraction_prompts
            if '"experiment_name": "Choose a framed option"' in prompt
        )
        study2_prompt = next(
            prompt
            for prompt in extraction_prompts
            if '"experiment_name": "Rank three alternatives"' in prompt
        )
        self.assertNotIn("STUDY_TWO_METHOD", study1_prompt)
        self.assertNotIn("STUDY_ONE_METHOD", study2_prompt)

    def test_stage2_verifier_uses_per_study_evidence(self):
        document = _document()
        stage1 = _stage1()
        extraction_client = _Stage2Client()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "generation_pipeline.extractors.study_data_extractor.parse_pdf_document",
                return_value=document,
            ):
                stage2 = StudyDataExtractor(extraction_client).process(
                    stage1,
                    Path("paper.pdf"),
                    artifacts_dir=Path(tmpdir) / "stage2",
                    extraction_workers=2,
                )
            verifier_client = _Stage2Client()
            with patch(
                "generation_pipeline.stage2_verifier.parse_pdf_document",
                return_value=document,
            ):
                report = verify_stage2_findings(
                    stage2,
                    stage1,
                    Path("paper.pdf"),
                    verifier_client,
                    artifacts_dir=Path(tmpdir) / "verifier",
                    workers=2,
                )

        self.assertEqual(report["overall"], "pass")
        self.assertEqual(report["evidence_audit"]["full_document_llm_calls"], 0)
        verifier_prompts = [
            prompt for prompt in verifier_client.prompts if "verifying one study extraction" in prompt
        ]
        self.assertEqual(len(verifier_prompts), 2)


if __name__ == "__main__":
    unittest.main()
