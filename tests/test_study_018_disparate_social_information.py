import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from generation_pipeline.stage5 import Stage5Options, run_stage5
from src.agents.llm_participant_agent import LLMParticipantAgent
from src.core.study import Study
from src.core.study_config import get_study_config
from src.llm.anthropic_client import _content_to_anthropic
from src.llm.openai_client import _normalize_content


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_PATH = REPO_ROOT / "studies" / "study_018"


def load_evaluator():
    path = STUDY_PATH / "scripts" / "evaluator.py"
    spec = importlib.util.spec_from_file_location("study_018_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_stimulus_generator():
    path = STUDY_PATH / "source" / "stimuli" / "generate_stimuli.py"
    spec = importlib.util.spec_from_file_location("study_018_stimuli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ScriptedParticipant:
    def __init__(self):
        self.messages = []
        self.starts = 0
        self.clears = 0

    def start_conversation(self):
        self.starts += 1

    def clear_conversation(self):
        self.clears += 1

    def continue_conversation(self, message, max_tokens=24):
        del max_tokens
        self.messages.append(message)
        if isinstance(message, str) and "ANSWERS=" in message:
            response_text = "ANSWERS=C,I,C,C"
        else:
            estimate = 60 if isinstance(message, list) else 52
            response_text = f"ESTIMATE={estimate}"
        return {
            "response_text": response_text,
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "cost": 0.0,
            },
        }


class FailingComprehensionParticipant(ScriptedParticipant):
    def continue_conversation(self, message, max_tokens=24):
        if isinstance(message, str) and (
            "ANSWERS=" in message or "statement(s)" in message
        ):
            self.messages.append(message)
            return {
                "response_text": "ANSWERS=C,C,C,C",
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                    "cost": 0.0,
                },
            }
        return super().continue_conversation(message, max_tokens=max_tokens)


class DisparateSocialInformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = Study.load(STUDY_PATH)
        cls.config = get_study_config("study_018", STUDY_PATH, cls.study.specification)
        cls.lookup = cls.config.load_material("peer_lookup")

    def test_package_loads_and_scope_is_explicit(self):
        self.assertTrue(self.study.validate())
        self.assertEqual(self.study.metadata["doi"], "10.1098/rspb.2020.2413")
        self.assertEqual(
            self.study.metadata["journal"],
            "Proceedings of the Royal Society B: Biological Sciences",
        )
        self.assertEqual(
            self.study.metadata["implementation_scope"]["status"],
            "complete_three_block_behavioral_task",
        )
        self.assertEqual(self.study.specification["participants"]["n"], 95)
        self.assertTrue(
            self.study.specification["runtime"]["requires_vision_capable_model"]
        )

    def test_published_schedule_and_exact_lookup_examples(self):
        rounds = self.lookup["main_task"]["rounds"]
        self.assertEqual(len(rounds), 30)
        self.assertEqual(
            Counter(row["condition"] for row in rounds),
            Counter({"LN": 5, "HN": 5, "HF": 5, "HC": 5, "filler": 10}),
        )
        self.assertEqual(
            self.config._peer_estimates(
                self.lookup["main_task"],
                round_index=11,
                initial_estimate=60,
            ),
            [45, 48, 51],
        )
        self.assertEqual(
            self.config._peer_estimates(
                self.lookup["four_peer_control"],
                round_index=0,
                initial_estimate=60,
            ),
            [62, 71, 80],
        )
        self.assertEqual(
            self.config._sample_one_peer(
                self.lookup["one_peer_control"],
                round_index=0,
                initial_estimate=60,
                rng=__import__("random").Random(3),
            )[0],
            72,
        )
        self.assertEqual(
            self.lookup["one_peer_control"]["source_pool_lengths"],
            [96, 96, 96, 96, 96],
        )
        for block_name in ("main_task", "four_peer_control"):
            block = self.lookup[block_name]
            self.assertTrue(
                all(
                    len(row) == 150
                    for peer_name in ("p1", "p2", "p3")
                    for row in block[peer_name]
                )
            )

    def test_stimuli_exist_without_counts_in_filenames(self):
        manifest = self.config.load_material("stimulus_manifest")
        self.assertEqual(len(manifest["items"]), 35)
        self.assertEqual(manifest["background_rgb"], [204, 204, 255])
        self.assertEqual(manifest["source_period_seed"], 1)
        lookup_by_block_and_round = {
            (block, int(row["round"])): row
            for block in ("one_peer_control", "main_task")
            for row in self.lookup[block]["rounds"]
        }
        for item in manifest["items"]:
            round_number = int(item["round"])
            filename = str(item["file"])
            true_count = str(
                lookup_by_block_and_round[
                    (str(item["block"]), round_number)
                ]["true_count"]
            )
            self.assertNotIn(true_count, filename)
            image_path = STUDY_PATH / "source" / "stimuli" / filename
            self.assertTrue(image_path.exists())
            with Image.open(image_path) as image:
                self.assertEqual(image.size, (1010, 590))

    def test_stimulus_positions_match_original_javascript_sort_semantics(self):
        positions = load_stimulus_generator()._original_positions(count=98)
        self.assertEqual(
            positions[:6],
            [
                (570, 54),
                (600, 54),
                (690, 54),
                (360, 0),
                (180, 72),
                (690, 72),
            ],
        )
        self.assertEqual(positions[-1], (780, 522))

    def test_multimodal_prompt_converts_for_supported_providers(self):
        image_path = (
            STUDY_PATH / "source" / "stimuli" / "round_01_ant.png"
        )
        content = self.config.prompt_builder.build_initial_prompt(
            round_number=1,
            species="ant",
            image_path=image_path,
        )
        openai_content = _normalize_content(content)
        self.assertEqual(openai_content[0]["type"], "text")
        self.assertEqual(openai_content[1]["type"], "image_url")
        self.assertTrue(
            openai_content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        )
        anthropic_content = _content_to_anthropic(content)
        self.assertEqual(anthropic_content[1]["type"], "image")
        self.assertEqual(
            anthropic_content[1]["source"]["media_type"],
            "image/png",
        )

    def test_participant_agent_sends_openai_multimodal_history(self):
        image_path = (
            STUDY_PATH / "source" / "stimuli" / "round_01_ant.png"
        )
        content = self.config.prompt_builder.build_initial_prompt(
            round_number=1,
            species="ant",
            image_path=image_path,
        )
        api_message = SimpleNamespace(
            content="ESTIMATE=42",
            role="assistant",
            reasoning=None,
            reasoning_details=None,
        )
        api_choice = SimpleNamespace(
            index=0,
            message=api_message,
            finish_reason="stop",
        )
        api_response = SimpleNamespace(
            choices=[api_choice],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
            ),
        )
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = api_response

        with patch("openai.OpenAI", return_value=fake_client):
            agent = LLMParticipantAgent(
                participant_id=0,
                profile={},
                model="gpt-4o",
                api_key="test-key",
                use_real_llm=True,
                system_prompt_preset="empty",
            )
            agent.start_conversation("test system")
            result = agent.continue_conversation(content, max_tokens=24)

        self.assertEqual(result["response_text"], "ESTIMATE=42")
        sent = fake_client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(sent[1]["content"][0]["type"], "text")
        self.assertEqual(sent[1]["content"][1]["type"], "image_url")
        self.assertTrue(
            sent[1]["content"][1]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
        )

    def test_main_round_removes_image_before_social_revision(self):
        participant = ScriptedParticipant()
        response = self.config._run_main_round(
            participant=participant,
            prompt_builder=self.config.prompt_builder,
            rng=__import__("random").Random(3),
            participant_id=0,
            global_trial_number=1,
            round_data=self.lookup["main_task"]["rounds"][0],
            main_lookup=self.lookup["main_task"],
        )
        self.assertEqual(participant.starts, 2)
        self.assertEqual(participant.clears, 2)
        self.assertIsInstance(participant.messages[0], list)
        self.assertIsInstance(participant.messages[1], str)
        self.assertEqual(response["trial_info"]["initial_estimate"], 60)
        self.assertEqual(response["response"], 52)
        visible = " ".join(
            response["trial_info"]["agent_visible_prompts"].values()
        )
        self.assertNotIn("98", visible)
        self.assertFalse(
            response["trial_info"]["answer_revealed_before_response"]
        )

    def test_one_peer_round_uses_source_rule_and_removes_image(self):
        participant = ScriptedParticipant()
        response = self.config._run_one_peer_round(
            participant=participant,
            prompt_builder=self.config.prompt_builder,
            rng=__import__("random").Random(3),
            participant_id=0,
            global_trial_number=1,
            round_data=self.lookup["one_peer_control"]["rounds"][0],
            one_peer_lookup=self.lookup["one_peer_control"],
        )
        self.assertEqual(participant.starts, 2)
        self.assertEqual(participant.clears, 2)
        self.assertIsInstance(participant.messages[0], list)
        self.assertIsInstance(participant.messages[1], str)
        self.assertEqual(response["trial_info"]["initial_estimate"], 60)
        self.assertEqual(response["trial_info"]["peer_estimates"], [72])
        self.assertEqual(
            response["trial_info"]["one_peer_selection"]["target_multiplier"],
            1.2,
        )
        self.assertNotIn(
            str(response["correct_answer"]),
            " ".join(response["trial_info"]["agent_visible_prompts"].values()),
        )

    def test_source_comprehension_checks_are_gated(self):
        material = self.config.load_material("disparate_social_information")
        participant = ScriptedParticipant()
        record = self.config._run_comprehension_check(
            participant=participant,
            prompt_builder=self.config.prompt_builder,
            block_name="one_peer_control",
            instructions=material["block_instructions"]["one_peer_control"],
            quiz=material["comprehension_checks"]["one_peer_control"],
        )
        self.assertTrue(record["passed"])
        self.assertEqual(record["attempts"], 1)
        self.assertEqual(record["answers"], [True, False, True, True])

    def test_comprehension_failure_terminates_before_block(self):
        trial = self.config.create_trials(n_trials=1)[0]
        participant = FailingComprehensionParticipant()
        result = self.config._run_one_participant(
            trial,
            participant=participant,
            profile={"participant_id": 0},
            prompt_builder=self.config.prompt_builder,
            base_seed=13,
            lookup=self.lookup,
        )
        self.assertTrue(result["terminated_early"])
        self.assertEqual(
            result["termination_reason"],
            "failed_comprehension_check:one_peer_control",
        )
        self.assertEqual(result["responses"], [])
        self.assertIsNone(result["payment"])
        self.assertEqual(result["comprehension_checks"][0]["attempts"], 3)

    def test_scripted_participant_completes_all_three_blocks(self):
        trial = self.config.create_trials(n_trials=1)[0]
        participant = ScriptedParticipant()
        result = self.config._run_one_participant(
            trial,
            participant=participant,
            profile={"participant_id": 0},
            prompt_builder=self.config.prompt_builder,
            base_seed=13,
            lookup=self.lookup,
        )
        self.assertFalse(result["terminated_early"])
        self.assertEqual(len(result["responses"]), 40)
        self.assertEqual(
            result["profile"]["block_order"],
            ["one_peer_control", "main_task", "four_peer_control"],
        )
        self.assertEqual(len(result["comprehension_checks"]), 3)
        self.assertTrue(all(check["passed"] for check in result["comprehension_checks"]))
        self.assertEqual(participant.starts, 78)
        self.assertEqual(participant.clears, 79)

    def test_zero_control_anchors_are_rejected(self):
        pools = self.lookup["four_peer_control"]["anchor_pools"]
        self.assertEqual(sum(value == 0 for pool in pools for value in pool), 2)
        self.assertEqual(
            self.lookup["four_peer_control"]["invalid_anchor_policy"].startswith(
                "The published pools contain two zero sentinels"
            ),
            True,
        )

    def test_mock_stage5_runs_complete_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_stage5(
                STUDY_PATH,
                runs_dir=Path(temp_dir),
                models=["mock"],
                options=Stage5Options(
                    n_participants=95,
                    mock=True,
                    seed=17,
                ),
                data_dir=REPO_ROOT,
            )
            output_path = Path(summary["runs"][0]["output_path"])
            output = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(output["individual_data"]), 95)
            self.assertEqual(
                sum(
                    len(participant["responses"])
                    for participant in output["individual_data"]
                ),
                3800,
            )
            self.assertEqual(
                output["descriptive_statistics"]["main_task_responses"],
                2850,
            )
            self.assertEqual(
                output["descriptive_statistics"]["one_peer_control_responses"],
                475,
            )
            self.assertEqual(
                output["descriptive_statistics"]["four_peer_control_responses"],
                475,
            )
            self.assertEqual(
                output["descriptive_statistics"]["comprehension_checks_passed"],
                285,
            )
            self.assertEqual(output["descriptive_statistics"]["parse_failures"], 0)
            self.assertTrue(
                all(
                    participant["profile"]["block_order_code"]
                    == participant["participant_id"] % 6 + 1
                    and len(participant["comprehension_checks"]) == 3
                    and all(
                        check["passed"]
                        for check in participant["comprehension_checks"]
                    )
                    for participant in output["individual_data"]
                )
            )
            self.assertTrue(
                all(
                    response["trial_info"]["study_type"]
                    == self.study.specification["study_type"]
                    for participant in output["individual_data"]
                    for response in participant["responses"]
                )
            )

            evaluation = load_evaluator().evaluate_study(output)
            self.assertEqual(evaluation["execution_score"], 1.0)
            self.assertTrue(evaluation["passed"])
            self.assertTrue(
                all(
                    test["passed"]
                    for test in evaluation["test_results"]
                )
            )


if __name__ == "__main__":
    unittest.main()
