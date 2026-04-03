import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_utils import BaseStudyConfig, PromptBuilder


CONDITION_MATERIAL_MAP = {
    "control": "main_experiment_control",
    "decoy_target_first": "main_experiment_decoy_target_first",
    "decoy_decoy_first": "main_experiment_decoy_decoy_first",
}

CONDITION_WEIGHTS = {
    "control": 101,
    "decoy_target_first": 52,
    "decoy_decoy_first": 50,
}


class DecoyEffectPromptBuilder(PromptBuilder):

    def build_trial_prompt(self, trial_metadata: Dict[str, Any]) -> str:
        condition = trial_metadata["condition"]
        material = trial_metadata["material"]
        invitation = material["invitation_email"]

        prompt = (
            "You are a university student who previously signed up for a research study. "
            "You completed a preliminary questionnaire about your attitudes and provided your "
            "email address to receive an invitation to a follow-up survey.\n\n"
            "You have now received the following email invitation:\n\n"
            "---\n"
        )

        prompt += f"Subject: {invitation['subject']}\n\n"
        prompt += f"{invitation['body']}\n"

        if condition == "control":
            prompt += (
                "\n---\n\n"
                "Based on this invitation, do you choose to complete the survey?\n\n"
                "RESPONSE_SPEC: Output CHOICE=<COMPLETE/DECLINE>\n"
            )
        else:
            table = material["comparison_table"]
            prompt += "\n\nHere is a comparison of the two available questionnaires:\n\n"

            for i, col in enumerate(table["columns"]):
                label = chr(65 + i)
                prompt += f"Option {label}: {col['title']}\n"
                prompt += f"  - Question type: {col['question_type']}\n"
                prompt += f"  - Reward: {col['reward']}\n"
                prompt += f"  - Reward timing: {col['reward_timing']}\n\n"

            prompt += f"{table['footer']}\n"
            prompt += (
                "\n---\n\n"
                "Based on this invitation, which option do you choose?\n"
                "  A) Complete Option A\n"
                "  B) Complete Option B\n"
                "  C) Decline to participate\n\n"
                "RESPONSE_SPEC: Output CHOICE=<A/B/C>\n"
            )

        return prompt


class StudyConfig(BaseStudyConfig):
    prompt_builder_class = DecoyEffectPromptBuilder
    PROMPT_VARIANT = "v1"

    def create_trials(self, n_trials: Optional[int] = None) -> List[Dict[str, Any]]:
        spec = self.load_specification()

        total_n = spec.get("participants", {}).get("main_experiment", {}).get("n", 0)
        if n_trials is not None:
            total_n = n_trials
        if total_n == 0:
            total_n = 50

        total_weight = sum(CONDITION_WEIGHTS.values())
        condition_ns = {}
        assigned = 0
        conditions = list(CONDITION_WEIGHTS.keys())
        for i, cond in enumerate(conditions):
            if i == len(conditions) - 1:
                condition_ns[cond] = total_n - assigned
            else:
                n_cond = round(total_n * CONDITION_WEIGHTS[cond] / total_weight)
                condition_ns[cond] = n_cond
                assigned += n_cond

        materials = {}
        for cond, mat_id in CONDITION_MATERIAL_MAP.items():
            materials[cond] = self.load_material(mat_id)

        trials = []
        trial_id = 0
        for cond in conditions:
            mat = materials[cond]
            for _ in range(condition_ns[cond]):
                trials.append({
                    "trial_id": trial_id,
                    "sub_study_id": CONDITION_MATERIAL_MAP[cond],
                    "condition": cond,
                    "material": mat,
                    "scenario_id": f"decoy_effect_{cond}",
                    "items": mat.get("items", mat.get("target_questionnaire", {}).get("items", [])),
                    "variant": self.PROMPT_VARIANT,
                })
                trial_id += 1

        random.shuffle(trials)
        for i, t in enumerate(trials):
            t["trial_id"] = i

        return trials
