import json
import re
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Any, List, Optional

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from study_utils import BaseStudyConfig, PromptBuilder

import random
from typing import Any, Dict, List
from pathlib import Path

class CustomPromptBuilder(PromptBuilder):
    def __init__(self, study_path: Path):
        super().__init__(study_path)

    def build_trial_prompt(self, trial_metadata: Dict[str, Any]) -> str:
        """
        Builds a prompt for the proposer in the bargaining games.
        Each participant (trial) receives instructions and a single decision task.
        """
        instructions = trial_metadata.get("instructions", "")
        items = trial_metadata.get("items", [])
        
        prompt = f"{instructions}\n\n"
        prompt += "Decision Task:\n"
        
        q_counter = 1
        for item in items:
            prompt += f"Q{q_counter}: {item['question']}\n"
            item["q_idx"] = q_counter
            q_counter += 1
            
        prompt += "\nRESPONSE_SPEC: Provide your response in the format Q1=<amount>. For example, Q1=2.50. Ensure the amount is a numeric value between 0 and the total amount available to divide.\n"
        return prompt

class StudyStudy011Config(BaseStudyConfig):
    prompt_builder_class = CustomPromptBuilder
    PROMPT_VARIANT = "v1"

    def __init__(self, study_path: Path, specification: Dict[str, Any]):
        super().__init__(study_path, specification)

    def create_trials(self, n_trials: int = None) -> List[Dict[str, Any]]:
        """
        Creates one trial per participant for each sub-study.
        Following the human design: Proposers make a single one-shot offer.
        """
        trials = []
        spec = self.load_specification()
        
        # Map sub_study_id from specification to specific material files containing participant instructions
        sub_study_map = {
            "DG-P_April_Sept": "experiment_1_dictator_game_pay_5",
            "UG-P_April_Sept": "experiment_2_ultimatum_game_pay_5",
            "DG-NP_April_Sept": "experiment_3_dictator_game_no_pay_5",
            "UG-NP_April_Sept": "experiment_4_ultimatum_game_no_pay_5",
            "DG-P_10_dollars": "experiment_5_dictator_game_pay_10",
            "UG-P_10_dollars": "experiment_6_ultimatum_game_pay_10"
        }
        
        # Target sample sizes based on the human experimental sessions
        default_ns = {
            "DG-P_April_Sept": 45,
            "UG-P_April_Sept": 43,
            "DG-NP_April_Sept": 46,
            "UG-NP_April_Sept": 48,
            "DG-P_10_dollars": 24,
            "UG-P_10_dollars": 24
        }

        for sub_study_id, material_filename in sub_study_map.items():
            material = self.load_material(material_filename)
            
            # Determine number of participants for this sub-study
            n = spec.get("participants", {}).get("by_sub_study", {}).get(sub_study_id, {}).get("n", 0)
            if n == 0:
                n = default_ns.get(sub_study_id, 50)
            
            # Allow global override if n_trials is provided
            if n_trials is not None:
                n = n_trials

            for _ in range(n):
                # Each participant is a unique trial
                trials.append({
                    "sub_study_id": sub_study_id,
                    "material_id": material_filename,
                    "instructions": material.get("instructions", ""),
                    "items": material.get("items", []),
                    "variant": self.PROMPT_VARIANT
                })
        
        return trials

    def parse_response(self, response_text: str, trial_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses the dollar amount offered to the receiver.
        """
        results = {}
        for item in trial_metadata.get("items", []):
            q_idx = item.get("q_idx")
            # Extract amount using the specified Q1= format
            val = self.extract_numeric(response_text, f"Q{q_idx}=")
            if val is None:
                # Fallback to finding the first numeric value if prefix matching fails
                val = self.extract_numeric(response_text)
            results[item["id"]] = val
        return results

    def dump_prompts(self, output_dir: str):
        """
        Dumps representative prompts for each sub-study condition.
        """
        sub_study_ids = [
            "DG-P_April_Sept", "UG-P_April_Sept", "DG-NP_April_Sept", 
            "UG-NP_April_Sept", "DG-P_10_dollars", "UG-P_10_dollars"
        ]
        
        sub_study_map = {
            "DG-P_April_Sept": "experiment_1_dictator_game_pay_5",
            "UG-P_April_Sept": "experiment_2_ultimatum_game_pay_5",
            "DG-NP_April_Sept": "experiment_3_dictator_game_no_pay_5",
            "UG-NP_April_Sept": "experiment_4_ultimatum_game_no_pay_5",
            "DG-P_10_dollars": "experiment_5_dictator_game_pay_10",
            "UG-P_10_dollars": "experiment_6_ultimatum_game_pay_10"
        }
        
        for sub_id in sub_study_ids:
            material = self.load_material(sub_study_map[sub_id])
            trial = {
                "sub_study_id": sub_id,
                "instructions": material.get("instructions", ""),
                "items": material.get("items", []),
                "variant": self.PROMPT_VARIANT
            }
            prompt = self.prompt_builder.build_trial_prompt(trial)
            output_path = Path(output_dir) / f"study_011_{sub_id}_prompt.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(prompt)
