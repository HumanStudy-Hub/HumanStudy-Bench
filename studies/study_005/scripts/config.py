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
from pathlib import Path
from typing import Any, Dict, List

class CustomPromptBuilder(PromptBuilder):
    def __init__(self, study_path: Path):
        super().__init__(study_path)

    def build_trial_prompt(self, trial_metadata: Dict[str, Any]) -> str:
        """
        Builds the prompt for a single participant (trial).
        """
        items = trial_metadata.get("items", [])
        instructions = trial_metadata.get("instructions", "")
        
        prompt = "You are participating in a psychology study on judgment and decision-making. "
        prompt += "Please read the following description carefully and answer the questions that follow.\n\n"
        
        # Add the vignette/instructions
        prompt += f"{instructions}\n\n"
        
        response_spec_parts = []
        for i, item in enumerate(items):
            q_num = i + 1
            question_text = item["question"]
            
            # Determine response format for instructions and RESPONSE_SPEC
            if "options" in item and item["options"]:
                # Multiple choice (e.g., Yes/No)
                options_str = "/".join(item["options"])
                prompt += f"Q{q_num} ({options_str}): {question_text}\n"
                response_spec_parts.append(f"Q{q_num}=<{options_str}>")
            else:
                # Scale or numeric (e.g., 0-6 scale)
                prompt += f"Q{q_num} (number): {question_text}\n"
                response_spec_parts.append(f"Q{q_num}=<number>")
        
        # Add the RESPONSE_SPEC
        prompt += "\nRESPONSE_SPEC: " + ", ".join(response_spec_parts) + "\n"
        
        return prompt

class StudyStudy005Config(BaseStudyConfig):
    prompt_builder_class = CustomPromptBuilder
    PROMPT_VARIANT = "v1"

    def __init__(self, study_path: Path, specification: Dict[str, Any]):
        super().__init__(study_path, specification)

    def create_trials(self, n_trials: int = None) -> List[Dict[str, Any]]:
        """
        Creates trials based on the between-subjects design of Knobe (2003).
        Each participant is assigned to one of four conditions:
        Exp 1 (Harm/Help) or Exp 2 (Harm/Help).
        """
        trials = []
        spec = self.specification
        sub_study_ids = [
            "experiment_1_harm", 
            "experiment_1_help", 
            "experiment_2_harm", 
            "experiment_2_help"
        ]

        for sub_id in sub_study_ids:
            # Load material for the specific condition
            material = self.load_material(sub_id)
            
            # Get n for this sub-study from specification
            n = spec.get("participants", {}).get("by_sub_study", {}).get(sub_id, {}).get("n", 50)
            
            # If n_trials is provided, it overrides the specification (usually for testing/dumping)
            if n_trials is not None:
                n = n_trials
            
            # Ensure n is at least a default if it was 0 in spec
            if n == 0:
                n = 50

            for _ in range(n):
                # Each trial represents one participant seeing all items in their condition
                trials.append({
                    "sub_study_id": sub_id,
                    "instructions": material.get("instructions", ""),
                    "items": material.get("items", []),
                    "profile": {
                        "age": random.randint(18, 65),
                        "gender": random.choice(["male", "female"])
                    },
                    "variant": self.PROMPT_VARIANT
                })
        
        return trials

    def dump_prompts(self, output_dir: str):
        """
        Dumps one sample prompt for each sub-study condition.
        """
        # Create one trial per sub-study to show the variety
        sub_study_ids = [
            "experiment_1_harm", 
            "experiment_1_help", 
            "experiment_2_harm", 
            "experiment_2_help"
        ]
        
        for sub_id in sub_study_ids:
            material = self.load_material(sub_id)
            trial = {
                "sub_study_id": sub_id,
                "instructions": material.get("instructions", ""),
                "items": material.get("items", []),
                "variant": self.PROMPT_VARIANT
            }
            prompt = self.prompt_builder.build_trial_prompt(trial)
            
            output_path = Path(output_dir) / f"study_005_{sub_id}_sample.txt"
            with open(output_path, "w") as f:
                f.write(prompt)
