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
        instructions = trial_metadata.get("instructions", "")
        condition = trial_metadata.get("condition", "cat_sim")
        profile = trial_metadata.get("profile", {})
        items = trial_metadata.get("items", [])
        
        # Define condition-specific contexts based on the paper's experimental setup
        condition_contexts = {
            "cat_sim": (
                "You have been assigned to the 'Klee Group' based on your art preferences in Part 1. "
                "The other participants are also divided into the 'Klee Group' and the 'Kandinsky Group'. "
                "In the following tasks, you will allot money to two other people. One is a member of your own group (Klee Group), "
                "and the other is a member of the other group (Kandinsky Group)."
            ),
            "cat_non_sim": (
                "You have been assigned to 'Group X' based on a random toss of a coin. "
                "The other participants are also divided into 'Group X' and 'Group W'. "
                "In the following tasks, you will allot money to two other people. One is a member of your own group (Group X), "
                "and the other is a member of the other group (Group W)."
            ),
            "non_cat_sim": (
                "Based on your art preferences in Part 1, you have been assigned a code number in the 70s. "
                "Participants who preferred Klee were given numbers in the 70s, and those who preferred Kandinsky were given numbers in the 40s. "
                "In the following tasks, you will allot money to two other people. One is a person who, like you, preferred Klee (Code in 70s), "
                "and the other is a person who preferred Kandinsky (Code in 40s)."
            ),
            "non_cat_non_sim": (
                "You have been assigned a code number in the 70s based on a random toss of a coin. "
                "Some participants were randomly given numbers in the 70s, and others were given numbers in the 40s. "
                "In the following tasks, you will allot money to two other people. One is a person with a code number in the 70s, "
                "and the other is a person with a code number in the 40s."
            )
        }

        # Define recipient labels for rows based on condition
        # Top row is generally the 'ingroup/similar' person in these matrices for the study
        recipient_labels = {
            "cat_sim": ("Member of your group (Klee Group)", "Member of the other group (Kandinsky Group)"),
            "cat_non_sim": ("Member of your group (Group X)", "Member of the other group (Group W)"),
            "non_cat_sim": ("Person who preferred Klee (Code 70s)", "Person who preferred Kandinsky (Code 40s)"),
            "non_cat_non_sim": ("Person with Code 70s", "Person with Code 40s")
        }
        
        labels = recipient_labels.get(condition)

        prompt = f"Background: You are a {profile.get('age')}-year-old {profile.get('gender')} participant in this study.\n\n"
        prompt += f"{instructions}\n\n"
        prompt += f"Group Assignment Info: {condition_contexts.get(condition)}\n\n"
        prompt += "For each matrix below, select the box (e.g., Box 1, Box 2, etc.) that represents your preferred distribution.\n\n"

        q_idx = 1
        for item in items:
            prompt += f"Matrix {q_idx}:\n"
            prompt += f"Recipient 1 (Top Row): {labels[0]}\n"
            prompt += f"Recipient 2 (Bottom Row): {labels[1]}\n"
            prompt += f"Options:\n"
            for i, opt in enumerate(item["options"]):
                prompt += f"Box {i+1}: {opt}\n"
            prompt += "\n"
            item["q_idx"] = f"Q{q_idx}"
            q_idx += 1

        # Build response specs list (using direct numbers)
        response_specs = [f"Q{i+1}=<number>" for i in range(len(items))]
        
        prompt += f"\nRESPONSE_SPEC (MANDATORY FORMAT):\n"
        prompt += "- Output ONLY answer lines in the format: Qk=<number> (where number is the box number, e.g., Q1=7, Q2=13, Q3=1)\n"
        prompt += f"- Use this format for ALL questions: {', '.join(response_specs)}\n"
        prompt += f"- Expected lines: {len(items)}\n"
        prompt += "- IMPORTANT: Use ONLY the box number (e.g., Q1=7, NOT Q1=Box 7, NOT Q1=box7)\n"

        return prompt

class StudyStudy007Config(BaseStudyConfig):
    prompt_builder_class = CustomPromptBuilder
    PROMPT_VARIANT = "v1"

    def __init__(self, study_path: Path, specification: Dict[str, Any]):
        super().__init__(study_path, specification)

    def create_trials(self, n_trials=None):
        trials = []
        material = self.load_material("2x2_factorial_design")
        spec = self.load_specification()
        
        # Total participants
        n = spec["participants"]["n"]
        if n == 0:
            n = 75 # Standard sample size from the paper
            
        # The paper used a 2x2 design with roughly equal groups
        conditions = ["cat_sim", "cat_non_sim", "non_cat_sim", "non_cat_non_sim"]
        
        for i in range(n):
            condition = conditions[i % len(conditions)]
            # Participants were schoolboys aged 14-16
            age = random.randint(14, 16)
            gender = "male"
            
            trials.append({
                "sub_study_id": "2x2_factorial_design",
                "condition": condition,
                "profile": {"age": age, "gender": gender},
                "instructions": material["instructions"],
                "items": material["items"],
                "variant": self.PROMPT_VARIANT
            })
        return trials

    def parse_response(self, trial_metadata: Dict[str, Any], response: str) -> Dict[str, Any]:
        results = {}
        items = trial_metadata.get("items", [])
        
        for item in items:
            q_idx = item["q_idx"]
            # Extract the choice, e.g., "Box 7"
            choice = self.extract_choice(response, [f"Box {i+1}" for i in range(len(item["options"]))], q_idx)
            
            # Map the choice back to the index for scoring
            if choice and "Box" in choice:
                try:
                    idx = int(choice.split(" ")[1]) - 1
                    results[f"{item['id']}_index"] = idx
                    results[f"{item['id']}_choice"] = item["options"][idx]
                except (ValueError, IndexError):
                    results[f"{item['id']}_index"] = None
            else:
                results[f"{item['id']}_index"] = None
                
        return results

    def dump_prompts(self, output_dir):
        # Create one trial for each condition to demonstrate the prompt variations
        conditions = ["cat_sim", "cat_non_sim", "non_cat_sim", "non_cat_non_sim"]
        material = self.load_material("2x2_factorial_design")
        
        for idx, cond in enumerate(conditions):
            trial = {
                "sub_study_id": "2x2_factorial_design",
                "condition": cond,
                "profile": {"age": 15, "gender": "male"},
                "instructions": material["instructions"],
                "items": material["items"]
            }
            prompt = self.prompt_builder.build_trial_prompt(trial)
            with open(f"{output_dir}/study_007_{cond}_trial.txt", "w") as f:
                f.write(prompt)
