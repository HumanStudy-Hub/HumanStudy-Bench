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
        Builds the prompt for a single participant based on the sub-study.
        Each participant receives all items for their assigned sub-study in a single trial.
        """
        instructions = trial_metadata.get("instructions", "")
        items = trial_metadata.get("items", [])
        variant = trial_metadata.get("variant", "v1")
        
        prompt = f"{instructions}\n\n"

        q_indices = []
        for idx, item in enumerate(items):
            q_idx = f"Q{idx + 1}"
            q_indices.append(q_idx)
            
            # Format multiple choice options as A/B/C
            if item.get("type") == "multiple_choice" and "options" in item:
                options = item.get("options", [])
                option_letters = [chr(65 + i) for i in range(len(options))]  # A, B, C, ...
                options_str = "\n".join([f"{letter}) {opt}" for letter, opt in zip(option_letters, options)])
                prompt += f"{q_idx}: {item['question']}\n{options_str}\n\n"
                # Store option mapping for response parsing
                item["option_map"] = {letter: opt for letter, opt in zip(option_letters, options)}
                item["option_letters"] = option_letters
            else:
                # For non-multiple choice items, use original format
                options_str = "\n".join([f"- {opt}" for opt in item.get("options", [])])
                prompt += f"{q_idx}: {item['question']}\nOptions:\n{options_str}\n\n"
        
        # Define the expected response format
        response_specs = []
        for idx, item in enumerate(items):
            q_idx = f"Q{idx + 1}"
            if item.get("type") == "multiple_choice" and "option_letters" in item:
                # Format as Q1=<A/B/C>
                response_specs.append(f"{q_idx}=<{'/'.join(item['option_letters'])}>")
            else:
                response_specs.append(f"{q_idx}=<choice>")
        
        prompt += f"RESPONSE_SPEC: Please provide your answers in the following format: {', '.join(response_specs)}\n"
        
        return prompt

class StudyStudy010Config(BaseStudyConfig):
    prompt_builder_class = CustomPromptBuilder
    PROMPT_VARIANT = "v1"

    def __init__(self, study_path: Path, specification: Dict[str, Any]):
        super().__init__(study_path, specification)

    def create_trials(self, n_trials: int = None) -> List[Dict[str, Any]]:
        trials = []
        sub_studies = ["pd_triad_tasks", "newcombs_computer_task", "pd_info_seeking_variation"]
        
        # Map sub_study_id to specification experiment names
        sub_to_experiment = {
            "pd_triad_tasks": "Experiment 1",
            "newcombs_computer_task": "Experiment 2",
            "pd_info_seeking_variation": "Experiment 3"
        }
        
        # Default sample sizes based on human experiments
        default_ns = {
            "pd_triad_tasks": 80,  # Experiment 1: 80 participants
            "newcombs_computer_task": 40,  # Experiment 2: 40 participants
            "pd_info_seeking_variation": 80  # Experiment 3: assume similar to Exp 1
        }
        
        # Load participant counts from specification
        spec = self.load_specification()
        n_by_sub = spec.get("participants", {}).get("by_sub_study", {})
        
        for sub_id in sub_studies:
            material = self.load_material(sub_id)
            
            # Determine n for this sub-study
            if n_trials is not None:
                n = n_trials
            else:
                # Try to get from specification using experiment name
                exp_name = sub_to_experiment.get(sub_id)
                if exp_name and exp_name in n_by_sub:
                    n = n_by_sub[exp_name].get("n", 0)
                else:
                    n = 0
                
                # If not found or 0, use default based on human experiment
                if n == 0:
                    n = default_ns.get(sub_id, 50)

            for _ in range(n):
                # In this study, a 'trial' is a single participant completing a whole task set
                trials.append({
                    "sub_study_id": sub_id,
                    "instructions": material.get("instructions", ""),
                    "items": material.get("items", []),
                    "variant": self.PROMPT_VARIANT
                })
        
        return trials

    def parse_responses(self, trial_metadata: Dict[str, Any], response_text: str) -> Dict[str, Any]:
        items = trial_metadata.get("items", [])
        results = {}
        
        # First parse Q1=value, Q2=value format
        parsed_responses = {}
        pattern = re.compile(r"(Q\d+(?:\.\d+)?)\s*=\s*([^,\n\s]+)")
        for k, v in pattern.findall(response_text):
            parsed_responses[k.strip()] = v.strip()
        
        for idx, item in enumerate(items):
            q_idx = f"Q{idx + 1}"
            item_id = item.get("id")
            
            # Get the value for this Q index
            choice_text = parsed_responses.get(q_idx, "")
            
            if choice_text:
                # For multiple choice, extract_choice can handle both A/B/C and option text
                choice_idx = self.extract_choice(choice_text, item.get("options", []))
                if choice_idx is not None:
                    # Return the actual option text, not the index
                    options = item.get("options", [])
                    if choice_idx < len(options):
                        results[item_id] = options[choice_idx]
                    else:
                        results[item_id] = choice_text  # Fallback to raw text
                else:
                    results[item_id] = choice_text  # Fallback to raw text if extraction fails
        
        return results

    def dump_prompts(self, output_dir: str):
        # Create one sample prompt for each sub-study
        sub_studies = ["pd_triad_tasks", "newcombs_computer_task", "pd_info_seeking_variation"]
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for sub_id in sub_studies:
            material = self.load_material(sub_id)
            trial = {
                "sub_study_id": sub_id,
                "instructions": material.get("instructions", ""),
                "items": material.get("items", []),
                "variant": self.PROMPT_VARIANT
            }
            prompt = self.prompt_builder.build_trial_prompt(trial)
            with open(output_path / f"study_010_{sub_id}_prompt.txt", "w") as f:
                f.write(prompt)
