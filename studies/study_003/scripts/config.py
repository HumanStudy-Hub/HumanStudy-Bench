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
from typing import Dict, Any, List

class CustomPromptBuilder(PromptBuilder):
    def __init__(self, study_path: Path):
        super().__init__(study_path)
    
    def build_trial_prompt(self, trial_metadata: Dict[str, Any]) -> str:
        """
        Builds a prompt for a single trial consisting of instructions and one or more decision items.
        """
        instructions = trial_metadata.get("instructions", "")
        items = trial_metadata.get("items", [])
        
        prompt = "You are participating in a study on human decision-making. Please read the following information and indicate your preferences.\n\n"
        
        if instructions:
            prompt += f"{instructions.strip()}\n\n"
            
        for i, item in enumerate(items):
            q_idx = i + 1
            prompt += f"Q{q_idx}: {item['question']}\n"
            if item.get("options"):
                for opt in item["options"]:
                    prompt += f"  - {opt}\n"
            prompt += "\n"
        
        # Define the expected response format
        q_labels = [f"Q{i+1}=<your choice>" for i in range(len(items))]
        prompt += f"RESPONSE_SPEC: Please provide your answers in the following format: {', '.join(q_labels)}"
        
        return prompt

class StudyStudy003Config(BaseStudyConfig):
    prompt_builder_class = CustomPromptBuilder
    PROMPT_VARIANT = "v1"
    
    def __init__(self, study_path: Path, specification: Dict[str, Any]):
        super().__init__(study_path, specification)
        
    def create_trials(self, n_trials: int = None) -> List[Dict[str, Any]]:
        """
        Creates trials for all sub-studies in the Tversky & Kahneman (1981) paper.
        """
        trials = []
        spec = self.load_specification()
        
        # List of all sub-study identifiers corresponding to the JSON files
        sub_studies = [
            "problem_1", "problem_2", "problem_3", "problem_4", 
            "problem_5", "problem_6", "problem_7", "problem_8", 
            "problem_9", "problem_10_version_1", "problem_10_version_2"
        ]
        
        for sub_id in sub_studies:
            material = self.load_material(sub_id)
            
            # Get target N from specification, default to 50 if not found or set to 0
            n = spec["participants"]["by_sub_study"].get(sub_id, {}).get("n", 50)
            if n == 0:
                n = 50
                
            # If n_trials is provided, it typically means we are generating a sample/dump
            count = n if n_trials is None else n_trials
            
            for _ in range(count):
                # Each trial contains all items within that specific sub-study (e.g., Problem 3 has 2 items)
                trials.append({
                    "sub_study_id": sub_id,
                    "instructions": material.get("instructions", ""),
                    "items": material.get("items", []),
                    "profile": {
                        "age": random.randint(18, 25), 
                        "gender": random.choice(["male", "female"])
                    },
                    "variant": self.PROMPT_VARIANT
                })
                
        return trials

    def dump_prompts(self, output_dir: str):
        """
        Dumps one sample prompt for each sub-study to the specified directory.
        """
        # We create one trial per sub-study to visualize the prompts
        spec = self.load_specification()
        sub_studies = list(spec["participants"]["by_sub_study"].keys())
        
        for sub_id in sub_studies:
            material = self.load_material(sub_id)
            trial = {
                "sub_study_id": sub_id,
                "instructions": material.get("instructions", ""),
                "items": material.get("items", []),
                "profile": {"age": 20, "gender": "female"},
                "variant": self.PROMPT_VARIANT
            }
            prompt = self.prompt_builder.build_trial_prompt(trial)
            
            output_path = Path(output_dir) / f"study_003_{sub_id}.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(prompt)
