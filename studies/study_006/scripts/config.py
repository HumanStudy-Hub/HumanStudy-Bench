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
        Constructs the prompt for a single participant (trial).
        """
        items = trial_metadata.get("items", [])
        instructions = trial_metadata.get("instructions", "")
        trait_list = trial_metadata.get("trait_list", "")
        
        prompt = "You are participating in a psychology study on how people form impressions of others. "
        prompt += "Please read the following information carefully and answer the questions that follow.\n\n"
        
        prompt += f"INSTRUCTIONS:\n{instructions}\n\n"
        
        if trait_list:
            prompt += f"CHARACTERISTICS OF THE PERSON:\n{trait_list}\n\n"
        
        prompt += "Now, based on the impression you have formed, please provide your responses to the following items:\n"
        
        q_specs = []
        for i, item in enumerate(items):
            q_idx = i + 1
            item_type = item.get("type", "multiple_choice")
            question = item.get("question", "")
            
            prompt += f"Q{q_idx}: {question}\n"
            
            if item_type == "multiple_choice":
                options = item.get("options", [])
                for j, opt in enumerate(options):
                    prompt += f"  {chr(65 + j)}) {opt}\n"
                q_specs.append(f"Q{q_idx}=<Letter A, B, etc.>")
            elif item_type == "ranking":
                prompt += "  (Please provide the specific trait name from the list provided above)\n"
                q_specs.append(f"Q{q_idx}=<Trait Name>")
            elif item_type == "open_ended":
                prompt += "  (Please provide a comma-separated list of words)\n"
                q_specs.append(f"Q{q_idx}=<Words>")
            elif item_type == "likert":
                prompt += "  (Answer with a number on a scale of 1 to 7, where 1 is 'Not at all' and 7 is 'Extremely')\n"
                q_specs.append(f"Q{q_idx}=<Number 1-7>")
        
        prompt += "\nRESPONSE_SPEC (MANDATORY FORMAT):\n"
        prompt += "Replace the placeholders with your actual answers, one per line:\n"
        for spec in q_specs:
            prompt += f"{spec}\n"
        prompt += "\nExample: Q1=A. Output ONLY answer lines, no reasoning.\n"
        
        return prompt

class StudyStudy006Config(BaseStudyConfig):
    prompt_builder_class = CustomPromptBuilder
    PROMPT_VARIANT = "v1"

    def __init__(self, study_path: Path, specification: Dict[str, Any]):
        super().__init__(study_path, specification)
        
        # Define the trait lists for the various experiments and conditions
        self.condition_traits = {
            "Experiment I Task": {
                "Warm": "intelligent, skillful, industrious, warm, determined, practical, cautious",
                "Cold": "intelligent, skillful, industrious, cold, determined, practical, cautious"
            },
            "Experiment II Task": {
                "Default": "intelligent, skillful, industrious, determined, practical, cautious"
            },
            "Experiment III Task": {
                "Polite": "intelligent, skillful, industrious, polite, determined, practical, cautious",
                "Blunt": "intelligent, skillful, industrious, blunt, determined, practical, cautious"
            },
            "Experiment IV Task": {
                "Series A": "obedient, weak, shallow, warm, unambitious, vain",
                "Series B": "vain, shrewd, unscrupulous, warm, shallow, envious",
                "Series C": "intelligent, skillful, sincere, cold, conscientious, helpful, modest"
            },
            "Experiment V Task": {
                "Kind": "kind, wise, honest, calm, strong",
                "Cruel": "cruel, shrewd, unscrupulous, calm, strong"
            },
            "Experiment VI Task": {
                "Positive First": "intelligent, industrious, impulsive, critical, stubborn, envious",
                "Negative First": "envious, stubborn, critical, impulsive, industrious, intelligent"
            },
            "Experiment VII Task": {
                "Positive First": "intelligent, skillful, industrious, determined, practical, cautious, evasive",
                "Negative First": "evasive, cautious, practical, determined, industrious, skillful, intelligent"
            },
            "Experiment VIII Task": {
                "Broken": "Initially described as two people: [intelligent, industrious, impulsive] and [critical, stubborn, envious].",
                "Continuous": "intelligent, industrious, impulsive, critical, stubborn, envious"
            },
            "Experiment IX Task": {
                "Default": "intelligent, skillful, warm"
            },
            "Experiment IXa Task": {
                "Warm": "warm",
                "Cold": "cold"
            },
            "Experiment X Task": {
                "Default": "Set 1: quick, skillful, helpful; Set 2: quick, clumsy, helpful; Set 3: slow, skillful, helpful; Set 4: slow, clumsy, helpful"
            }
        }
    
    def load_material(self, sub_study_id: str) -> Dict[str, Any]:
        """从 materials 目录加载指定 sub_study 的 JSON 文件，尝试多种文件名格式"""
        # Try multiple possible filename formats
        possible_names = [
            f"{sub_study_id}.json",
            f"{sub_study_id} Task.json",
            f"{sub_study_id.replace(' ', '_').lower()}.json"
        ]
        
        for filename in possible_names:
            file_path = self.study_path / "materials" / filename
            if file_path.exists():
                with open(file_path, "r", encoding='utf-8') as f:
                    return json.load(f)
        
        # If none found, raise error with all attempted paths
        attempted = [str(self.study_path / "materials" / name) for name in possible_names]
        raise FileNotFoundError(f"Material not found for '{sub_study_id}'. Tried: {attempted}")

    def create_trials(self, n_trials: int = None) -> List[Dict[str, Any]]:
        trials = []
        spec = self.specification
        by_sub_study = spec.get("participants", {}).get("by_sub_study", {})
        
        # Map specification keys to config keys (e.g., "Experiment I" -> "Experiment I Task")
        spec_to_config_map = {
            "Experiment I": "Experiment I Task",
            "Experiment II": "Experiment II Task",
            "Experiment III": "Experiment III Task",
            "Experiment IV": "Experiment IV Task",
            "Experiment V": "Experiment V Task",
            "Experiment VI": "Experiment VI Task",
            "Experiment VII": "Experiment VII Task",
            "Experiment VIII": "Experiment VIII Task",
            "Experiment IX": "Experiment IX Task",
            "Experiment IXa": "Experiment IXa Task",
            "Experiment X": "Experiment X Task"
        }

        for spec_key, sub_spec in by_sub_study.items():
            n = sub_spec.get("n", 50)
            if n == 0:
                n = 50
            
            # Map specification key to config key
            sub_id = spec_to_config_map.get(spec_key, f"{spec_key} Task")
            
            material = self.load_material(sub_id)
            items = material.get("items", [])
            instructions = material.get("instructions", "")
            
            # Identify conditions for this sub-study
            conditions = list(self.condition_traits.get(sub_id, {"Default": ""}).keys())
            
            for _ in range(n):
                condition = random.choice(conditions)
                trait_list = self.condition_traits.get(sub_id, {}).get(condition, "")
                
                # Create a participant trial with all items for that sub-study
                trials.append({
                    "sub_study_id": sub_id,
                    "condition": condition,
                    "instructions": instructions,
                    "trait_list": trait_list,
                    "items": items,
                    "variant": self.PROMPT_VARIANT,
                    "profile": {
                        "age": random.randint(18, 25),
                        "gender": random.choice(["male", "female"]),
                        "background": "beginner in psychology"
                    }
                })
        
        return trials

    def dump_prompts(self, output_dir: str):
        # Generate one sample prompt per sub-study/condition to verify
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        sub_studies = list(self.condition_traits.keys())
        for sub_id in sub_studies:
            conditions = list(self.condition_traits[sub_id].keys())
            for cond in conditions:
                material = self.load_material(sub_id)
                trial = {
                    "sub_study_id": sub_id,
                    "condition": cond,
                    "instructions": material.get("instructions", ""),
                    "trait_list": self.condition_traits[sub_id][cond],
                    "items": material.get("items", [])
                }
                prompt = self.prompt_builder.build_trial_prompt(trial)
                filename = f"{sub_id.replace(' ', '_')}_{cond.replace(' ', '_')}.txt"
                with open(output_path / filename, "w") as f:
                    f.write(prompt)
