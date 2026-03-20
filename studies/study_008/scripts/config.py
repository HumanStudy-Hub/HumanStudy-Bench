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
        items = trial_metadata.get("items", [])
        sub_id = trial_metadata.get("sub_study_id")
        
        prompt = "Please answer the following questions.\n\n"
        
        if sub_id == "study_4_keg_ban_alienation":
            prompt += "The university has recently instituted a new policy banning kegs of beer on campus. Please respond to the following questions regarding this policy.\n\n"

        q_indices = []
        for i, item in enumerate(items):
            q_idx = f"Q{i+1}"
            q_indices.append(q_idx)
            prompt += f"{q_idx}: {item['question']}\n"
            if "options" in item and item["options"] is not None:
                # Check if options is iterable (list/tuple) before joining
                if isinstance(item["options"], (list, tuple)) and len(item["options"]) > 0:
                    prompt += f"Options: {', '.join(str(opt) for opt in item['options'])}\n"
            prompt += "\n"

        spec_format = ", ".join([f"{idx}=<number>" for idx in q_indices])
        prompt += f"RESPONSE_SPEC: Provide your answers in the following format: {spec_format}"
        
        return prompt

class StudyStudy008Config(BaseStudyConfig):
    prompt_builder_class = CustomPromptBuilder
    PROMPT_VARIANT = "v1"

    def __init__(self, study_path: Path, specification: Dict[str, Any]):
        super().__init__(study_path, specification)

    def create_trials(self, n_trials: int = None) -> List[Dict[str, Any]]:
        trials = []
        spec = self.specification
        
        # Study 1: Pluralistic Ignorance regarding Alcohol Habits
        sub_id_1 = "study_1_comfort_estimation"
        material_1 = self.load_material(sub_id_1)
        n_1 = spec["participants"]["by_sub_study"].get(sub_id_1, {}).get("n", 50)
        if n_1 == 0: n_1 = 50
        
        for _ in range(n_1):
            trials.append({
                "sub_study_id": sub_id_1,
                "items": material_1["items"],
                "profile": {"age": random.randint(18, 22), "gender": random.choice(["Male", "Female"])},
                "variant": self.PROMPT_VARIANT
            })

        # Study 2: Pluralistic Ignorance regarding Friends and Order Effects
        sub_id_2 = "study_2_order_and_friend_comparison"
        material_2 = self.load_material(sub_id_2)
        n_2 = spec["participants"]["by_sub_study"].get(sub_id_2, {}).get("n", 50)
        if n_2 == 0: n_2 = 50
        
        # In Study 2, questions were Self, Average Student, and Friends.
        # Order of Self and Average Student was manipulated.
        for i in range(n_2):
            items = material_2["items"].copy()
            # items[0] is Self, items[1] is Average Student, items[2] is Friend
            order = "self_first" if i < n_2 // 2 else "average_first"
            
            if order == "average_first":
                # Swap first two items
                items[0], items[1] = items[1], items[0]
            
            trials.append({
                "sub_study_id": sub_id_2,
                "items": items,
                "order_condition": order,
                "profile": {"age": random.randint(18, 22), "gender": random.choice(["Male", "Female"])},
                "variant": self.PROMPT_VARIANT
            })

        # Study 4: Pluralistic Ignorance and Campus Alienation regarding Keg Ban
        sub_id_4 = "study_4_keg_ban_alienation"
        material_4 = self.load_material(sub_id_4)
        n_4 = spec["participants"]["by_sub_study"].get(sub_id_4, {}).get("n", 50)
        if n_4 == 0: n_4 = 50
        
        for _ in range(n_4):
            trials.append({
                "sub_study_id": sub_id_4,
                "items": material_4["items"],
                "profile": {"age": random.randint(18, 22), "gender": random.choice(["Male", "Female"])},
                "variant": self.PROMPT_VARIANT
            })

        return trials

    def extract_results(self, response_text: str, trial_metadata: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        items = trial_metadata.get("items", [])
        
        for i, item in enumerate(items):
            q_idx = f"Q{i+1}"
            val = self.extract_numeric(response_text, q_idx)
            
            # Map back to original question type for analysis
            # Since items were copied/reordered in Study 2, we use the 'id' from the item
            item_id = item.get("id")
            results[item_id] = val
            
        return results

    def dump_prompts(self, output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        # Create one sample trial per sub-study for dumping
        sub_studies = ["study_1_comfort_estimation", "study_2_order_and_friend_comparison", "study_4_keg_ban_alienation"]
        
        for sub_id in sub_studies:
            all_trials = self.create_trials(n_trials=2)
            trial = next(t for t in all_trials if t["sub_study_id"] == sub_id)
            prompt = self.prompt_builder.build_trial_prompt(trial)
            with open(output_path / f"study_008_{sub_id}.txt", "w", encoding="utf-8") as f:
                f.write(prompt)
