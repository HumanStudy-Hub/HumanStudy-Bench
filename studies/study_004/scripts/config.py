import json
import re
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Any, List, Optional

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from study_utils import BaseStudyConfig, PromptBuilder


class CustomPromptBuilder(PromptBuilder):
    def __init__(self, study_path: Path):
        super().__init__(study_path)

    def build_trial_prompt(self, trial_metadata):
        items = trial_metadata.get("items", [])
        instructions = trial_metadata.get("instructions", "")
        
        prompt = f"You are a participant in a psychology study about human intuition and probability. {instructions}\n\n"
        
        q_idx = 1
        response_specs = []
        
        for item in items:
            item_type = item.get("type")
            question = item.get("question")
            
            # Special handling for Study 6 (Subjective Sampling Distributions) 
            # which requires 11 category estimates
            if item["id"] == "sampling_distribution_estimate":
                prompt += f"Q{q_idx}: {question}\n"
                options = item.get("options", [])
                for i, opt in enumerate(options):
                    prompt += f"  Q{q_idx}.{i+1} - Percentage for category '{opt}': \n"
                    response_specs.append(f"Q{q_idx}.{i+1}=<percentage>")
                item["q_indices"] = [f"Q{q_idx}.{i+1}" for i in range(len(options))]
            
            elif item_type == "multiple_choice":
                options = item.get("options", [])
                option_letters = [chr(65 + i) for i in range(len(options))]
                options_str = "\n".join([f"{letter}) {opt}" for letter, opt in zip(option_letters, options)])
                prompt += f"Q{q_idx}: {question}\n{options_str}\n"
                response_specs.append(f"Q{q_idx}=<{'/'.join(option_letters)}>")
                item["q_idx"] = f"Q{q_idx}"
                item["option_map"] = {letter: opt for letter, opt in zip(option_letters, options)}
            
            elif item_type == "estimation":
                prompt += f"Q{q_idx}: {question}\n"
                response_specs.append(f"Q{q_idx}=<number>")
                item["q_idx"] = f"Q{q_idx}"
            
            q_idx += 1

        prompt += "\nPlease provide your answers based on your intuition. Do not use a calculator.\n"
        prompt += f"\nRESPONSE_SPEC (MANDATORY FORMAT):\n"
        prompt += "- Output ONLY answer lines in the format: Qk=<value> or Qk.n=<value>\n"
        prompt += f"- Use this format for ALL questions: {', '.join(response_specs)}\n"
        prompt += f"- Expected lines: {len(response_specs)}\n"
        
        return prompt

class StudyStudy004Config(BaseStudyConfig):
    prompt_builder_class = CustomPromptBuilder
    PROMPT_VARIANT = "v1"

    def __init__(self, study_path: Path, specification: Dict[str, Any]):
        super().__init__(study_path, specification)

    def create_trials(self, n_trials=None):
        trials = []
        spec = self.load_specification()
        sub_studies = [
            "study_1_proportion", "study_1_randomness", "study_2_programs", 
            "study_3_binomial", "study_4_psychologists", "study_5_marbles", 
            "study_6_sampling_distributions", "study_7_ordinal", 
            "study_8_posterior", "study_9_heights"
        ]

        for sub_id in sub_studies:
            material = self.load_material(sub_id)
            n = spec["participants"]["by_sub_study"].get(sub_id, {}).get("n", 50)
            if n == 0:
                n = 50

            # Determine if sub-study is between-subjects or within-subjects
            # Based on the paper: 
            # Study 6 and 8 are between-subjects (participants see one condition)
            # Study 7 and 9 are within-subjects (participants see all items in the sub-study)
            
            if sub_id == "study_6_sampling_distributions":
                # 9 conditions: 3 Pop (Sex, Heart, Height) x 3 N (10, 100, 1000)
                populations = ["sexes", "heartbeat", "height"]
                sample_sizes = [10, 100, 1000]
                conditions = [(p, s) for p in populations for s in sample_sizes]
                
                for i in range(n):
                    pop, size = conditions[i % len(conditions)]
                    # Create a deep copy of items to modify metadata per participant
                    items_copy = []
                    for it in material["items"]:
                        it_c = it.copy()
                        it_c["question"] = it_c["question"].replace("(N=10, 100, or 1000)", f"(N={size})")
                        it_c["metadata"] = {"population": pop, "sample_size": size}
                        items_copy.append(it_c)
                        
                    trials.append({
                        "sub_study_id": sub_id,
                        "items": items_copy,
                        "instructions": material["instructions"].replace("(sexes p=.50, heartbeat type p=.80, and height)", f"({pop})"),
                        "condition": {"population": pop, "sample_size": size}
                    })

            elif sub_id == "study_8_posterior":
                # 10 conditions (problems)
                for i in range(n):
                    # In a real scenario, we'd have 10 distinct items or metadata for Study 8
                    # Here we follow the JSON structure provided
                    trials.append({
                        "sub_study_id": sub_id,
                        "items": material["items"],
                        "instructions": material["instructions"]
                    })

            elif sub_id in ["study_7_ordinal", "study_9_heights"]:
                # Within-subjects: participant gets all items
                for i in range(n):
                    trials.append({
                        "sub_study_id": sub_id,
                        "items": material["items"],
                        "instructions": material["instructions"]
                    })

            else:
                # Standard one-item sub-studies
                for i in range(n):
                    trials.append({
                        "sub_study_id": sub_id,
                        "items": material["items"],
                        "instructions": material["instructions"]
                    })

        return trials

    def dump_prompts(self, output_dir):
        # Generate one prompt for each sub-study to verify
        sub_studies = [
            "study_1_proportion", "study_1_randomness", "study_2_programs", 
            "study_3_binomial", "study_4_psychologists", "study_5_marbles", 
            "study_6_sampling_distributions", "study_7_ordinal", 
            "study_8_posterior", "study_9_heights"
        ]
        
        for sub_id in sub_studies:
            # Filter trials to find one for this sub_study
            all_trials = self.create_trials(n_trials=10)
            target_trial = next((t for t in all_trials if t["sub_study_id"] == sub_id), None)
            
            if target_trial:
                prompt = self.prompt_builder.build_trial_prompt(target_trial)
                with open(Path(output_dir) / f"study_004_{sub_id}.txt", "w") as f:
                    f.write(prompt)
