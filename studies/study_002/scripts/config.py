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

    def build_trial_prompt(self, trial_metadata: Dict[str, Any]) -> str:
        # Note: System prompt is now handled separately by SystemPromptRegistry
        # This method only builds the task/trial content
        sub_study_id = trial_metadata.get("sub_study_id", "")
        items = trial_metadata.get("items", [])
        
        prompt = ""

        # Scenario / Instructions
        # We use the instructions from the first item or sub-study
        if "exp_1_calibration" in sub_study_id:
            prompt += "Please provide your best estimate for the following 15 uncertain quantities. After providing each estimate, please rate how confident you are that your answer is within 10% of the true value on a scale from 1 (not confident at all) to 10 (very confident).\n\n"
        elif "exp_1_anchored_estimation" in sub_study_id:
            prompt += "For each of the following 15 quantities, you will perform two tasks: 1. Judge whether the true value is higher or lower than a specified anchor value. 2. Provide your best numerical estimate of the quantity. Finally, rate your confidence in your estimate on a scale from 1 to 10.\n\n"
        elif "exp_2_discredited_anchor" in sub_study_id:
            prompt += "This task involves making judgments about 15 quantities. For each, you will first be shown a number. This number was chosen arbitrarily (e.g., by a wheel of chance) and is not intended to be informative. Please judge if the true value is higher or lower than this number, then provide your best estimate.\n\n"
        elif "exp_3_wtp_estimation" in sub_study_id:
            prompt += "You will be asked about your willingness to pay (WTP) for public goods. First, answer a referendum question (Yes/No) about a specific dollar amount. Then, state the maximum amount you would be willing to pay.\n\n"

        # 3. Items and Q-Index Schema
        q_counter = 1
        spec_lines = []
        
        for item in items:
            q_text = item.get("question", "")
            meta = item.get("metadata", {})
            
            if sub_study_id == "exp_1_calibration":
                # Part 1: Estimate
                prompt += f"Item {q_counter}: {q_text}\n"
                prompt += f"Q{q_counter}.1 (answer with number only): Provide your estimate.\n"
                # Part 2: Confidence
                prompt += f"Q{q_counter}.2 (answer with number only, 1-10): Rate your confidence.\n\n"
                item["q_idx_estimate"] = f"Q{q_counter}.1"
                item["q_idx_confidence"] = f"Q{q_counter}.2"
                spec_lines.append(f"Q{q_counter}.1=<number>, Q{q_counter}.2=<number>")
                q_counter += 1

            elif sub_study_id in ["exp_1_anchored_estimation", "exp_2_discredited_anchor"]:
                # Determine which anchor this trial uses (stored in item metadata during create_trials)
                anchor_type = item.get("assigned_anchor_type", "low") # default to low if missing
                anchor_val = meta.get(f"{anchor_type}_anchor")
                
                # If the question text contains [ANCHOR], replace it. 
                # Otherwise, it's the judgment question.
                display_q = q_text.replace("[ANCHOR]", str(anchor_val))
                
                prompt += f"Item {q_counter}: {display_q}\n"
                prompt += f"Q{q_counter}.1 (answer with letter only: A for Higher, B for Lower): Is the true value higher or lower than {anchor_val}?\n"
                prompt += f"Q{q_counter}.2 (answer with number only): What is your best numerical estimate?\n"
                
                if sub_study_id == "exp_1_anchored_estimation":
                    prompt += f"Q{q_counter}.3 (answer with number only, 1-10): Rate your confidence.\n"
                    item["q_idx_confidence"] = f"Q{q_counter}.3"
                    spec_lines.append(f"Q{q_counter}.1=<A/B>, Q{q_counter}.2=<number>, Q{q_counter}.3=<number>")
                else:
                    spec_lines.append(f"Q{q_counter}.1=<A/B>, Q{q_counter}.2=<number>")
                
                item["q_idx_choice"] = f"Q{q_counter}.1"
                item["q_idx_estimate"] = f"Q{q_counter}.2"
                q_counter += 1

            elif sub_study_id == "exp_3_wtp_estimation":
                # This sub-study has referendum and amount as separate items in materials, 
                # but they are logically paired.
                # Determine which anchor this trial uses (stored in item metadata during create_trials)
                anchor_type = item.get("assigned_anchor_type", "low")  # default to low if missing
                anchor_val = meta.get(f"{anchor_type}_anchor")
                
                # If the question text contains [ANCHOR], replace it
                display_q = q_text.replace("[ANCHOR]", str(anchor_val)) if anchor_val else q_text
                
                prompt += f"Item {q_counter}: {display_q}\n"
                if "referendum" in item.get("id", ""):
                    prompt += f"Q{q_counter} (answer with letter only: A for Yes, B for No): \n"
                    item["q_idx"] = f"Q{q_counter}"
                    spec_lines.append(f"Q{q_counter}=<A/B>")
                else:
                    prompt += f"Q{q_counter} (answer with number only): \n"
                    item["q_idx"] = f"Q{q_counter}"
                    spec_lines.append(f"Q{q_counter}=<number>")
                q_counter += 1

        # 4. Response Spec
        prompt += "\nRESPONSE_SPEC (MANDATORY FORMAT):\n"
        prompt += "- Output ONLY answer lines in the format: Qk=<value> or Qk.n=<value>\n"
        prompt += f"- Expected format: {', '.join(spec_lines)}\n"
        prompt += f"- Expected lines: {len(spec_lines)}\n"

        return prompt

class StudyStudy002Config(BaseStudyConfig):
    prompt_builder_class = CustomPromptBuilder
    PROMPT_VARIANT = "v3"

    def __init__(self, study_path: Path, specification: Dict[str, Any]):
        super().__init__(study_path, specification)

    def create_trials(self, n_trials: int = None) -> List[Dict[str, Any]]:
        trials = []
        sub_studies = ["exp_1_calibration", "exp_1_anchored_estimation", "exp_2_discredited_anchor", "exp_3_wtp_estimation"]
        
        participant_spec = self.specification.get("participants", {})
        by_sub_study = participant_spec.get("by_sub_study", {})

        for sub_id in sub_studies:
            material = self.load_material(sub_id)
            if not material:
                continue
                
            # Determine N
            if n_trials is not None:
                n = n_trials // len(sub_studies)
            else:
                n = by_sub_study.get(sub_id, {}).get("n", 0)
                if n == 0 and "n_per_group" in by_sub_study.get(sub_id, {}):
                    # Handle "45 to 84" range logic
                    n_str = str(by_sub_study[sub_id]["n_per_group"])
                    if "to" in n_str:
                        low, high = [int(s) for s in n_str.split(" to ")]
                        n_per = (low + high) // 2
                    else:
                        n_per = int(n_str)
                    n = n_per * by_sub_study[sub_id].get("total_groups", 1)
            
            # If n is still 0, use a default to ensure all experiments run
            if n == 0:
                n = 50  # Default minimum to ensure experiments run

            # For each participant
            for i in range(n):
                # Random profile
                profile = {
                    "age": np.random.randint(18, 25),
                    "gender": np.random.choice(["male", "female"])
                }
                
                # In anchored conditions, participants were assigned to HIGH or LOW anchor condition
                # Each participant gets ALL items with the SAME anchor type (between-subjects design)
                assigned_anchor_type = None
                if sub_id in ["exp_1_anchored_estimation", "exp_2_discredited_anchor", "exp_3_wtp_estimation"]:
                    # Assign anchor type at participant level (all items get same anchor)
                    assigned_anchor_type = np.random.choice(["low", "high"])
                
                # Create items for this participant
                assigned_items = []
                for item in material.get("items", []):
                    item_copy = item.copy()
                    if assigned_anchor_type:
                        item_copy["assigned_anchor_type"] = assigned_anchor_type
                        # For Exp 3, add anchor values (using reasonable defaults since not in materials)
                        if sub_id == "exp_3_wtp_estimation":
                            item_copy["metadata"] = item_copy.get("metadata", {}).copy()
                            # Use reasonable WTP anchor values (can be refined from paper)
                            item_copy["metadata"]["high_anchor"] = 500
                            item_copy["metadata"]["low_anchor"] = 5
                    assigned_items.append(item_copy)
                
                # ONE trial per participant with ALL items
                trials.append({
                    "sub_study_id": sub_id,
                    "scenario_id": sub_id,
                    "scenario": sub_id,
                    "items": assigned_items,
                    "profile": profile,
                    "variant": self.PROMPT_VARIANT
                })

        return trials

    def dump_prompts(self, output_dir: Path):
        # Create one representative trial per sub-study
        sub_studies = ["exp_1_calibration", "exp_1_anchored_estimation", "exp_2_discredited_anchor", "exp_3_wtp_estimation"]
        for idx, sub_id in enumerate(sub_studies):
            trial_list = self.create_trials(n_trials=len(sub_studies))
            # Find first trial matching sub_id
            target_trial = next((t for t in trial_list if t["sub_study_id"] == sub_id), None)
            if target_trial:
                prompt = self.prompt_builder.build_trial_prompt(target_trial)
                with open(output_dir / f"study_002_trial_{sub_id}.txt", "w") as f:
                    f.write(prompt)

    def aggregate_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        individual_data = []
        
        # We need to track estimates for AI calculation
        # calibration_medians[item_label] = median_value
        calibration_estimates = {}
        anchored_estimates = {} # {item_label: { 'low': [], 'high': [] }}

        for result in raw_results.get("individual_data", []):
            trial_info = result.get("trial_info", {})
            sub_id = trial_info.get("sub_study_id")
            response_text = result.get("response_text", "")
            
            parsed_responses = {}
            for line in response_text.split("\n"):
                if "=" in line:
                    parts = line.split("=")
                    if len(parts) == 2:
                        parsed_responses[parts[0].strip()] = parts[1].strip()

            for item in trial_info.get("items", []):
                label = item.get("metadata", {}).get("label", item.get("id"))
                
                if sub_id == "exp_1_calibration":
                    est_key = item.get("q_idx_estimate")
                    conf_key = item.get("q_idx_confidence")
                    est_val = self.extract_numeric(parsed_responses.get(est_key, ""))
                    conf_val = self.extract_numeric(parsed_responses.get(conf_key, ""))
                    
                    if est_val is not None:
                        if label not in calibration_estimates: calibration_estimates[label] = []
                        calibration_estimates[label].append(est_val)
                    
                    individual_data.append({
                        "sub_study": sub_id,
                        "label": label,
                        "estimate": est_val,
                        "confidence": conf_val
                    })

                elif sub_id in ["exp_1_anchored_estimation", "exp_2_discredited_anchor"]:
                    choice_key = item.get("q_idx_choice")
                    est_key = item.get("q_idx_estimate")
                    conf_key = item.get("q_idx_confidence")
                    
                    choice_val = self.extract_choice(parsed_responses.get(choice_key, ""), ["A", "B"])
                    est_val = self.extract_numeric(parsed_responses.get(est_key, ""))
                    conf_val = self.extract_numeric(parsed_responses.get(conf_key, "")) if conf_key else None
                    
                    anchor_type = item.get("assigned_anchor_type")
                    
                    if est_val is not None:
                        if label not in anchored_estimates: anchored_estimates[label] = {"low": [], "high": []}
                        anchored_estimates[label][anchor_type].append(est_val)

                    individual_data.append({
                        "sub_study": sub_id,
                        "label": label,
                        "anchor_type": anchor_type,
                        "choice": choice_val,
                        "estimate": est_val,
                        "confidence": conf_val
                    })

        # Calculate AI (Anchoring Index)
        # AI = (Median_High - Median_Low) / (Anchor_High - Anchor_Low)
        # Note: We'd need the anchor values from the materials to do this accurately per item
        ai_results = {}
        for label, data in anchored_estimates.items():
            low_ests = data["low"]
            high_ests = data["high"]
            if low_ests and high_ests:
                median_low = np.median(low_ests)
                median_high = np.median(high_ests)
                ai_results[label] = {
                    "median_low": median_low,
                    "median_high": median_high
                    }

        return {
            "descriptive_statistics": {
                "n_total": len(individual_data),
                "ai_by_item": ai_results
            },
            "inferential_statistics": {}
        }
