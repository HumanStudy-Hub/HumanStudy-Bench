import json
import re
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Any, List, Optional

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from study_utils import BaseStudyConfig, PromptBuilder

import numpy as np
from scipy import stats
import random

class CustomPromptBuilder(PromptBuilder):
    def build_trial_prompt(self, trial_metadata):
        # Note: System prompt is now handled separately by SystemPromptRegistry
        # This method only builds the task/trial content
        sub_study_id = trial_metadata.get("sub_study_id", "")
        items = trial_metadata.get("items", [])

        full_prompt = []
        
        # Add task context introduction (similar to study_003/004)
        full_prompt.append("You are participating in a psychology study on decision-making and social judgment. Please read the following scenario and answer the questions.\n")

        q_counter = 1

        if sub_study_id == "study_2_personal_description_items":
            full_prompt.append("For each of the following categories, please (1) choose which sub-category describes you best, and (2) estimate what percentage of college students in general fit into the first sub-category.\n")
            for item in items:
                q_text = item["question"]
                # Split into Choice and Estimate
                full_prompt.append(f"CATEGORY: {q_text}")
                
                # Q_choice
                full_prompt.append(f"Q{q_counter} (Choose one: A or B):")
                item["q_idx_choice"] = q_counter
                q_counter += 1
                
                # Q_estimate
                full_prompt.append(f"Q{q_counter} (Estimate 0-100% for the first category):")
                item["q_idx_estimate"] = q_counter
                q_counter += 1
                full_prompt.append("")

        else:
            # Study 1 and 3: Scenario based
            scenario_item = items[0]
            full_prompt.append(f"SCENARIO: {scenario_item['question']}\n")
            
            options = scenario_item.get("options", ["Option A", "Option B"])
            full_prompt.append(f"Option A: {options[0]}")
            full_prompt.append(f"Option B: {options[1]}\n")

            # 1. Estimates (Peer consensus)
            full_prompt.append(f"Q{q_counter} (What % of your peers do you estimate would choose Option A? Answer 0-100):")
            scenario_item["q_idx_est_a"] = q_counter
            q_counter += 1
            
            full_prompt.append(f"Q{q_counter} (What % of your peers do you estimate would choose Option B? Answer 0-100. Total should be 100%):")
            scenario_item["q_idx_est_b"] = q_counter
            q_counter += 1
            
            # 2. Personal Choice
            full_prompt.append(f"Q{q_counter} (Which option would you personally choose? Answer A or B):")
            scenario_item["q_idx_choice"] = q_counter
            q_counter += 1
            
            # 3. Trait Ratings
            traits = scenario_item.get("metadata", {}).get("traits_to_rate", [])
            if traits:
                full_prompt.append("\nNow, please rate the personal traits of a 'typical person' who would choose each option. Use a scale of 0 to 100, where 50 is 'Average', 0 is 'Very low/Certainly less than average', and 100 is 'Very high/Certainly more than average'.\n")
                scenario_item["trait_q_map"] = {}
                for trait in traits:
                    # Rate Person A
                    full_prompt.append(f"Q{q_counter} (How would you rate the {trait} of a typical person who chooses Option A? Answer 0-100):")
                    q_a = q_counter
                    q_counter += 1
                    
                    # Rate Person B
                    full_prompt.append(f"Q{q_counter} (How would you rate the {trait} of a typical person who chooses Option B? Answer 0-100):")
                    q_b = q_counter
                    q_counter += 1
                    
                    scenario_item["trait_q_map"][trait] = {"opt_a": q_a, "opt_b": q_b}

        full_prompt.append("\nRESPONSE_SPEC (MANDATORY FORMAT):")
        full_prompt.append("- Output ONLY answer lines in the format: Qk=<value>")
        full_prompt.append("- Use this format for ALL questions: Q1=X, Q2=Y, etc.")
        full_prompt.append(f"- Expected lines: {q_counter - 1}")

        return "\n".join(full_prompt)

class StudyStudy001Config(BaseStudyConfig):
    prompt_builder_class = CustomPromptBuilder
    PROMPT_VARIANT = "v3"

    def create_trials(self, n_trials=None):
        trials = []
        
        # Load Specification for demographics
        spec = self.load_specification()
        n_total = spec["participants"]["n"]
        
        # Study 1: Hypothetical Stories
        s1_materials = self.load_material("study_1_hypothetical_stories")
        # 4 scenarios, 80 participants each
        n_per_s1 = 80 if n_trials is None else n_trials
        for item in s1_materials["items"]:
            scenario_id = item["id"] # e.g., supermarket_story
            for _ in range(n_per_s1):
                profile = {"age": random.randint(18, 22), "gender": random.choice(["Male", "Female"])}
                trials.append({
                    "sub_study_id": "study_1_hypothetical_stories",
                    "scenario_id": scenario_id,
                    "scenario": scenario_id,
                    "items": [item],
                    "profile": profile,
                    "variant": self.PROMPT_VARIANT
                })

        # Study 2: Personal Descriptions
        s2_materials = self.load_material("study_2_personal_description_items")
        n_per_s2 = 80 if n_trials is None else n_trials
        for _ in range(n_per_s2):
            profile = {"age": random.randint(18, 22), "gender": random.choice(["Male", "Female"])}
            trials.append({
                "sub_study_id": "study_2_personal_description_items",
                "scenario_id": "personal_description_items",
                "scenario": "personal_description_items",
                "items": s2_materials["items"],
                "profile": profile,
                "variant": self.PROMPT_VARIANT
            })

        # Study 3: Sandwich Board
        s3_materials = self.load_material("study_3_sandwich_board_hypothetical")
        # 2 versions, ~52 participants each
        n_per_s3 = 52 if n_trials is None else n_trials
        for item in s3_materials["items"]:
            scenario_id = item["metadata"]["label"].lower().replace(" ", "_") + "_version"
            for _ in range(n_per_s3):
                profile = {"age": random.randint(18, 22), "gender": random.choice(["Male", "Female"])}
                trials.append({
                    "sub_study_id": "study_3_sandwich_board_hypothetical",
                    "scenario_id": scenario_id,
                    "scenario": scenario_id,
                    "items": [item],
                    "profile": profile,
                    "variant": self.PROMPT_VARIANT
                })

        return trials

    def dump_prompts(self, output_dir):
        # Generate one trial per scenario for inspection
        trials = self.create_trials(n_trials=1)
        # Use set to avoid duplicates since create_trials returns N per scenario
        seen = set()
        unique_trials = []
        for t in trials:
            if t["scenario_id"] not in seen:
                unique_trials.append(t)
                seen.add(t["scenario_id"])
        
        for idx, trial in enumerate(unique_trials):
            prompt = self.prompt_builder.build_trial_prompt(trial)
            with open(f"{output_dir}/study_001_trial_{idx}.txt", "w") as f:
                f.write(prompt)

    def aggregate_results(self, raw_results):
        aggregated = {}
        
        for record in raw_results["individual_data"]:
            trial_info = record["trial_info"]
            response_text = record["response_text"]
            scenario_id = trial_info["scenario_id"]
            sub_study_id = trial_info["sub_study_id"]
            
            if scenario_id not in aggregated:
                aggregated[scenario_id] = []
            
            # Parse Qk=Value
            responses = {}
            for line in response_text.split("\n"):
                if "=" in line and line.startswith("Q"):
                    try:
                        key, val = line.split("=", 1)
                        responses[key.strip()] = val.strip()
                    except: continue

            if sub_study_id == "study_2_personal_description_items":
                # Process all items in the questionnaire
                item_results = []
                for item in trial_info["items"]:
                    choice = self.extract_choice(responses.get(f"Q{item['q_idx_choice']}", ""), ["A", "B"])
                    estimate = self.extract_numeric(responses.get(f"Q{item['q_idx_estimate']}", ""))
                    if choice and estimate is not None:
                        item_results.append({
                            "label": item["metadata"]["gt_key"],
                            "choice": choice,
                            "estimate_cat1": estimate
                        })
                aggregated[scenario_id].append(item_results)
            else:
                # Study 1 or 3
                item = trial_info["items"][0]
                choice = self.extract_choice(responses.get(f"Q{item['q_idx_choice']}", ""), ["A", "B"])
                est_a = self.extract_numeric(responses.get(f"Q{item['q_idx_est_a']}", ""))
                est_b = self.extract_numeric(responses.get(f"Q{item['q_idx_est_b']}", ""))
                
                # Trait ratings logic
                trait_map = item.get("trait_q_map", {})
                sum_abs_diff_a = 0
                sum_abs_diff_b = 0
                count = 0
                for trait, qs in trait_map.items():
                    val_a = self.extract_numeric(responses.get(f"Q{qs['opt_a']}", ""))
                    val_b = self.extract_numeric(responses.get(f"Q{qs['opt_b']}", ""))
                    if val_a is not None and val_b is not None:
                        sum_abs_diff_a += abs(val_a - 50)
                        sum_abs_diff_b += abs(val_b - 50)
                        count += 1
                
                if choice and est_a is not None:
                    aggregated[scenario_id].append({
                        "choice": choice,
                        "est_a": est_a,
                        "est_b": est_b,
                        "trait_sum_a": sum_abs_diff_a,
                        "trait_sum_b": sum_abs_diff_b,
                        "trait_diff_score": sum_abs_diff_a - sum_abs_diff_b # Opt A - Opt B
                    })

        # Calculate Final Stats
        final_results = {"descriptive_statistics": {}, "inferential_statistics": []}
        
        for scenario_id, data in aggregated.items():
            if not data: continue
            
            if scenario_id == "personal_description_items":
                # Flatten items across participants
                items_dict = {}
                for participant_items in data:
                    for item in participant_items:
                        label = item["label"]
                        if label not in items_dict: items_dict[label] = {"cat1": [], "cat2": []}
                        if item["choice"] == "A":
                            items_dict[label]["cat1"].append(item["estimate_cat1"])
                        else:
                            items_dict[label]["cat2"].append(item["estimate_cat1"])
                
                for label, ests in items_dict.items():
                    m1 = np.mean(ests["cat1"]) if ests["cat1"] else 0
                    m2 = np.mean(ests["cat2"]) if ests["cat2"] else 0
                    final_results["descriptive_statistics"][f"{label}_mean_est_cat1_by_cat1"] = m1
                    final_results["descriptive_statistics"][f"{label}_mean_est_cat1_by_cat2"] = m2
                    if ests["cat1"] and ests["cat2"]:
                        t, p = stats.ttest_ind(ests["cat1"], ests["cat2"])
                        final_results["inferential_statistics"].append({
                            "scenario": scenario_id, "item": label, "test": "t-test", "statistic": t, "p_value": p
                        })

            else:
                # Study 1/3 aggregation
                group_a = [d for d in data if d["choice"] == "A"]
                group_b = [d for d in data if d["choice"] == "B"]
                
                est_a_by_a = [d["est_a"] for d in group_a]
                est_a_by_b = [d["est_a"] for d in group_b]
                
                final_results["descriptive_statistics"][f"{scenario_id}_est_a_by_choosers_a"] = np.mean(est_a_by_a) if est_a_by_a else 0
                final_results["descriptive_statistics"][f"{scenario_id}_est_a_by_choosers_b"] = np.mean(est_a_by_b) if est_a_by_b else 0
                
                if est_a_by_a and est_a_by_b:
                    t, p = stats.ttest_ind(est_a_by_a, est_a_by_b)
                    final_results["inferential_statistics"].append({
                        "scenario": scenario_id, "test": "t-test_consensus", "statistic": t, "p_value": p
                    })
                
                # Traits
                diff_by_a = [d["trait_diff_score"] for d in group_a]
                diff_by_b = [d["trait_diff_score"] for d in group_b]
                final_results["descriptive_statistics"][f"{scenario_id}_trait_diff_by_choosers_a"] = np.mean(diff_by_a) if diff_by_a else 0
                final_results["descriptive_statistics"][f"{scenario_id}_trait_diff_by_choosers_b"] = np.mean(diff_by_b) if diff_by_b else 0

        return final_results
