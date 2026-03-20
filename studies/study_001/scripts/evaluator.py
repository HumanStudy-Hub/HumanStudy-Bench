import json
import re
import numpy as np
from scipy import stats
from pathlib import Path

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from stats_lib import parse_p_value_from_reported
from typing import Dict, Any, List

# Module-level cache for ground truth and metadata (loaded once, reused across bootstrap iterations)
_ground_truth_cache = None
_metadata_cache = None


def parse_agent_responses(response_text: str) -> Dict[str, str]:
    """
    Parse standardized responses: Qk=Value or Qk: Value
    Regex handles lines, commas, or spaces as delimiters
    Supports both = and : separators
    """
    parsed_responses = {}
    matches = re.findall(r"Q(\d+)\s*[:=]\s*([^,\s\n]+)", response_text)
    for q_num, val in matches:
        parsed_responses[f"Q{q_num}"] = val.strip()
    return parsed_responses


def get_required_q_numbers(trial_info: Dict[str, Any]) -> set:
    """
    从trial_info中提取所有需要的Q编号。
    Study_001: 需要处理q_idx_choice, q_idx_est_a, q_idx_est_b, trait_q_map
    """
    required = set()
    items = trial_info.get("items", [])

    for item in items:
        # 基本Q编号
        for key in ['q_idx_choice', 'q_idx_est_a', 'q_idx_est_b', 'q_idx_estimate']:
            q_idx = item.get(key)
            if q_idx:
                if isinstance(q_idx, str) and q_idx.startswith("Q"):
                    required.add(q_idx)
                else:
                    required.add(f"Q{q_idx}")

        trait_map = item.get("trait_q_map", {})
        for trait, qs in trait_map.items():
            if isinstance(qs, dict):
                for opt_key in ['opt_a', 'opt_b']:
                    q_idx = qs.get(opt_key)
                    if q_idx:
                        if isinstance(q_idx, str) and q_idx.startswith("Q"):
                            required.add(q_idx)
                        else:
                            required.add(f"Q{q_idx}")

    return required


def _expected_direction_to_int(expected_dir_str: str) -> int:
    """Convert expected_direction string to int: 1 = positive, -1 = negative, 0 = unknown."""
    if not expected_dir_str:
        return 0
    s = str(expected_dir_str).lower()
    if s in ("positive", "greater", ">"):
        return 1
    if s in ("negative", "less", "<"):
        return -1
    return 0


def evaluate_study(results):
    """
    Evaluates the agent's performance on Study 001 (False Consensus Effect).
    Extracts answers from agent responses and computes statistics (t-test).
    Returns test_results with raw stats per scenario; no BF/PAS aggregation.
    """
    global _ground_truth_cache, _metadata_cache

    # 1. Load Ground Truth Data (with caching)
    if _ground_truth_cache is None:
        study_dir = Path(__file__).resolve().parent.parent / "source"
        with open(study_dir / "ground_truth.json", 'r') as f:
            _ground_truth_cache = json.load(f)

    ground_truth = _ground_truth_cache

    # Load metadata (cached) - only for optional context, no weights
    if _metadata_cache is None:
        study_dir = Path(__file__).resolve().parent.parent / "source"
        metadata_path = study_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                _metadata_cache = json.load(f)
        else:
            _metadata_cache = {}

    metadata = _metadata_cache

    # Cache for materials files
    materials_cache = {}
    study_dir = Path(__file__).resolve().parent.parent / "source"

    def load_materials(sub_study_id: str) -> List[Dict[str, Any]]:
        if sub_study_id not in materials_cache:
            materials_path = study_dir / "materials" / f"{sub_study_id}.json"
            if materials_path.exists():
                try:
                    with open(materials_path, 'r') as f:
                        material_data = json.load(f)
                        materials_cache[sub_study_id] = material_data.get("items", [])
                except Exception:
                    materials_cache[sub_study_id] = []
            else:
                materials_cache[sub_study_id] = []
        return materials_cache[sub_study_id]

    # 2. Extract and Aggregate Agent Data
    aggregated = {}

    for participant in results.get("individual_data", []):
        for response in participant.get("responses", []):
            response_text = response.get("response_text", "")
            trial_info = response.get("trial_info", {})
            sub_study_id = trial_info.get("sub_study_id")
            scenario_id = trial_info.get("scenario_id")
            items = trial_info.get("items", [])

            if not items and sub_study_id:
                items = load_materials(sub_study_id)

            if not sub_study_id:
                continue

            if sub_study_id not in aggregated:
                aggregated[sub_study_id] = {}

            parsed_responses = parse_agent_responses(response_text)

            if sub_study_id == "study_2_personal_description_items":
                q_counter = 1
                for item in items:
                    if 'q_idx_choice' not in item:
                        item['q_idx_choice'] = q_counter
                        q_counter += 1
                        item['q_idx_estimate'] = q_counter
                        q_counter += 1

                    gt_key = item.get("metadata", {}).get("gt_key")
                    if not gt_key:
                        continue

                    if gt_key not in aggregated[sub_study_id]:
                        aggregated[sub_study_id][gt_key] = []

                    q_choice_idx = item.get('q_idx_choice')
                    q_est_idx = item.get('q_idx_estimate')

                    raw_choice = parsed_responses.get(f"Q{q_choice_idx}")
                    raw_est = parsed_responses.get(f"Q{q_est_idx}")

                    if raw_choice and raw_est:
                        try:
                            choice_raw = raw_choice.upper().strip()
                            pos_a = choice_raw.find("A")
                            pos_b = choice_raw.find("B")

                            if pos_a != -1 and pos_b != -1:
                                choice = "A" if pos_a < pos_b else "B"
                            elif pos_a != -1:
                                choice = "A"
                            elif pos_b != -1:
                                choice = "B"
                            else:
                                first_char = choice_raw[0] if choice_raw else None
                                if first_char in ["A", "B"]:
                                    choice = first_char
                                else:
                                    continue

                            est = float(raw_est)
                            aggregated[sub_study_id][gt_key].append({
                                "choice": choice,
                                "estimate_cat1": est
                            })
                        except (ValueError, IndexError):
                            continue
            else:
                if not items:
                    continue
                item = items[0]

                if 'q_idx_choice' not in item:
                    item['q_idx_est_a'] = 1
                    item['q_idx_est_b'] = 2
                    item['q_idx_choice'] = 3

                    traits = item.get("metadata", {}).get("traits_to_rate", [])
                    if traits:
                        item["trait_q_map"] = {}
                        q_curr = 4
                        for trait in traits:
                            item["trait_q_map"][trait] = {"opt_a": q_curr, "opt_b": q_curr + 1}
                            q_curr += 2

                if scenario_id not in aggregated[sub_study_id]:
                    aggregated[sub_study_id][scenario_id] = []

                q_choice_idx = item.get('q_idx_choice')
                q_est_a_idx = item.get('q_idx_est_a')
                q_est_b_idx = item.get('q_idx_est_b')

                raw_choice = parsed_responses.get(f"Q{q_choice_idx}")
                raw_est_a = parsed_responses.get(f"Q{q_est_a_idx}")
                raw_est_b = parsed_responses.get(f"Q{q_est_b_idx}")

                if raw_choice and raw_est_a:
                    try:
                        choice_raw = raw_choice.upper().strip()
                        pos_a = choice_raw.find("A")
                        pos_b = choice_raw.find("B")

                        if pos_a != -1 and pos_b != -1:
                            choice = "A" if pos_a < pos_b else "B"
                        elif pos_a != -1:
                            choice = "A"
                        elif pos_b != -1:
                            choice = "B"
                        else:
                            first_char = choice_raw[0] if choice_raw else None
                            if first_char in ["A", "B"]:
                                choice = first_char
                            else:
                                continue

                        est_a = float(raw_est_a)
                        est_b = float(raw_est_b) if raw_est_b else (100.0 - est_a)

                        trait_map = item.get("trait_q_map", {})
                        sum_abs_diff_a = 0
                        sum_abs_diff_b = 0
                        trait_count = 0
                        for trait, qs in trait_map.items():
                            val_a = parsed_responses.get(f"Q{qs['opt_a']}")
                            val_b = parsed_responses.get(f"Q{qs['opt_b']}")
                            if val_a and val_b:
                                try:
                                    sum_abs_diff_a += abs(float(val_a) - 50)
                                    sum_abs_diff_b += abs(float(val_b) - 50)
                                    trait_count += 1
                                except ValueError:
                                    pass

                        aggregated[sub_study_id][scenario_id].append({
                            "choice": choice,
                            "est_a": est_a,
                            "est_b": est_b,
                            "trait_diff_score": (sum_abs_diff_a - sum_abs_diff_b) if trait_count > 0 else None
                        })
                    except (ValueError, IndexError):
                        pass

    # 3. Compute stats per scenario and build test_results
    test_results = []
    sig_level = 0.05

    def normalize_key(k):
        return re.sub(r'[^a-z0-9]', '', str(k).lower().replace("version", ""))

    for study_gt in ground_truth.get("studies", []):
        study_label = study_gt.get("study_id")
        for finding in study_gt.get("findings", []):
            finding_id = finding.get("finding_id")
            statistical_tests = finding.get("statistical_tests", [])
            test_gt = statistical_tests[0] if statistical_tests else {}
            expected_dir_str = test_gt.get("expected_direction", "positive")
            h_expected = _expected_direction_to_int(expected_dir_str)
            reported_stats = test_gt.get("reported_statistics", "")

            if study_label == "Study 1":
                sub_study_id = "study_1_hypothetical_stories"
            elif study_label == "Study 2":
                sub_study_id = "study_2_personal_description_items"
            else:
                sub_study_id = "study_3_sandwich_board_hypothetical"

            gt_data_points = finding.get("original_data_points", {}).get("data", {})

            for scenario_key, scenario_gt in gt_data_points.items():
                agent_scenario_key = None
                norm_scenario_key = normalize_key(scenario_key)

                if sub_study_id in aggregated:
                    for k in aggregated[sub_study_id].keys():
                        if normalize_key(k) == norm_scenario_key:
                            agent_scenario_key = k
                            break

                # Parse human p-value for replication check
                human_p_value = None
                parsed_p, parsed_sig, _ = parse_p_value_from_reported(reported_stats, sig_level)
                if parsed_p is not None:
                    human_p_value = parsed_p
                human_significant = parsed_sig if parsed_p is not None else None

                # Compute agent stats
                n_agent_1 = None
                n_agent_2 = None
                mean_agent_1 = None
                mean_agent_2 = None
                t_stat = None
                p_value = None
                agent_significant = None
                direction_match = None

                if agent_scenario_key and agent_scenario_key in aggregated[sub_study_id]:
                    data = aggregated[sub_study_id][agent_scenario_key]

                    if finding_id in ["F1", "F3", "F4"]:
                        group_a = []
                        group_b = []
                        for d in data:
                            val = d.get("est_a") if finding_id != "F3" else d.get("estimate_cat1")
                            if val is not None:
                                if d["choice"] == "A":
                                    group_a.append(val)
                                else:
                                    group_b.append(val)

                        if len(group_a) >= 2 and len(group_b) >= 2:
                            t_stat, p_value = stats.ttest_ind(group_a, group_b)
                            n_agent_1 = len(group_a)
                            n_agent_2 = len(group_b)
                            mean_agent_1 = float(np.mean(group_a))
                            mean_agent_2 = float(np.mean(group_b))
                            if not np.isnan(t_stat):
                                agent_significant = p_value < sig_level
                                a_dir = 1 if t_stat > 0 else -1
                                direction_match = (h_expected == 0) or (a_dir == h_expected)

                    elif finding_id in ["F2", "F5"]:
                        group_a = [d["trait_diff_score"] for d in data if d["choice"] == "A" and d["trait_diff_score"] is not None]
                        group_b = [d["trait_diff_score"] for d in data if d["choice"] == "B" and d["trait_diff_score"] is not None]

                        if len(group_a) >= 2 and len(group_b) >= 2:
                            t_stat, p_value = stats.ttest_ind(group_a, group_b)
                            n_agent_1 = len(group_a)
                            n_agent_2 = len(group_b)
                            mean_agent_1 = float(np.mean(group_a))
                            mean_agent_2 = float(np.mean(group_b))
                            if not np.isnan(t_stat):
                                agent_significant = p_value < sig_level
                                a_dir = 1 if t_stat > 0 else -1
                                direction_match = (h_expected == 0) or (a_dir == h_expected)

                replication = None
                if human_significant is not None and agent_significant is not None and direction_match is not None:
                    replication = human_significant and agent_significant and direction_match

                test_result = {
                    "study_id": study_label,
                    "sub_study_id": sub_study_id,
                    "finding_id": finding_id,
                    "scenario": scenario_key,
                    "n_agent_1": n_agent_1,
                    "n_agent_2": n_agent_2,
                    "mean_agent_1": mean_agent_1,
                    "mean_agent_2": mean_agent_2,
                    "t_stat": float(t_stat) if t_stat is not None and not np.isnan(t_stat) else None,
                    "p_value": float(p_value) if p_value is not None and not np.isnan(p_value) else None,
                    "significant": agent_significant,
                    "direction_match": direction_match,
                    "human_p_value": human_p_value,
                    "human_significant": human_significant,
                    "replication": replication,
                }
                test_results.append(test_result)

    return {"test_results": test_results}
