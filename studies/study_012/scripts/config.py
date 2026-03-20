import json
import re
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from study_utils import BaseStudyConfig, PromptBuilder
from participant_pool import ParticipantPool


class RoomAPromptBuilder(PromptBuilder):
    """Prompt builder for Room A (trust decision)."""
    def __init__(self, study_path: Path):
        super().__init__(study_path)

    def build_trial_prompt(self, trial_metadata: Dict[str, Any]) -> str:
        """Build high-fidelity prompt for Room A decision."""
        sub_study_id = trial_metadata.get("sub_study_id", "")
        items = trial_metadata.get("items", [])
        instructions = trial_metadata.get("instructions", "")
        
        # Set q_idx on items so evaluator can extract required Q numbers
        for item in items:
            if item.get("id") == "investment_decision":
                item["q_idx"] = "Q1"
        
        # High-fidelity first-person prompt
        prompt = "You are a participant in an economic experiment. You are in Room A. Another participant is in Room B. You will never know their identity, nor will they know yours.\n\n"
        prompt += "You have been given $10 as a show-up fee.\n\n"
        prompt += "Task: You must decide how much of your $10 to send to the person in Room B.\n\n"
        prompt += "- You can send $0, $1, ..., up to $10.\n"
        prompt += "- Whatever you send will be TRIPLED by the experimenter before it reaches Room B.\n"
        prompt += "- The person in Room B will then decide how much of that tripled money to send back to you.\n"
        prompt += "- You keep whatever you didn't send, plus whatever Room B sends back.\n\n"
        
        # Add Social History report if applicable
        if "social_history" in sub_study_id:
            prompt += "[Social History Report]\n\n"
            prompt += "Note: Experiments like this have been run before. Here is a summary of the results from previous sessions:\n\n"
            prompt += "- On average, people in Room A sent $5.16.\n"
            prompt += "- People who sent $5.00 received back an average of $7.17.\n"
            prompt += "- People who sent $10.00 received back an average of $10.20.\n"
            prompt += "- Out of 32 pairs, 30 people sent money.\n\n"
        
        prompt += "Decision:\n"
        prompt += "How many dollars will you send to Room B? Output only the integer (0-10).\n"
        prompt += "Format: Q1=<number>\n"
        
        return prompt


class RoomBPromptBuilder(PromptBuilder):
    """Prompt builder for Room B (reciprocity decision)."""
    def __init__(self, study_path: Path):
        super().__init__(study_path)
    
    def build_trial_prompt(self, trial_metadata: Dict[str, Any]) -> str:
        """Build high-fidelity prompt for Room B decision with actual received amount."""
        sub_study_id = trial_metadata.get("sub_study_id", "")
        amount_sent = trial_metadata.get("amount_sent", 0)
        tripled_amount = 3 * amount_sent
        items = trial_metadata.get("items", [])
        
        # Set q_idx on items so evaluator can extract required Q numbers
        # Room B trials might not have items, but if they do, set q_idx
        for item in items:
            if item.get("id") in ["payback_decision", "investment_decision"]:
                item["q_idx"] = "Q1"
        
        # High-fidelity first-person prompt
        prompt = "You are a participant in an economic experiment. You are in Room B. You have $10 as a show-up fee which is yours to keep.\n\n"
        prompt += f"The participant in Room A was given $10 and decided to send you ${amount_sent:.0f}.\n\n"
        prompt += "This amount has been TRIPLED.\n\n"
        prompt += f"You have received ${tripled_amount:.0f}.\n\n"
        prompt += "Task: You must decide how much of this ${tripled_amount:.0f} to send back to Room A.\n\n"
        prompt += f"- You can send back $0 up to ${tripled_amount:.0f}.\n"
        prompt += "- Room A will keep whatever you send back.\n"
        prompt += f"- You will keep your $10 show-up fee + (${tripled_amount:.0f} - amount you send back).\n\n"
        
        # Add Social History report if applicable (Room B also sees it)
        if "social_history" in sub_study_id:
            prompt += "[Social History Report]\n\n"
            prompt += "Note: Experiments like this have been run before. Here is a summary of the results from previous sessions:\n\n"
            prompt += "- On average, people in Room A sent $5.16.\n"
            prompt += "- People who sent $5.00 received back an average of $7.17.\n"
            prompt += "- People who sent $10.00 received back an average of $10.20.\n"
            prompt += "- Out of 32 pairs, 30 people sent money.\n\n"
        
        prompt += "Decision:\n"
        prompt += f"How many dollars will you return to Room A? Output only the integer (0-{tripled_amount:.0f}).\n"
        prompt += "Format: Q1=<number>\n"
        
        return prompt


class StudyStudy012Config(BaseStudyConfig):
    prompt_builder_class = RoomAPromptBuilder  # Default, but we use separate builders
    PROMPT_VARIANT = "v1"
    REQUIRES_GROUP_TRIALS = True  # Enable two-stage execution

    def __init__(self, study_path: Path, specification: Dict[str, Any]):
        super().__init__(study_path, specification)
        self.room_a_builder = RoomAPromptBuilder(self.source_path)
        self.room_b_builder = RoomBPromptBuilder(self.source_path)

    def create_trials(self, n_trials=None, stage="room_a"):
        """
        Create trials for the investment game.
        
        Args:
            n_trials: Number of trials to create
            stage: "room_a" for Stage 1 (trust decisions), "room_b" for Stage 2 (reciprocity decisions)
        """
        if stage == "room_b":
            # Room B trials are created dynamically based on Room A results
            # This method is called from run_two_stage_experiment
            return []
        
        trials = []
        spec = self.load_specification()
        sub_studies = ["no_history_investment_game", "social_history_investment_game"]
        
        for sub_id in sub_studies:
            material = self.load_material(sub_id)
            # Use the number of subjects from the human study or default
            n = spec["participants"]["by_sub_study"].get(sub_id, {}).get("n", 50)
            if n == 0:
                n = 50
            if n_trials is not None:
                # If n_trials specified, split evenly between sub-studies
                n = n_trials // len(sub_studies)
            
            for i in range(n):
                # Create Room A trial
                items_copy = []
                for item in material.get("items", []):
                    if item["id"] == "investment_decision":
                        items_copy.append(dict(item))
                    elif item["id"] == "social_history_report" and "social_history" in sub_id:
                        items_copy.append(dict(item))
                
                trials.append({
                    "sub_study_id": sub_id,
                    "role": "room_a",
                    "items": items_copy,
                    "instructions": material.get("instructions", ""),
                    "variant": self.PROMPT_VARIANT,
                    "pair_index": len(trials)  # Track pairing
                })
        
        return trials

    def create_room_b_trials(self, room_a_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create Room B trials based on Room A decisions.
        
        Args:
            room_a_results: List of Room A responses with amount_sent
            
        Returns:
            List of Room B trial dictionaries
        """
        room_b_trials = []
        
        for room_a_result in room_a_results:
            trial_info = room_a_result.get("trial_info", {})
            sub_study_id = trial_info.get("sub_study_id", "")
            pair_index = trial_info.get("pair_index", len(room_b_trials))
            
            # Extract amount sent from Room A response
            response_text = room_a_result.get("response_text", "")
            amount_sent = self._extract_amount_sent(response_text)
            
            # Skip if Room A sent $0 (no decision for Room B)
            if amount_sent == 0:
                continue
            
            # Create Room B trial with actual received amount
            room_b_trials.append({
                "sub_study_id": sub_study_id,
                "role": "room_b",
                "amount_sent": amount_sent,
                "tripled_amount": 3 * amount_sent,
                "instructions": "You are in Room B. Decide how much to return.",
                "variant": self.PROMPT_VARIANT,
                "pair_index": pair_index,
                "paired_room_a_id": room_a_result.get("participant_id")
            })
        
        return room_b_trials
    
    def _extract_amount_sent(self, response_text: str) -> float:
        """Extract amount sent from Room A response."""
        # Try Q1= format first
        pattern = re.compile(r"Q1\s*=\s*([\d\.]+)")
        match = pattern.search(response_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        
        # Fallback: extract first number
        numbers = re.findall(r'\d+\.?\d*', response_text)
        if numbers:
            try:
                val = float(numbers[0])
                return max(0, min(10, val))  # Clamp to 0-10
            except ValueError:
                pass
        
        return 0.0
    
    def parse_responses(self, trial_metadata, response_text):
        """Parse responses for both Room A and Room B."""
        parsed_results = {}
        role = trial_metadata.get("role", "room_a")
        
        if role == "room_a":
            # Room A: extract amount sent
            amount_sent = self._extract_amount_sent(response_text)
            parsed_results["investment_decision"] = amount_sent
        else:
            # Room B: extract amount returned
            pattern = re.compile(r"Q1\s*=\s*([\d\.]+)")
            match = pattern.search(response_text)
            if match:
                try:
                    amount_returned = float(match.group(1))
                    parsed_results["payback_decision"] = amount_returned
                except ValueError:
                    parsed_results["payback_decision"] = None
            else:
                # Fallback
                numbers = re.findall(r'\d+\.?\d*', response_text)
                if numbers:
                    try:
                        parsed_results["payback_decision"] = float(numbers[0])
                    except ValueError:
                        parsed_results["payback_decision"] = None
                else:
                    parsed_results["payback_decision"] = None
        
        return parsed_results

    def run_group_experiment(
        self,
        trials: List[Dict[str, Any]],
        instructions: str,
        participant_pool_kwargs: Dict[str, Any],
        prompt_builder: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run two-stage experiment: Stage 1 (Room A) then Stage 2 (Room B).
        
        This implements the dynamic pairing where Room B decisions depend on
        Room A's actual choices. This method is called by the pipeline when
        REQUIRES_GROUP_TRIALS is True.
        """
        print("Running two-stage experiment for study_012")
        print(f"Stage 1: Room A decisions ({len(trials)} participants)")
        
        # Stage 1: Run Room A trials
        # CRITICAL: Set n_participants = len(trials) so each participant gets exactly 1 trial
        room_a_pool_kwargs = participant_pool_kwargs.copy()
        room_a_pool_kwargs["n_participants"] = len(trials)  # 1 participant per trial
        room_a_pool = ParticipantPool(**room_a_pool_kwargs)
        
        # Update participants to be Room A and assign one trial per participant
        for idx, participant in enumerate(room_a_pool.participants):
            participant.participant_id = idx
            participant.profile["participant_id"] = idx
            participant.profile["role"] = "room_a"
        
        # Run Stage 1: Each participant completes exactly 1 trial (their assigned trial)
        # Use multithreading if num_workers > 1 and use_real_llm is True
        num_workers = participant_pool_kwargs.get("num_workers", 1)
        use_real_llm = participant_pool_kwargs.get("use_real_llm", False)
        
        room_a_individual_data = []
        
        def process_room_a_trial(participant_trial_pair):
            """Process a single Room A trial."""
            participant, trial = participant_trial_pair
            try:
                # Merge participant profile into trial
                trial_with_profile = {**trial, "participant_profile": participant.profile}
                
                # Build prompt for this specific trial
                trial_prompt = self.room_a_builder.build_trial_prompt(trial_with_profile)
                
                # Complete this single trial
                response_data = participant.complete_trial(trial_prompt, trial_with_profile)
                
                # Format response data to match expected structure
                return {
                    "participant_id": participant.participant_id,
                    "responses": [{
                        "response_text": response_data.get("response_text", ""),
                        "usage": response_data.get("usage", {}),
                        "trial_info": trial_with_profile
                    }]
                }
            except Exception as e:
                print(f"Error processing Room A trial for participant {participant.participant_id}: {e}")
                return {
                    "participant_id": participant.participant_id,
                    "responses": [{
                        "response_text": "",
                        "usage": {},
                        "trial_info": trial_with_profile if 'trial_with_profile' in locals() else trial
                    }]
                }
        
        # Use multithreading if enabled
        if num_workers > 1 and use_real_llm:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from tqdm import tqdm
            
            participant_trial_pairs = list(zip(room_a_pool.participants, trials))
            room_a_individual_data = [None] * len(participant_trial_pairs)
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                future_to_idx = {
                    executor.submit(process_room_a_trial, pair): idx
                    for idx, pair in enumerate(participant_trial_pairs)
                }
                
                # Collect results with progress bar
                with tqdm(total=len(participant_trial_pairs), desc="Stage 1 (Room A)", unit="trial") as pbar:
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            result = future.result()
                            room_a_individual_data[idx] = result
                        except Exception as e:
                            print(f"Error in Room A trial {idx}: {e}")
                            room_a_individual_data[idx] = {
                                "participant_id": participant_trial_pairs[idx][0].participant_id,
                                "responses": [{"response_text": "", "trial_info": participant_trial_pairs[idx][1]}]
                            }
                        pbar.update(1)
        else:
            # Sequential processing
            from tqdm import tqdm
            participant_trial_pairs = list(zip(room_a_pool.participants, trials))
            with tqdm(total=len(participant_trial_pairs), desc="Stage 1 (Room A)", unit="trial") as pbar:
                for participant, trial in participant_trial_pairs:
                    result = process_room_a_trial((participant, trial))
                    room_a_individual_data.append(result)
                    pbar.update(1)
        
        # Create results structure matching run_experiment output
        room_a_results = {"individual_data": room_a_individual_data}
        
        print(f"Stage 1 complete: {len(room_a_individual_data)} Room A responses")
        
        # Extract Room A decisions and create Room B trials
        room_a_decisions = []
        for participant_data in room_a_individual_data:
            for response in participant_data.get("responses", []):
                trial_info = response.get("trial_info", {})
                if trial_info.get("role") == "room_a":
                    amount_sent = self._extract_amount_sent(response.get("response_text", ""))
                    room_a_decisions.append({
                        "participant_id": participant_data.get("participant_id"),
                        "trial_info": trial_info,
                        "response_text": response.get("response_text", ""),
                        "amount_sent": amount_sent
                    })
        
        # Create Room B trials based on Room A decisions
        room_b_trials = self.create_room_b_trials([
            {
                "trial_info": d["trial_info"],
                "response_text": d["response_text"],
                "participant_id": d["participant_id"]
            }
            for d in room_a_decisions
        ])
        
        print(f"Stage 2: Room B decisions ({len(room_b_trials)} participants)")
        
        if not room_b_trials:
            print("Warning: No Room B trials created (all Room A sent $0?)")
            return room_a_results
        
        # Stage 2: Run Room B trials
        room_b_pool_kwargs = participant_pool_kwargs.copy()
        room_b_pool_kwargs["n_participants"] = len(room_b_trials)  # 1 participant per trial
        room_b_pool = ParticipantPool(**room_b_pool_kwargs)
        
        # Update participants to be Room B
        for idx, participant in enumerate(room_b_pool.participants):
            participant.participant_id = len(room_a_individual_data) + idx
            participant.profile["participant_id"] = participant.participant_id
            participant.profile["role"] = "room_b"
        
        # Run Stage 2: Each participant completes exactly 1 trial (their assigned trial)
        # Use multithreading if num_workers > 1 and use_real_llm is True
        # Re-get num_workers and use_real_llm for clarity
        num_workers = participant_pool_kwargs.get("num_workers", 1)
        use_real_llm = participant_pool_kwargs.get("use_real_llm", False)
        
        room_b_individual_data = []
        
        def process_room_b_trial(participant_trial_pair):
            """Process a single Room B trial."""
            participant, trial = participant_trial_pair
            try:
                # Merge participant profile into trial
                trial_with_profile = {**trial, "participant_profile": participant.profile}
                
                # Build prompt for this specific trial
                trial_prompt = self.room_b_builder.build_trial_prompt(trial_with_profile)
                
                # Complete this single trial
                response_data = participant.complete_trial(trial_prompt, trial_with_profile)
                
                # Format response data to match expected structure
                return {
                    "participant_id": participant.participant_id,
                    "responses": [{
                        "response_text": response_data.get("response_text", ""),
                        "usage": response_data.get("usage", {}),
                        "trial_info": trial_with_profile
                    }]
                }
            except Exception as e:
                print(f"Error processing Room B trial for participant {participant.participant_id}: {e}")
                return {
                    "participant_id": participant.participant_id,
                    "responses": [{
                        "response_text": "",
                        "usage": {},
                        "trial_info": trial_with_profile if 'trial_with_profile' in locals() else trial
                    }]
                }
        
        # Use multithreading if enabled
        if num_workers > 1 and use_real_llm:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from tqdm import tqdm
            
            participant_trial_pairs = list(zip(room_b_pool.participants, room_b_trials))
            room_b_individual_data = [None] * len(participant_trial_pairs)
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                future_to_idx = {
                    executor.submit(process_room_b_trial, pair): idx
                    for idx, pair in enumerate(participant_trial_pairs)
                }
                
                # Collect results with progress bar
                with tqdm(total=len(participant_trial_pairs), desc="Stage 2 (Room B)", unit="trial") as pbar:
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            result = future.result()
                            room_b_individual_data[idx] = result
                        except Exception as e:
                            print(f"Error in Room B trial {idx}: {e}")
                            room_b_individual_data[idx] = {
                                "participant_id": participant_trial_pairs[idx][0].participant_id,
                                "responses": [{"response_text": "", "trial_info": participant_trial_pairs[idx][1]}]
                            }
                        pbar.update(1)
        else:
            # Sequential processing
            from tqdm import tqdm
            participant_trial_pairs = list(zip(room_b_pool.participants, room_b_trials))
            with tqdm(total=len(participant_trial_pairs), desc="Stage 2 (Room B)", unit="trial") as pbar:
                for participant, trial in participant_trial_pairs:
                    result = process_room_b_trial((participant, trial))
                    room_b_individual_data.append(result)
                    pbar.update(1)
        
        # Create results structure matching run_experiment output
        room_b_results = {"individual_data": room_b_individual_data}
        
        print(f"Stage 2 complete: {len(room_b_individual_data)} Room B responses")
        
        # Combine results: merge Room A and Room B data
        # Structure: each "participant" represents a pair (Room A + Room B)
        combined_individual_data = []
        
        # Create pairs
        room_b_by_pair = {}
        for participant_data in room_b_individual_data:
            for response in participant_data.get("responses", []):
                trial_info = response.get("trial_info", {})
                pair_index = trial_info.get("pair_index")
                if pair_index is not None:
                    if pair_index not in room_b_by_pair:
                        room_b_by_pair[pair_index] = []
                    room_b_by_pair[pair_index].append({
                        "participant_id": participant_data.get("participant_id"),
                        "response": response
                    })
        
        # Match Room A with Room B
        for room_a_data in room_a_individual_data:
            pair_responses = []
            for response in room_a_data.get("responses", []):
                trial_info = response.get("trial_info", {})
                pair_index = trial_info.get("pair_index")
                
                # Add Room A response
                pair_responses.append(response)
                
                # Add corresponding Room B response if exists
                if pair_index in room_b_by_pair and room_b_by_pair[pair_index]:
                    room_b_response = room_b_by_pair[pair_index][0]["response"]
                    pair_responses.append(room_b_response)
            
            combined_individual_data.append({
                "participant_id": room_a_data.get("participant_id"),
                "responses": pair_responses
            })
        
        return {
            "individual_data": combined_individual_data
        }
    
    def dump_prompts(self, output_dir):
        """Dump example prompts for both roles."""
        trials = self.create_trials(n_trials=2)
        for idx, trial in enumerate(trials):
            prompt = self.room_a_builder.build_trial_prompt(trial)
            output_path = Path(output_dir) / f"study_012_room_a_{trial['sub_study_id']}_trial_{idx}.txt"
            with open(output_path, "w") as f:
                f.write(prompt)
        
        # Example Room B prompt
        room_b_trial = {
            "sub_study_id": "no_history_investment_game",
            "amount_sent": 5,
            "tripled_amount": 15
        }
        prompt = self.room_b_builder.build_trial_prompt(room_b_trial)
        output_path = Path(output_dir) / f"study_012_room_b_example.txt"
        with open(output_path, "w") as f:
            f.write(prompt)
