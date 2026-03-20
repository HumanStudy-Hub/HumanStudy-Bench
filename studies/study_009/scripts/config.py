import json
import re
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from study_utils import BaseStudyConfig, PromptBuilder
from participant_pool import LLMParticipantAgent, ParticipantPool

import random
from typing import Dict, Any, List
from pathlib import Path

class CustomPromptBuilder(PromptBuilder):
    def __init__(self, study_path: Path):
        super().__init__(study_path)

    def build_trial_prompt(self, trial_metadata: Dict[str, Any]) -> str:
        """
        Builds a prompt for the p-Beauty Contest game.
        The participant is asked to provide choices for 4 consecutive rounds.
        (Legacy method - kept for backward compatibility)
        """
        instructions = trial_metadata.get("instructions", "")
        items = trial_metadata.get("items", [])
        # Each sub-study JSON contains one main item describing the task
        item = items[0]
        
        prompt = f"{instructions}\n\n"
        prompt += f"Task: {item['question']}\n\n"
        prompt += (
            "You will play this game for 4 consecutive rounds with the same group. "
            "Please provide the number you would choose for each round. "
            "Assume that after each round, you receive feedback about the mean and the winning number before making your next choice.\n\n"
        )
        
        prompt += "RESPONSE_SPEC: Please provide your choices in the following format:\n"
        prompt += "Q1=<number for Round 1>, Q2=<number for Round 2>, Q3=<number for Round 3>, Q4=<number for Round 4>"
        
        return prompt
    
    def build_round_prompt(
        self, 
        round_number: int, 
        previous_round_feedback: Optional[Dict[str, Any]], 
        trial_metadata: Dict[str, Any],
        p_value: float,
        participant_previous_choice: Optional[float] = None
    ) -> str:
        """
        Builds a prompt for a specific round with feedback from previous rounds.
        Designed for stateful conversation where instructions are only provided in Round 1.
        
        Args:
            round_number: Current round (1-4)
            previous_round_feedback: Dict with keys: mean, p_times_mean, winning_number
            trial_metadata: Trial metadata including instructions and items
            p_value: The parameter p (0.5, 0.66, or 1.33)
            participant_previous_choice: This participant's choice from the previous round
        """
        if round_number == 1:
            # Round 1: Provide full instructions and task
            instructions = trial_metadata.get("instructions", "")
            items = trial_metadata.get("items", [])
            item = items[0]
            
            prompt = f"{instructions}\n\n"
            prompt += f"Task: {item['question']}\n\n"
            prompt += (
                "The game begins now. All participants will choose a number simultaneously. "
                "After this round, you will receive feedback about the mean of all choices and the winning number.\n\n"
            )
            prompt += "Please provide your choice for Round 1. "
            prompt += "RESPONSE_SPEC: Your response MUST contain the chosen number in the format: Q1=<number>"
        else:
            # Rounds 2-4: Provide feedback and brief reminder
            prompt = "You are continuing the guessing game.\n\n"
            
            if previous_round_feedback:
                mean = previous_round_feedback.get("mean", 0)
                p_times_mean = previous_round_feedback.get("p_times_mean", 0)
                winning_number = previous_round_feedback.get("winning_number", 0)
                
                prompt += f"Results from Round {round_number - 1}:\n"
                prompt += f"- The average of all numbers chosen was: {mean:.2f}\n"
                prompt += f"- The target (p * mean) was: {p_value} * {mean:.2f} = {p_times_mean:.2f}\n"
                prompt += f"- The winning choice (closest to target) was: {winning_number:.2f}\n"
                if participant_previous_choice is not None:
                    prompt += f"- Your previous choice was: {participant_previous_choice:.2f}\n"
                prompt += "\n"
            
            prompt += f"Now, please make your choice for Round {round_number}.\n"
            if round_number < 4:
                prompt += f"After this round, you will receive feedback before Round {round_number + 1}.\n\n"
            else:
                prompt += "This is the final round.\n\n"
            
            prompt += f"RESPONSE_SPEC: Your response MUST contain the chosen number in the format: Q{round_number}=<number>"
        
        return prompt

class StudyStudy009Config(BaseStudyConfig):
    prompt_builder_class = CustomPromptBuilder
    PROMPT_VARIANT = "standard"
    REQUIRES_GROUP_TRIALS = True  # Flag to signal pipeline to use custom runner

    def __init__(self, study_path: Path, specification: Dict[str, Any]):
        super().__init__(study_path, specification)
        # Map sub_study_id to p value
        self.p_value_map = {
            "p_0.5_condition": 0.5,
            "p_0.66_condition": 2/3,  # 0.666...
            "p_1.33_condition": 4/3   # 1.333...
        }

    def create_trials(self, n_trials: int = None) -> List[Dict[str, Any]]:
        """
        Creates trials for the three experimental conditions (p=1/2, p=2/3, p=4/3).
        Each trial represents one participant playing 4 rounds in a specific condition.
        """
        trials = []
        spec = self.load_specification()
        
        # Sub-study IDs based on the provided material files
        sub_study_ids = ["p_0.5_condition", "p_0.66_condition", "p_1.33_condition"]
        
        for sub_id in sub_study_ids:
            material = self.load_material(sub_id)
            
            # Get participant count from specification or use default
            n = spec["participants"]["by_sub_study"].get(sub_id, {}).get("n", 0)
            if n == 0:
                n = 50  # Default to ensure experiment runs
            
            # If n_trials is provided (usually for testing), override the per-condition count
            count = n if n_trials is None else n_trials
            
            for _ in range(count):
                # Each participant is assigned a profile and plays one condition
                trials.append({
                    "sub_study_id": sub_id,
                    "instructions": material.get("instructions", ""),
                    "items": material.get("items", []),
                    "profile": {
                        "age": random.randint(18, 35),
                        "gender": random.choice(["male", "female"]),
                        "background": "university student"
                    },
                    "variant": self.PROMPT_VARIANT
                })
                
        return trials

    def dump_prompts(self, output_dir: str):
        """
        Dumps a sample of generated prompts to the specified directory.
        """
        # Create one trial per sub-study for inspection
        trials = self.create_trials(n_trials=1)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for idx, trial in enumerate(trials):
            prompt = self.prompt_builder.build_trial_prompt(trial)
            file_name = f"study_009_{trial['sub_study_id']}_trial_{idx}.txt"
            with open(output_path / file_name, "w", encoding="utf-8") as f:
                f.write(prompt)

    def parse_response(self, response_text: str, trial_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses the LLM response to extract the numbers chosen for each round.
        """
        results = {}
        for i in range(1, 5):
            key = f"Q{i}"
            # Extract numeric value following the Qk= pattern
            try:
                part = response_text.split(f"{key}=")[-1].split(",")[0].strip()
                val = self.extract_numeric(part)
                results[key] = val
            except (IndexError, ValueError):
                results[key] = None
        return results
    
    def _calculate_round_feedback(
        self, 
        choices: Dict[int, float], 
        p_value: float
    ) -> Dict[str, Any]:
        """
        Calculate feedback for a round given all participants' choices.
        
        Args:
            choices: Dict mapping participant_id to their choice
            p_value: Parameter p for this condition
            
        Returns:
            Dict with mean, p_times_mean, winning_number, and all choices
        """
        if not choices:
            return {
                "mean": 0.0,
                "p_times_mean": 0.0,
                "winning_number": 0.0,
                "choices": {}
            }
        
        # Calculate mean
        choice_values = [v for v in choices.values() if v is not None]
        if not choice_values:
            return {
                "mean": 0.0,
                "p_times_mean": 0.0,
                "winning_number": 0.0,
                "choices": choices
            }
        
        mean = np.mean(choice_values)
        p_times_mean = p_value * mean
        
        # Find winning number (closest to p_times_mean)
        winning_number = min(choice_values, key=lambda x: abs(x - p_times_mean))
        
        return {
            "mean": float(mean),
            "p_times_mean": float(p_times_mean),
            "winning_number": float(winning_number),
            "choices": choices.copy()
        }
    
    def _group_participants_by_condition(
        self, 
        trials: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group trials by sub_study_id (condition).
        
        Args:
            trials: List of trial dictionaries
            
        Returns:
            Dict mapping sub_study_id to list of trials for that condition
        """
        groups = {}
        for trial in trials:
            sub_id = trial.get("sub_study_id")
            if sub_id not in groups:
                groups[sub_id] = []
            groups[sub_id].append(trial)
        return groups
    
    def _extract_choice_from_response(
        self, 
        response_text: str, 
        round_number: int
    ) -> Optional[float]:
        """
        Extract a single round's choice from response text.
        Supports various formats: Q1=50, Q1: 50, Round 1: 50, etc.
        """
        if not response_text:
            return None
        
        # Clean response text (remove any leading/trailing whitespace)
        response_text = response_text.strip()
        
        key = f"Q{round_number}"
        # Flexible pattern: Q1=50, Q1: 50, Q1 = 50.0
        patterns = [
            rf"{key}\s*[=:]\s*(\d+(?:\.\d+)?)",  # Q1=50 or Q1: 50
            rf"Round\s*{round_number}\s*[=:]\s*(\d+(?:\.\d+)?)",  # Round 1: 50
            rf"choice\s*[=:]\s*(\d+(?:\.\d+)?)",  # choice: 50
            rf"{key}\s*=\s*(\d+(?:\.\d+)?)",  # Q1=50 (strict)
            rf"{key}\s*:\s*(\d+(?:\.\d+)?)",  # Q1: 50 (strict)
        ]
        
        for p in patterns:
            match = re.search(p, response_text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # Last resort: just find any number in the response if it's short
        # This helps when LLM just says "50" or "25.5" without the Q1= prefix
        if len(response_text.strip()) < 50:  # Increased threshold for v1_empty responses
            nums = re.findall(r"(\d+(?:\.\d+)?)", response_text)
            if nums:
                # Prefer the first number that looks like a choice (0-100 range)
                for num_str in nums:
                    try:
                        num = float(num_str)
                        if 0 <= num <= 100:  # Valid range for p-beauty contest
                            return num
                    except ValueError:
                        continue
                # If no number in valid range, return the first number anyway
                try:
                    return float(nums[0])
                except ValueError:
                    pass
                
        return None
    
    def run_group_experiment(
        self,
        trials: List[Dict[str, Any]],
        instructions: str,
        participant_pool_kwargs: Dict[str, Any],
        prompt_builder: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run experiment with multi-participant groups and sequential rounds.
        
        This method:
        1. Groups participants by condition (sub_study_id)
        2. For each group, runs 4 rounds sequentially
        3. Calculates feedback from all participants' choices each round
        4. Provides feedback between rounds
        
        Args:
            trials: List of trial dictionaries (one per participant)
            instructions: Experimental instructions
            participant_pool_kwargs: Keyword arguments for ParticipantPool creation
            prompt_builder: Prompt builder instance
            
        Returns:
            Results dictionary with individual_data in expected format
        """
        if prompt_builder is None:
            prompt_builder = self.prompt_builder
        
        # Group trials by condition
        groups = self._group_participants_by_condition(trials)
        
        all_individual_data = []
        participant_id_offset = 0  # Track offset to ensure unique IDs across groups
        
        # Process each condition group independently
        # Split each condition into multiple sessions (15-18 participants per session, matching human experiment)
        for sub_study_id, group_trials in groups.items():
            p_value = self.p_value_map.get(sub_study_id, 0.5)
            n_participants_total = len(group_trials)
            
            # Split into sessions: 15-18 participants per session (matching human experiment design)
            # Use random session size between 15-18 to match human variability
            import random
            session_size_min = 15
            session_size_max = 18
            
            # Create sessions
            sessions = []
            remaining_trials = group_trials.copy()
            while remaining_trials:
                # Random session size between 15-18
                session_size = random.randint(session_size_min, session_size_max)
                session_size = min(session_size, len(remaining_trials))  # Don't exceed remaining
                
                session_trials = remaining_trials[:session_size]
                remaining_trials = remaining_trials[session_size:]
                sessions.append(session_trials)
            
            print(f"Running group experiment for {sub_study_id} with {n_participants_total} participants in {len(sessions)} sessions")
            
            # Pre-calculate participant ID offsets for each session (needed for parallel execution)
            session_id_offsets = []
            current_offset = participant_id_offset
            for session_trials in sessions:
                session_id_offsets.append(current_offset)
                current_offset += len(session_trials)
            
            # Define function to process a single session (for parallelization)
            def process_session(session_data):
                """Process a single session and return its results."""
                session_idx, session_trials, session_participant_id_offset = session_data
                n_participants = len(session_trials)
                print(f"  Session {session_idx + 1}/{len(sessions)}: {n_participants} participants")
                
                session_results = []
                
                # Extract profiles from trials
                group_profiles = [trial.get("profile", {}) for trial in session_trials]
                
                # Create participant pool for this session with profiles from trials
                pool_kwargs = participant_pool_kwargs.copy()
                # Remove study_specification since we pass it explicitly
                pool_kwargs.pop("study_specification", None)
                pool_kwargs["profiles"] = group_profiles
                pool_kwargs["n_participants"] = n_participants
                
                pool = ParticipantPool(
                    study_specification=self.specification,
                    **pool_kwargs
                )
                
                # Ensure participant IDs are sequential and unique across all groups and sessions
                for idx, (participant, trial) in enumerate(zip(pool.participants, session_trials)):
                    participant.participant_id = session_participant_id_offset + idx
                    participant.profile["participant_id"] = session_participant_id_offset + idx
                    # Update profile with any additional trial metadata
                    if "profile" in trial:
                        participant.profile.update(trial["profile"])
                
                # Initialize conversation sessions for all participants in this session
                system_prompt = pool.participants[0]._construct_system_prompt() if pool.participants else ""
                for participant in pool.participants:
                    participant.start_conversation(system_prompt)
                
                # Store round-by-round choices and usage for each participant in this session
                participant_choices = {p.participant_id: {} for p in pool.participants}
                participant_usage = {p.participant_id: {
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0
                } for p in pool.participants}
                # Store raw responses for each round
                participant_round_responses = {p.participant_id: {} for p in pool.participants}
                round_feedbacks = []
                
                # Run 4 rounds sequentially for this session
                previous_feedback = None
                
                for round_num in range(1, 5):
                    print(f"    Round {round_num}...")
                    
                    # Build prompts for this round with previous feedback
                    round_prompts = {}
                    for participant, trial in zip(pool.participants, session_trials):
                        trial_with_profile = {**trial, "participant_profile": participant.profile}
                        # Get this participant's previous round choice
                        prev_choice = None
                        if round_num > 1:
                            prev_choice = participant_choices[participant.participant_id].get(f"Q{round_num - 1}")
                        
                        round_prompts[participant.participant_id] = prompt_builder.build_round_prompt(
                            round_number=round_num,
                            previous_round_feedback=previous_feedback,
                            trial_metadata=trial_with_profile,
                            p_value=p_value,
                            participant_previous_choice=prev_choice
                        )
                    
                    # Collect choices from all participants simultaneously
                    round_choices = {}
                    
                    # Run all participants in parallel for this round using stateful conversation
                    def get_round_choice(participant, prompt):
                        """Get a single participant's choice for this round using stateful conversation."""
                        try:
                            # Use continue_conversation for stateful interaction
                            # Use 8192 tokens (4096*2) to allow for full responses including explanations/reasoning
                            result = participant.continue_conversation(prompt, max_tokens=8192)
                            response_text = result["response_text"]
                            usage = result.get("usage", {})
                            full_api_response = result.get("full_api_response", {})
                            
                            # Log first few responses for debugging (especially for v1_empty)
                            if participant.participant_id < 2:
                                print(f"      [Debug] P{participant.participant_id} R{round_num} response: {response_text[:100]}...")
                            
                            choice = self._extract_choice_from_response(response_text, round_num)
                            
                            # If extraction failed, log the full response for debugging
                            if choice is None and participant.participant_id < 2:
                                print(f"      [Warning] P{participant.participant_id} R{round_num} failed to extract choice from: {response_text}")
                            
                            return participant.participant_id, choice, usage, response_text, full_api_response
                        except Exception as e:
                            print(f"      Warning: Participant {participant.participant_id} failed in round {round_num}: {e}")
                            import traceback
                            traceback.print_exc()
                            return participant.participant_id, None, {}, "", {}
                    
                    # Execute all participants in parallel
                    with ThreadPoolExecutor(max_workers=participant_pool_kwargs.get("num_workers", 1)) as executor:
                        futures = {
                            executor.submit(get_round_choice, p, round_prompts[p.participant_id]): p.participant_id
                            for p in pool.participants
                        }
                        
                        for future in as_completed(futures):
                            pid, choice, usage, response_text, full_api_response = future.result()
                            round_choices[pid] = choice
                            participant_choices[pid][f"Q{round_num}"] = choice
                            # Store raw response for this round
                            participant_round_responses[pid][f"round_{round_num}"] = {
                                "response_text": response_text,
                                "raw_response_text": response_text,  # Keep for compatibility
                                "full_api_response": full_api_response,
                                "usage": usage,
                                "extracted_choice": choice
                            }
                            # Accumulate usage
                            for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                                participant_usage[pid][key] += usage.get(key, 0)
                            participant_usage[pid]["cost"] += usage.get("cost", 0.0)
                    
                    # Calculate feedback for this round
                    round_feedback = self._calculate_round_feedback(round_choices, p_value)
                    
                    # Store round feedback
                    round_feedbacks.append({
                        "round": round_num,
                        "feedback": round_feedback
                    })
                    
                    # Prepare feedback for next round (will be customized per participant)
                    # The build_round_prompt will add each participant's own choice
                    if round_num < 4:
                        previous_feedback = {
                            "mean": round_feedback["mean"],
                            "p_times_mean": round_feedback["p_times_mean"],
                            "winning_number": round_feedback["winning_number"]
                        }
                
                # Clear conversation history for all participants in this session (cleanup)
                # This should happen AFTER all rounds are complete
                for participant in pool.participants:
                    participant.clear_conversation()
                
                # Aggregate results per participant in expected format for this session
                for participant in pool.participants:
                    pid = participant.participant_id
                    choices = participant_choices[pid]
                    
                    # Build response text in Q1=, Q2=, Q3=, Q4= format for evaluator compatibility
                    response_parts = []
                    for i in range(1, 5):
                        choice = choices.get(f'Q{i}')
                        if choice is not None:
                            response_parts.append(f"Q{i}={choice}")
                        else:
                            response_parts.append(f"Q{i}=N/A")
                    response_text = ", ".join(response_parts)
                    
                    # Find corresponding trial
                    trial = next((t for t in session_trials if t.get("profile", {}).get("participant_id") == pid), None)
                    if trial is None:
                        # Fallback: match by index
                        trial_idx = pool.participants.index(participant) if participant in pool.participants else 0
                        trial = session_trials[trial_idx] if trial_idx < len(session_trials) else {}
                    
                    individual_data = {
                        "participant_id": pid,
                        "profile": participant.profile,
                        "responses": [{
                            "response_text": response_text,
                            "raw_response_text": response_text,  # Keep for compatibility
                            "usage": participant_usage[pid],
                            "round_responses": participant_round_responses[pid],  # Store all round-by-round responses
                            "trial_info": {
                                "sub_study_id": sub_study_id,
                                "session_id": f"{sub_study_id}_session_{session_idx}",  # Unique session identifier
                                "session_idx": session_idx,  # Also save index for sorting
                                "round_feedbacks": round_feedbacks.copy()  # Same feedback for all participants in session
                            }
                        }]
                    }
                    session_results.append(individual_data)
                
                return session_results
            
            # Parallelize session execution
            session_data_list = [
                (idx, session_trials, session_id_offsets[idx])
                for idx, session_trials in enumerate(sessions)
            ]
            
            # Use ThreadPoolExecutor to run sessions in parallel
            max_workers = participant_pool_kwargs.get("num_workers", 1)
            # For sessions, we can use more workers (sessions are independent)
            # But limit to number of sessions to avoid overhead
            session_workers = min(max_workers, len(sessions), 10)  # Cap at 10 to avoid too many threads
            
            with ThreadPoolExecutor(max_workers=session_workers) as executor:
                futures = {
                    executor.submit(process_session, data): data[0]
                    for data in session_data_list
                }
                
                for future in as_completed(futures):
                    session_idx = futures[future]
                    try:
                        session_results = future.result()
                        all_individual_data.extend(session_results)
                    except Exception as e:
                        print(f"  Error in session {session_idx + 1}: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Update participant_id_offset for next condition
            participant_id_offset = current_offset
        
        # Sort by participant_id
        all_individual_data.sort(key=lambda x: x.get('participant_id', 0))
        
        return {
            "individual_data": all_individual_data
        }
