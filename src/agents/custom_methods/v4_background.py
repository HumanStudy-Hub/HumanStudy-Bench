"""
V4 Background: Generative Agents-style Background Prompt

This prompt method loads Generative Agents-style background descriptions
that were pre-generated and stored for each participant.

The method automatically loads backgrounds from data/backgrounds/{study_id}/
based on participant_id. If a background is not found, it raises a
FileNotFoundError with instructions on how to generate the backgrounds.

Usage:
    1. First generate backgrounds:
       python scripts/generate_study_backgrounds.py --study study_001

    2. Then use this preset in experiments:
       python scripts/run_baseline_pipeline.py --real-llm --presets v4_background
       OR
       python generation_pipeline/pipeline.py run_stage5 --study study_001 --system-prompt-preset v4_background
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path
import os


# Cache for loaded backgrounds
_background_cache: Dict[str, Dict[str, Any]] = {}


def _load_background(study_id: str, participant_id: int, storage_dir: Path = None) -> Optional[str]:
    """
    Load a stored background for a participant.

    Args:
        study_id: Study identifier
        participant_id: Participant ID
        storage_dir: Storage directory (default: data/backgrounds)

    Returns:
        Background text or None if not found
    """
    if storage_dir is None:
        # Try to find storage dir relative to project root
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        storage_dir = project_root / "data" / "backgrounds"

    # Check cache first
    cache_key = f"{study_id}_{participant_id}"
    if cache_key in _background_cache:
        return _background_cache[cache_key].get("background")

    # Load from file
    background_path = storage_dir / study_id / f"participant_{participant_id:04d}.json"

    if not background_path.exists():
        # Try to find study_id from common patterns
        # Sometimes study_id might be in different format
        for possible_study_id in [study_id, study_id.replace("_", "-"), study_id.replace("-", "_")]:
            possible_path = storage_dir / possible_study_id / f"participant_{participant_id:04d}.json"
            if possible_path.exists():
                background_path = possible_path
                break
        else:
            return None

    try:
        with open(background_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            background = data.get("background", "")
            # Cache it
            _background_cache[cache_key] = data
            return background
    except Exception:
        return None


def generate_prompt(profile: Dict[str, Any]) -> str:
    """
    Generate a system prompt with Generative Agents-style background.

    Args:
        profile: Dictionary containing participant characteristics:
            - age: int - Participant's age
            - gender: str (optional) - Participant's gender
            - education: str - Education level
            - background: str (optional) - If provided, use this directly
            - study_id: str (optional) - Study identifier for loading stored background
            - participant_id: int (optional) - Participant ID for loading stored background
            - generative_background: str (optional) - Pre-loaded Generative Agents background

    Returns:
        str: Complete system prompt with Generative Agents-style background

    Raises:
        FileNotFoundError: If background file is not found for the participant.
                           The error message includes instructions on how to generate backgrounds.
    """
    base_sentence = "You are participating in a psychology experiment as a human participant."

    # Check if Generative Agents background is already provided
    background = profile.get('generative_background') or profile.get('background')

    # If not provided, try to load from storage
    if not background:
        study_id = profile.get('study_id')
        participant_id = profile.get('participant_id')

        # Try to extract study_id from environment or profile
        if not study_id:
            # Check environment variable (set by study runner)
            study_id = os.getenv('CURRENT_STUDY_ID')

        if study_id and participant_id is not None:
            background = _load_background(study_id, participant_id)

    # If still no background, raise an error
    if not background:
        study_id = profile.get('study_id', 'unknown')
        participant_id = profile.get('participant_id', 'unknown')
        background_path = Path(__file__).parent.parent.parent.parent / "data" / "backgrounds" / study_id / f"participant_{participant_id:04d}.json"

        raise FileNotFoundError(
            f"v4_background: Background not found for participant {participant_id} in study {study_id}. "
            f"Expected file: {background_path}\n"
            f"Please generate backgrounds first using: "
            f"python scripts/generate_study_backgrounds.py --study {study_id}"
        )

    # Use Generative Agents-style background
    # Split into memories if it's semicolon-delimited
    if ";" in background:
        # Format as memories
        memories = [m.strip() for m in background.split(";") if m.strip()]
        memories_section = "\n".join([f"- {memory}" for memory in memories[:10]])  # Limit to 10 memories
        if len(memories) > 10:
            memories_section += f"\n- ... and {len(memories) - 10} more memories"
    else:
        # Single paragraph format
        memories_section = background

    return f"""{base_sentence}

YOUR BACKGROUND AND MEMORIES:
{memories_section}

Based on your background and memories above, respond as this participant would in the experiment.
Follow the experimenter's instructions and answer each task in the requested format.
Be concise. Do not add extra explanations unless explicitly asked.
Your responses should reflect your background, experiences, and characteristics as described above."""
