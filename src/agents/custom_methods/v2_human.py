"""
V2: Basic Human Participant Prompt

This prompt provides a simple instruction that the LLM is participating as a human.
No demographic information is included.

Usage:
    python scripts/run_v1_v2_pipeline.py --real-llm --presets v2_human
"""

from typing import Dict, Any


def generate_prompt(profile: Dict[str, Any]) -> str:
    """
    Generate a basic human participant system prompt.

    Args:
        profile: Dictionary containing participant characteristics (unused for v2)

    Returns:
        str: Basic instruction to act as a human participant
    """
    return "You are participating in a psychology experiment as a human participant."
