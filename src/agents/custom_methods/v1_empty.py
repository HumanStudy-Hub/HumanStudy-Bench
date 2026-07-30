"""
V1: Empty System Prompt

This is the baseline "empty" prompt - no system message is provided to the LLM.
This serves as a control condition to measure the effect of system prompts.

Usage:
    python scripts/run_v1_v2_pipeline.py --real-llm --presets v1_empty
    # Also available as "empty" for backward compatibility
"""

from typing import Dict, Any


def generate_prompt(profile: Dict[str, Any]) -> str:
    """
    Generate an empty system prompt (baseline control condition).

    Args:
        profile: Dictionary containing participant characteristics (unused for v1)

    Returns:
        str: Empty string (no system prompt)
    """
    return ""
