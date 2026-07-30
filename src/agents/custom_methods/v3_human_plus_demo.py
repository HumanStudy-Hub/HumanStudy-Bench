"""
V3: Human Participant with Demographics

This prompt includes the basic human participant instruction plus demographic
information (age, gender, education/background) to give the LLM a specific identity.

Usage:
    python scripts/run_v1_v2_pipeline.py --real-llm --presets v3_human_plus_demo
"""

from typing import Dict, Any


def generate_prompt(profile: Dict[str, Any]) -> str:
    """
    Generate a system prompt with demographic information.

    Args:
        profile: Dictionary containing participant characteristics:
            - age: int - Participant's age
            - gender: str (optional) - Participant's gender
            - education: str - Education level (e.g., "college student")
            - background: str (optional) - Professional or personal background

    Returns:
        str: Complete system prompt with identity information
    """
    base_sentence = "You are participating in a psychology experiment as a human participant."

    age = profile.get('age', 'unknown age')
    gender = profile.get('gender')
    education = profile.get('education', 'college student')
    background = profile.get('background')

    identity_parts = [f"- Age: {age} years old"]
    if gender is not None:
        identity_parts.append(f"- Gender: {gender}")
    if background:
        identity_parts.append(f"- Background: {background}")
    else:
        identity_parts.append(f"- Education: {education}")

    identity_section = "\n".join(identity_parts)

    return f"""{base_sentence}

YOUR IDENTITY:
{identity_section}

Follow the experimenter's instructions and answer each task in the requested format.
Be concise. Do not add extra explanations unless explicitly asked."""
