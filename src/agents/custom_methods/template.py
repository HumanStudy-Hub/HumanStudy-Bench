"""
Template for creating custom system prompt methods.

To create your own method:
1. Copy this file and rename it (e.g., "v4_emotional.py", "my_custom_prompt.py")
2. Implement the generate_prompt function below
3. Place it in this directory (src/agents/custom_methods/)
4. The method will be automatically discovered and registered using the filename

Example usage:
    python scripts/run_v1_v2_pipeline.py --real-llm --presets v4_emotional
"""

from typing import Dict, Any


def generate_prompt(profile: Dict[str, Any]) -> str:
    """
    Generate a system prompt for an LLM participant agent.

    This function will be automatically discovered and registered by the
    SystemPromptRegistry. The filename (without .py) will be used as the
    preset name.

    Args:
        profile: Dictionary containing participant characteristics:
            - age: int - Participant's age
            - gender: str (optional) - Participant's gender
            - education: str - Education level (e.g., "college student")
            - background: str (optional) - Professional or personal background
            - Any other custom keys you add to the profile

    Returns:
        str: The complete system prompt to use for this participant

    Example:
        >>> profile = {"age": 25, "gender": "female", "education": "graduate student"}
        >>> prompt = generate_prompt(profile)
        >>> print(prompt)
    """
    # TODO: Implement your custom prompt logic here
    #
    # Example implementation:
    age = profile.get('age', 25)
    gender = profile.get('gender', 'unknown')

    return f"""You are a human participant in a psychology experiment.
You are {age} years old and identify as {gender}.

Follow the experimenter's instructions carefully and respond naturally.
"""
