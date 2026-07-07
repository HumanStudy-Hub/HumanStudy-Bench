"""
Custom Methods Directory

This directory is for user-defined system prompt methods.

Any Python file in this directory that contains a `generate_prompt` function
will be automatically discovered and registered by the SystemPromptRegistry.

The filename (without .py extension) will be used as the preset name.

Example:
    - File: v4_emotional.py
    - Function: generate_prompt(profile: Dict[str, Any]) -> str
    - Usage: --presets v4_emotional
"""
