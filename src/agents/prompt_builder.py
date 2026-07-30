"""
Prompt Builder - Convert study specifications into LLM prompts.

This module handles the critical translation from experimental design (specification.json)
to actual prompts that LLM agents receive as participants.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional


class PromptBuilder:
    """
    Base class for building prompts from study specifications.

    Key responsibility: Transform technical specification.json
    into natural language prompts for LLM participants.
    """

    def __init__(self, study_path: Path):
        """
        Initialize prompt builder with study materials.

        Args:
            study_path: Path to study directory (e.g., data/studies/study_003/)
        """
        self.study_path = Path(study_path)
        self.materials_path = self.study_path / "materials"

        # Load specification
        with open(self.study_path / "specification.json", "r", encoding='utf-8', errors='replace') as f:
            self.specification = json.load(f)

        # Load instructions
        instructions_file = self.materials_path / "instructions.txt"
        if instructions_file.exists():
            with open(instructions_file, "r", encoding='utf-8', errors='replace') as f:
                self.instructions = f.read()
        else:
            self.instructions = None

        # Load custom system prompt template (optional)
        system_prompt_file = self.materials_path / "system_prompt.txt"
        if system_prompt_file.exists():
            with open(system_prompt_file, "r", encoding='utf-8', errors='replace') as f:
                self.system_prompt_template = f.read()
        else:
            self.system_prompt_template = None

    def build_system_prompt(self, participant_profile: Dict[str, Any] = None) -> Optional[str]:
        """
        Get the custom system prompt content if it exists.

        Note: Custom system prompt is appended as-is without template variable substitution.
        Age, gender, and other profile information are already included in the default
        system prompt, so the custom content should be additional instructions only.

        Args:
            participant_profile: Not used (kept for compatibility), custom prompt is used as-is

        Returns:
            Custom system prompt content string, or None if not provided
        """
        # Return custom system prompt template as-is (no template variable substitution)
        return self.system_prompt_template

    def get_system_prompt_template(self) -> Optional[str]:
        """
        Get the custom system prompt template if it exists.

        Returns:
            Custom system prompt template string, or None if not provided
        """
        return self.system_prompt_template

    def build_trial_prompt(self, trial_data: Dict[str, Any]) -> str:
        """
        Build the prompt for a single trial.

        Args:
            trial_data: Trial-specific information (stimuli, confederate responses, etc.)
                       May include 'participant_profile' for frame-specific prompts

        Returns:
            Complete trial prompt string
        """
        return self._build_generic_trial_prompt(trial_data)

    def get_instructions(self) -> str:
        """
        Get the experimental instructions.

        Returns:
            Instructions text (from materials/instructions.txt)
        """
        return self.instructions if self.instructions else "No instructions provided."

    def _fill_template(self, template: str, data: Dict[str, Any]) -> str:
        """
        Fill template with data using simple {{variable}} syntax.

        Supports:
        - {{variable}}: Simple substitution
        - {{object.key}}: Nested property access
        - {{#if variable}}...{{/if}}: Conditional blocks
        - {{#each array}}{{this}}{{/each}}: Loops (simplified)
        """
        result = template

        # Handle nested property access (e.g., {{comparison_lines.A}})
        nested_pattern = r'\{\{([\w.]+)\}\}'

        def replace_nested(match):
            path = match.group(1)
            parts = path.split('.')

            value = data
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return match.group(0)  # Keep original if not found

            return str(value)

        result = re.sub(nested_pattern, replace_nested, result)

        # Handle conditional blocks (simplified)
        # {{#if variable}}...{{/if}}
        if_pattern = r'\{\{#if\s+(\w+)\}\}(.*?)\{\{/if\}\}'

        def replace_if(match):
            var_name = match.group(1)
            content = match.group(2)
            # Check if variable exists and is truthy
            if var_name in data and data[var_name]:
                return content
            return ""

        result = re.sub(if_pattern, replace_if, result, flags=re.DOTALL)

        # Handle each loops (simplified)
        # {{#each array}}{{this}}{{/each}}
        each_pattern = r'\{\{#each\s+(\w+)\}\}(.*?)\{\{/each\}\}'

        def replace_each(match):
            var_name = match.group(1)
            content = match.group(2)

            if var_name not in data:
                return ""

            items = data[var_name]
            if isinstance(items, dict):
                # Dictionary: replace {{@key}} and {{this}}
                parts = []
                for key, value in items.items():
                    item_content = content.replace("{{@key}}", str(key))
                    item_content = item_content.replace("{{this}}", str(value))
                    parts.append(item_content)
                return "\n".join(parts)
            elif isinstance(items, list):
                # List: replace {{this}} and {{@index}}
                parts = []
                for idx, item in enumerate(items):
                    item_content = content.replace("{{@index}}", str(idx + 1))
                    item_content = item_content.replace("{{this}}", str(item))
                    parts.append(item_content)
                return "\n".join(parts)
            return ""

        result = re.sub(each_pattern, replace_each, result, flags=re.DOTALL)

        # Clean up any remaining unfilled placeholders
        result = re.sub(r'\{\{[^}]+\}\}', '', result)

        return result

    def _build_generic_system_prompt(self, profile: Dict[str, Any]) -> str:
        """Fallback generic system prompt."""
        age = profile.get('age', 'unknown age')
        gender = profile.get('gender', 'unspecified gender')

        return f"""You are participating in a psychology experiment as a real human participant.

Your identity: {age} years old, {gender}

Respond naturally as this person would. Do not explain your reasoning - just give direct responses as a real participant would."""

    def _build_generic_trial_prompt(self, trial_data: Dict[str, Any]) -> str:
        """Fallback generic trial prompt."""
        return f"Trial {trial_data.get('trial_number', '?')}: Please respond to the following stimulus."


def create_prompt_builder(study_path: Path) -> PromptBuilder:
    """
    Factory function to create PromptBuilder for a study.

    Args:
        study_path: Path to study directory

    Returns:
        PromptBuilder instance
    """
    return PromptBuilder(study_path)

# Convenience function for users
def get_prompt_builder(study_id: str, data_dir: str = "data") -> PromptBuilder:
    """
    Get prompt builder for a study by ID.

    Args:
        study_id: Study identifier (e.g., "study_003")
        data_dir: Path to data directory

    Returns:
        PromptBuilder instance

    Example:
        >>> builder = get_prompt_builder("study_003")
        >>> system_prompt = builder.build_system_prompt({"age": 20, "education": "university_student"})
        >>> trial_prompt = builder.build_trial_prompt({"trial_number": 1, ...})
    """
    study_path = Path(data_dir) / "studies" / study_id
    return create_prompt_builder(study_path)
