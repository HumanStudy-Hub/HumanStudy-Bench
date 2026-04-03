"""
Standalone study utilities: BaseStudyConfig and PromptBuilder.
No dependency on src/; for use within each study's scripts/.
"""
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional


class PromptBuilder:
    """Build prompts from study specification and materials. study_path = source directory."""

    def __init__(self, study_path: Path):
        self.study_path = Path(study_path)
        self.materials_path = self.study_path / "materials"
        with open(self.study_path / "specification.json", "r", encoding="utf-8", errors="replace") as f:
            self.specification = json.load(f)
        instructions_file = self.materials_path / "instructions.txt"
        self.instructions = instructions_file.read_text(encoding="utf-8", errors="replace") if instructions_file.exists() else None
        system_prompt_file = self.materials_path / "system_prompt.txt"
        self.system_prompt_template = system_prompt_file.read_text(encoding="utf-8", errors="replace") if system_prompt_file.exists() else None

    def build_system_prompt(self, participant_profile: Dict[str, Any] = None) -> Optional[str]:
        return self.system_prompt_template

    def get_system_prompt_template(self) -> Optional[str]:
        return self.system_prompt_template

    def build_trial_prompt(self, trial_data: Dict[str, Any]) -> str:
        return self._build_generic_trial_prompt(trial_data)

    def get_instructions(self) -> str:
        return self.instructions if self.instructions else "No instructions provided."

    def _fill_template(self, template: str, data: Dict[str, Any]) -> str:
        result = template
        nested_pattern = r"\{\{([\w.]+)\}\}"
        def replace_nested(match):
            path = match.group(1)
            value = data
            for part in path.split("."):
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return match.group(0)
            return str(value)
        result = re.sub(nested_pattern, replace_nested, result)
        if_pattern = r"\{\{#if\s+(\w+)\}\}(.*?)\{\{/if\}\}"
        def replace_if(match):
            if match.group(1) in data and data[match.group(1)]:
                return match.group(2)
            return ""
        result = re.sub(if_pattern, replace_if, result, flags=re.DOTALL)
        each_pattern = r"\{\{#each\s+(\w+)\}\}(.*?)\{\{/each\}\}"
        def replace_each(match):
            var_name, content = match.group(1), match.group(2)
            if var_name not in data:
                return ""
            items = data[var_name]
            if isinstance(items, dict):
                parts = [content.replace("{{@key}}", str(k)).replace("{{this}}", str(v)) for k, v in items.items()]
                return "\n".join(parts)
            if isinstance(items, list):
                parts = [content.replace("{{@index}}", str(i + 1)).replace("{{this}}", str(item)) for i, item in enumerate(items)]
                return "\n".join(parts)
            return ""
        result = re.sub(each_pattern, replace_each, result, flags=re.DOTALL)
        result = re.sub(r"\{\{[^}]+\}\}", "", result)
        return result

    def _build_generic_trial_prompt(self, trial_data: Dict[str, Any]) -> str:
        return f"Trial {trial_data.get('trial_number', '?')}: Please respond to the following stimulus."


class BaseStudyConfig(ABC):
    """Study config base. study_path = study root (e.g. studies/study_001); data under source/."""

    prompt_builder_class = PromptBuilder

    def __init__(self, study_path: Path, specification: Dict[str, Any]):
        self.study_path = Path(study_path)
        self.source_path = self.study_path / "source"
        self.specification = specification
        self.study_id = specification["study_id"]
        self.prompt_builder = self.prompt_builder_class(self.source_path)

    def load_material(self, sub_study_id: str) -> Dict[str, Any]:
        file_path = self.source_path / "materials" / f"{sub_study_id}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Material not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_metadata(self) -> Dict[str, Any]:
        with open(self.source_path / "metadata.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def load_specification(self) -> Dict[str, Any]:
        with open(self.source_path / "specification.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def load_ground_truth(self) -> Dict[str, Any]:
        with open(self.source_path / "ground_truth.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def extract_numeric(self, text: str, default: float = 0.0) -> float:
        if text is None:
            return default
        match = re.search(r"(-?\d+\.?\d*)", str(text))
        return float(match.group(1)) if match else default

    def extract_choice(self, text: str, options: List[str] = None) -> Optional[int]:
        if text is None:
            return None
        text_s = str(text).strip()
        if options:
            for i, opt in enumerate(options):
                if opt.lower() in text_s.lower():
                    return i
        match = re.search(r"\b([A-Z])\b", text_s.upper())
        if match:
            return ord(match.group(1)) - ord("A")
        return None

    @abstractmethod
    def create_trials(self, n_trials: Optional[int] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_prompt_builder(self) -> PromptBuilder:
        return self.prompt_builder

    def get_instructions(self) -> str:
        return self.prompt_builder.get_instructions()

    def aggregate_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        return raw_results

    def custom_scoring(self, results: Dict[str, Any], ground_truth: Dict[str, Any]) -> Optional[Dict[str, float]]:
        return None

    def get_n_participants(self) -> int:
        return self.specification["participants"]["n"]

    def get_study_type(self) -> str:
        return self.specification.get("study_type", self.study_id)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(study_id='{self.study_id}')"
