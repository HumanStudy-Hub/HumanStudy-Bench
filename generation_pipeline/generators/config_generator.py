"""
Config Generator - Generates StudyConfig classes from extraction results using LLM
"""

import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.llm.factory import get_client


class ConfigGenerator:
    """Generate StudyConfig classes from extraction results using LLM"""

    def __init__(
        self,
        provider: str = "gemini",
        model: str = "models/gemini-3-flash-preview",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        """
        Initialize config generator.

        Args:
            provider: One of gemini, openai, anthropic, xai, openrouter
            model: Model name
            api_key: Optional API key
            api_base: Optional API base URL
        """
        self.client = get_client(provider=provider, model=model, api_key=api_key, api_base=api_base)

    def generate(
        self,
        extraction_result: Dict[str, Any],
        study_id: str,
        output_path: Path,
        pdf_path: Optional[Path] = None,
        study_dir: Optional[Path] = None,
        standalone: bool = False,
    ) -> Path:
        """
        Generate StudyConfig class file using a combination of Template and LLM.
        """
        studies = extraction_result.get('studies', [])
        if not studies:
            raise ValueError("No studies found in extraction result")

        if pdf_path is None and study_dir:
            pdf_files = list(study_dir.glob("*.pdf"))
            if pdf_files:
                pdf_path = pdf_files[0]

        study_context = {}
        material_previews = []
        if study_dir:
            for json_file in ["metadata.json", "specification.json", "ground_truth.json"]:
                p = study_dir / json_file
                if p.exists():
                    try:
                        study_context[json_file] = json.loads(p.read_text(encoding='utf-8'))
                    except: pass

            materials_dir = study_dir / "materials"
            if materials_dir.exists():
                for json_file in materials_dir.glob("*.json"):
                    try:
                        content = json_file.read_text(encoding='utf-8')
                        preview = "\n".join(content.splitlines()[:20])
                        material_previews.append(f"FILE: {json_file.name}\n{preview}\n...")
                    except: pass

        material_context = "\n\n".join(material_previews)

        prompt = self._build_logic_only_prompt(
            json.dumps(extraction_result, indent=2, ensure_ascii=False),
            study_id,
            json.dumps(study_context, indent=2, ensure_ascii=False),
            material_context
        )

        try:
            response = self.client.generate_content(prompt=prompt)
        except Exception as e:
            raise RuntimeError(f"LLM Error: {e}")

        logic_code = self._extract_code_from_response(response)
        final_code = self._assemble_final_code(logic_code, study_id, standalone)
        validation_error = self._validate_generated_code(final_code)
        if validation_error:
            repair_prompt = self._build_repair_prompt(
                study_id=study_id,
                logic_code=logic_code,
                validation_error=validation_error,
            )
            try:
                repair_response = self.client.generate_content(prompt=repair_prompt)
            except Exception as e:
                raise RuntimeError(f"LLM config repair error: {e}") from e

            logic_code = self._extract_code_from_response(repair_response)
            final_code = self._assemble_final_code(logic_code, study_id, standalone)
            validation_error = self._validate_generated_code(final_code)
            if validation_error:
                raise ValueError(
                    "Generated StudyConfig remained invalid after one repair attempt: "
                    f"{validation_error}"
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_code, encoding='utf-8')
        return output_path

    def _assemble_final_code(self, logic_code: str, study_id: str, standalone: bool) -> str:
        class_name = f"Study{study_id.replace('_', '').capitalize()}Config"
        if standalone:
            imports = """import json
import os
import random
import re
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Any, List, Optional

from study_utils import BaseStudyConfig, PromptBuilder
"""
        else:
            imports = """import json
import os
import random
import re
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.core.study_config import BaseStudyConfig, StudyConfigRegistry
from src.agents.prompt_builder import PromptBuilder
"""

        final_code = f"""{imports}
{logic_code}
"""
        if f"class {class_name}" not in final_code:
            final_code = re.sub(r"class StudyConfig\s*:", f"class {class_name}(BaseStudyConfig):", final_code)
            final_code = re.sub(r"class \w+Config\s*:", f"class {class_name}(BaseStudyConfig):", final_code)
            final_code = re.sub(r"class \w+Config\(BaseStudyConfig\)\s*:", f"class {class_name}(BaseStudyConfig):", final_code)

        if standalone:
            final_code = re.sub(r"^@StudyConfigRegistry\.register\([^\n]+\)\n", "", final_code, flags=re.MULTILINE)
            final_code = final_code.replace(
                "from src.core.study_config import BaseStudyConfig, StudyConfigRegistry",
                "from study_utils import BaseStudyConfig, PromptBuilder",
            )
            final_code = final_code.replace(
                "from src.agents.prompt_builder import PromptBuilder\n",
                "",
            )
            final_code = final_code.replace("StudyConfigRegistry.register", "# StudyConfigRegistry.register")
        elif f'@StudyConfigRegistry.register("{study_id}")' not in final_code:
            final_code = final_code.replace(f"class {class_name}", f'@StudyConfigRegistry.register("{study_id}")\nclass {class_name}')

        return final_code

    @staticmethod
    def _validate_generated_code(code: str) -> Optional[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"syntax error at line {exc.lineno}: {exc.msg}"

        config_classes = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            base_names = {
                base.id
                for base in node.bases
                if isinstance(base, ast.Name)
            }
            if "BaseStudyConfig" in base_names:
                config_classes.append(node)

        if not config_classes:
            return "missing a class that subclasses BaseStudyConfig"
        if not any(
            isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            and member.name == "create_trials"
            for config_class in config_classes
            for member in config_class.body
        ):
            return "BaseStudyConfig subclass is missing create_trials()"
        return None

    @staticmethod
    def _build_repair_prompt(study_id: str, logic_code: str, validation_error: str) -> str:
        return f"""You generated invalid core Python logic for a HumanStudyBench StudyConfig.

STUDY ID: {study_id}
VALIDATION ERROR: {validation_error}

Repair the code while preserving its study design, canonical material IDs, trial structure,
and response format. Escape line breaks inside Python string literals as \\n. The corrected
code must define a class that subclasses BaseStudyConfig and implements create_trials().
Return only the corrected core logic. Do not include imports, markdown fences, or a registry
decorator.

INVALID CORE LOGIC:
{logic_code}
"""

    def _build_logic_only_prompt(self, extraction_summary, study_id, context_summary, material_context):
        # Use a template string and manual replacement to avoid f-string curly brace errors with JSON/Code
        template = """You are a Python expert for HumanStudyBench. Your task is to write the CORE LOGIC for `[[STUDY_ID]]_config.py`.

STUDY ID: [[STUDY_ID]]

### Core Principles
1. **Match the human experimental design exactly** - One trial per participant with all items (unless within-subjects explicitly requires multiple trials)
2. **Use class attributes** - `prompt_builder_class` and `PROMPT_VARIANT` must be class attributes, not instance attributes
3. **Never skip sub-studies** - If `n=0` in specification, use a default (e.g., `n=50`) to ensure all experiments run

### Available Methods (from BaseStudyConfig)
- `self.load_material(sub_id)` - Load material JSON (sub_id is filename without .json extension)
- `self.load_specification()` - Returns `{"participants": {"n": ..., "by_sub_study": {...}}, ...}`
- `self.load_ground_truth()` - Returns `{"studies": [{"findings": [...]}], ...}`
- `self.extract_numeric(text)`, `self.extract_choice(text, options)` - Parse responses

### Note on Findings
- The study's `metadata.json` contains a `findings` array with finding-level weights (used for evaluation aggregation)
- Each finding has a `finding_id` that matches the `finding_id` in `ground_truth.json`
- This information is primarily used by evaluators, not config generation

### EXTRACTION SUMMARY (Goal)
[[EXTRACTION_SUMMARY]]

### MATERIALS (Context)
[[MATERIAL_CONTEXT]]

### Working Examples

**Example 1: Simple Study (study_001)**
```python
class CustomPromptBuilder(PromptBuilder):
    def __init__(self, study_path: Path):
        super().__init__(study_path)

    def build_trial_prompt(self, trial_metadata):
        # Note: System prompt is now handled separately by SystemPromptRegistry
        # This method only builds the task/trial content
        items = trial_metadata.get("items", [])

        prompt = ""

        # Add task context introduction (similar to study_003/004)
        prompt += "You are participating in a psychology study on decision-making. Please read the following instructions and provide your responses.\n\n"

        # Build questions with Q indices
        q_counter = 1
        for item in items:
            prompt += f"Q{q_counter} (answer with letter: A or B): {item['question']}\n"
            item["q_idx"] = q_counter
            q_counter += 1

        # RESPONSE_SPEC
        prompt += "\nRESPONSE_SPEC: Output Q1=<A/B>, Q2=<A/B>, etc.\n"
        return prompt

@StudyConfigRegistry.register("study_001")
class StudyStudy001Config(BaseStudyConfig):
    prompt_builder_class = CustomPromptBuilder  # Class attribute
    PROMPT_VARIANT = "v3"  # Class attribute

    def __init__(self, study_path: Path, specification: Dict[str, Any]):
        super().__init__(study_path, specification)

    def create_trials(self, n_trials=None):
        trials = []
        material = self.load_material("study_1_hypothetical_stories")
        n = 80 if n_trials is None else n_trials

        for item in material["items"]:
            for _ in range(n):
                trials.append({
                    "sub_study_id": "study_1_hypothetical_stories",
                    "scenario_id": item["id"],
                    "scenario": item["id"],
                    "items": [item],
                    "profile": {"age": random.randint(18, 22), "gender": random.choice(["Male", "Female"])},
                    "variant": self.PROMPT_VARIANT
                })
        return trials

    def dump_prompts(self, output_dir):
        trials = self.create_trials(n_trials=1)
        for idx, trial in enumerate(trials):
            prompt = self.prompt_builder.build_trial_prompt(trial)  # Use self.prompt_builder
            with open(f"{output_dir}/study_001_trial_{idx}.txt", "w") as f:
                f.write(prompt)
```

**Example 2: Between-Subjects Design (study_002)**
```python
class CustomPromptBuilder(PromptBuilder):
    def __init__(self, study_path: Path):
        super().__init__(study_path)

    def build_trial_prompt(self, trial_metadata):
        # Note: System prompt is now handled separately by SystemPromptRegistry
        # This method only builds the task/trial content
        items = trial_metadata.get("items", [])

        prompt = ""

        # Add task context introduction
        prompt += "You are participating in a psychology study on judgment and decision-making. Please read the following instructions and provide your responses.\n\n"

        q_counter = 1

        for item in items:
            # For anchored studies, assign anchor type at participant level
            anchor_type = item.get("assigned_anchor_type")
            anchor_val = item.get("metadata", {}).get(f"{anchor_type}_anchor")

            prompt += f"Q{q_counter}.1 (A/B): Is value higher/lower than {anchor_val}?\n"
            prompt += f"Q{q_counter}.2 (number): Your estimate?\n"
            item["q_idx_choice"] = f"Q{q_counter}.1"
            item["q_idx_estimate"] = f"Q{q_counter}.2"
            q_counter += 1

        prompt += f"\nRESPONSE_SPEC: Q1.1=<A/B>, Q1.2=<number>, Q2.1=<A/B>, Q2.2=<number>\n"
        return prompt

@StudyConfigRegistry.register("study_002")
class StudyStudy002Config(BaseStudyConfig):
    prompt_builder_class = CustomPromptBuilder
    PROMPT_VARIANT = "v3"

    def __init__(self, study_path: Path, specification: Dict[str, Any]):
        super().__init__(study_path, specification)

    def create_trials(self, n_trials=None):
        trials = []
        material = self.load_material("exp_1_anchored_estimation")
        spec = self.specification
        n = spec["participants"]["by_sub_study"]["exp_1_anchored_estimation"]["n"]
        if n == 0:
            n = 50  # Default to ensure experiment runs

        for i in range(n):
            # Assign anchor type at PARTICIPANT level (all items get same anchor)
            assigned_anchor_type = random.choice(["low", "high"])
            assigned_items = []
            for item in material["items"]:
                item_copy = item.copy()
                item_copy["assigned_anchor_type"] = assigned_anchor_type
                assigned_items.append(item_copy)

            # ONE trial per participant with ALL items
            trials.append({
                "sub_study_id": "exp_1_anchored_estimation",
                "scenario_id": "exp_1_anchored_estimation",
                "scenario": "exp_1_anchored_estimation",
                "items": assigned_items,
                "profile": {"age": random.randint(18, 25), "gender": random.choice(["male", "female"])},
                "variant": self.PROMPT_VARIANT
            })
        return trials
```

### Your Task
Generate the complete `CustomPromptBuilder` and `StudyConfig` classes following these patterns:
- Use class attributes for `prompt_builder_class` and `PROMPT_VARIANT`
- Call `super().__init__()` in both `__init__` methods
- Create ONE trial per participant with ALL items (unless within-subjects design)
- Use `Qk=<value>` or `Qk.n=<value>` format for responses. Please specify the exact format expected in the RESPONSE_SPEC.
- Include RESPONSE_SPEC with exact format expected
- Use `self.prompt_builder` (not `self.prompt_builder_class()`) in `dump_prompts`
- If `n=0`, use default `n=50` to ensure experiments run

DO NOT write import statements or the registration decorator - these will be added automatically.

Generate the code now:
"""
        return template.replace("[[STUDY_ID]]", study_id)\
                       .replace("[[EXTRACTION_SUMMARY]]", extraction_summary)\
                       .replace("[[MATERIAL_CONTEXT]]", material_context)

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response"""
        response_text = response.strip()

        if '```python' in response_text:
            response_text = response_text.split('```python')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()

        lines = response_text.split('\n')

        start_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith('\"\"\"') or
                stripped.startswith("'''") or
                stripped.startswith('import ') or
                stripped.startswith('from ') or
                stripped.startswith('class ') or
                stripped.startswith('def ') or
                stripped.startswith('@')):
                start_idx = i
                break

        end_idx = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            stripped = lines[i].strip()
            if stripped and not stripped.startswith('#'):
                end_idx = i + 1
                break

        code = '\n'.join(lines[start_idx:end_idx])

        if not code:
            print(f"Warning: Extracted code is empty! Raw response length: {len(response)}")
            if response.strip():
                code = response.strip()

        return code
