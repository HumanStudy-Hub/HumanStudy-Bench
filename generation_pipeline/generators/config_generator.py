""" Config Generator - Generates StudyConfig classes from extraction results using LLM """

import ast
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

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
        material_context: List[Dict[str, Any]] = []
        material_ids: Set[str] = set()
        if study_dir:
            for json_file in ["metadata.json", "specification.json", "ground_truth.json"]:
                p = study_dir / json_file
                if p.exists():
                    try:
                        study_context[json_file] = json.loads(p.read_text(encoding='utf-8'))
                    except (OSError, json.JSONDecodeError):
                        pass

            materials_dir = study_dir / "materials"
            if materials_dir.exists():
                for json_file in sorted(materials_dir.glob("*.json")):
                    try:
                        payload = json.loads(json_file.read_text(encoding='utf-8'))
                    except (OSError, json.JSONDecodeError):
                        continue
                    material_id = str(
                        payload.get("sub_study_id") or json_file.stem
                    ).strip()
                    if material_id:
                        material_ids.add(material_id)
                    material_context.append(
                        self._compact_material_context(payload, material_id or json_file.stem)
                    )

        prompt = self._build_logic_only_prompt(
            json.dumps(
                self._compact_extraction_context(extraction_result),
                indent=2,
                ensure_ascii=False,
            ),
            study_id,
            json.dumps(study_context, indent=2, ensure_ascii=False),
            json.dumps(material_context, indent=2, ensure_ascii=False),
        )

        try:
            response = self._generate_with_retry(prompt)
        except Exception as e:
            raise RuntimeError(f"LLM Error: {e}")

        logic_code = self._extract_code_from_response(response)
        final_code = self._assemble_final_code(logic_code, study_id, standalone)
        validation_error = self._validate_generated_code(
            final_code,
            material_ids=material_ids,
        )
        max_repair_attempts = 3
        for _repair_attempt in range(max_repair_attempts):
            if not validation_error:
                break
            repair_prompt = self._build_repair_prompt(
                study_id=study_id,
                logic_code=logic_code,
                validation_error=validation_error,
            )
            try:
                repair_response = self._generate_with_retry(repair_prompt)
            except Exception as e:
                raise RuntimeError(f"LLM config repair error: {e}") from e

            logic_code = self._extract_code_from_response(repair_response)
            final_code = self._assemble_final_code(logic_code, study_id, standalone)
            validation_error = self._validate_generated_code(
                final_code,
                material_ids=material_ids,
            )
        if validation_error:
            raise ValueError(
                "Generated StudyConfig remained invalid after "
                f"{max_repair_attempts} repair attempts: {validation_error}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_code, encoding='utf-8')
        return output_path

    def _generate_with_retry(
        self,
        prompt: str,
        *,
        attempts: int = 3,
        timeout: float = 300.0,
        max_tokens: int = 16000,
    ) -> str:
        last_error: Optional[BaseException] = None
        for attempt in range(max(1, attempts)):
            try:
                return self.client.generate_content(
                    prompt=prompt,
                    timeout=timeout,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                last_error = exc
                if attempt + 1 < max(1, attempts):
                    time.sleep(2 ** attempt)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _compact_extraction_context(extraction: Dict[str, Any]) -> Dict[str, Any]:
        studies: List[Dict[str, Any]] = []
        for study in extraction.get("studies", []) or []:
            if not isinstance(study, dict):
                continue
            effects = []
            for effect in study.get("effects", []) or []:
                if not isinstance(effect, dict):
                    continue
                effects.append(
                    {
                        key: effect.get(key)
                        for key in (
                            "effect_id",
                            "effecttype",
                            "IV",
                            "DV",
                            "direction",
                            "analysis_scope",
                        )
                        if effect.get(key) not in (None, "", [], {})
                    }
                )
            studies.append(
                {
                    "study_id": study.get("study_id") or study.get("sub_study_id"),
                    "study_name": study.get("study_name") or study.get("study"),
                    "phenomenon": study.get("phenomenon"),
                    "sample": study.get("sample"),
                    "effects": effects,
                }
            )
        return {
            "study_id": extraction.get("study_id"),
            "paper_id": extraction.get("paper_id"),
            "paper_title": extraction.get("paper_title"),
            "paper_authors": extraction.get("paper_authors"),
            "paper_year": extraction.get("paper_year"),
            "source_schema": extraction.get("source_schema"),
            "studies": studies,
        }

    @staticmethod
    def _compact_material_context(
        material: Dict[str, Any],
        material_id: str,
    ) -> Dict[str, Any]:
        items = [
            item for item in material.get("items", []) or [] if isinstance(item, dict)
        ]
        unit_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in items:
            condition = item.get("condition") if isinstance(item.get("condition"), dict) else {}
            unit_id = str(
                condition.get("runtime_unit_id")
                or item.get("runtime_unit_id")
                or item.get("block")
                or "default"
            )
            unit_items[unit_id].append(item)

        runtime_units: List[Dict[str, Any]] = []
        for unit_id, grouped in sorted(unit_items.items()):
            scopes = sorted(
                {
                    str((item.get("condition") or {}).get("assignment_scope") or "")
                    for item in grouped
                    if isinstance(item.get("condition"), dict)
                    and str((item.get("condition") or {}).get("assignment_scope") or "")
                }
            )
            runtime_units.append(
                {
                    "runtime_unit_id": unit_id,
                    "item_count": len(grouped),
                    "assignment_scopes": scopes,
                    "timepoints": sorted(
                        {str(item.get("timepoint")) for item in grouped if item.get("timepoint")}
                    ),
                    "trial_groups": sorted(
                        {str(item.get("trial_group")) for item in grouped if item.get("trial_group")}
                    ),
                    "item_ids": [str(item.get("id") or "") for item in grouped],
                    "sample_item": ConfigGenerator._compact_item(grouped[0]),
                }
            )

        return {
            "material_id": material_id,
            "study_name": material.get("study_name"),
            "readiness": material.get("readiness"),
            "preserve_full_instrument_for_runtime": bool(
                material.get("preserve_full_instrument_for_runtime")
            ),
            "instructions": str(material.get("instructions") or "")[:1200],
            "conditions": material.get("conditions") or [],
            "response_schema": material.get("response_schema") or {},
            "item_count": len(items),
            "item_types": dict(Counter(str(item.get("type") or "unknown") for item in items)),
            "item_schema_keys": sorted({key for item in items for key in item}),
            "runtime_unit_summary_prompt_only": runtime_units,
        }

    @staticmethod
    def _compact_item(item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": item.get("id"),
            "question": str(item.get("question") or "")[:800],
            "type": item.get("type"),
            "options": list(item.get("options") or [])[:5],
            "scale": item.get("scale") or {},
            "matrix": item.get("matrix") or {},
            "condition": item.get("condition") or {},
            "block": item.get("block"),
            "timepoint": item.get("timepoint"),
            "trial_group": item.get("trial_group"),
        }

    def _assemble_final_code(self, logic_code: str, study_id: str, standalone: bool) -> str:
        class_name = f"Study{study_id.replace('_', '').capitalize()}Config"
        if standalone:
            imports = """import copy
import json
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
            imports = """import copy
import json
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
    def _validate_generated_code(
        code: str,
        *,
        material_ids: Optional[Set[str]] = None,
    ) -> Optional[str]:
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
        prompt_builders = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and any(isinstance(base, ast.Name) and base.id == "PromptBuilder" for base in node.bases)
        ]
        if not prompt_builders:
            return "missing a class that subclasses PromptBuilder"
        if not any(
            isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
            and member.name == "build_trial_prompt"
            for builder in prompt_builders
            for member in builder.body
        ):
            return "PromptBuilder subclass is missing build_trial_prompt()"
        if "RESPONSE_SPEC" not in code:
            return "build_trial_prompt() must emit an explicit RESPONSE_SPEC"
        if re.search(r"\.get\(\s*['\"](?:runtime_units|sample_item)['\"]", code):
            return (
                "runtime_units/sample_item are prompt-only summaries; create_trials() "
                "must group the real material['items'] by item.condition.runtime_unit_id"
            )

        loaded_materials: Set[str] = set()
        dynamic_material_load = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "load_material" or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                loaded_materials.add(first.value)
            else:
                dynamic_material_load = True
        expected = set(material_ids or set())
        unknown = loaded_materials - expected if expected else set()
        if unknown:
            return "load_material() references unknown material ids: " + ", ".join(sorted(unknown))
        missing = expected - loaded_materials
        if missing and not dynamic_material_load:
            return "create_trials() skips material ids: " + ", ".join(sorted(missing))
        if expected and not loaded_materials and not dynamic_material_load:
            return "create_trials() never calls load_material()"
        return None

    @staticmethod
    def _build_repair_prompt(study_id: str, logic_code: str, validation_error: str) -> str:
        return f"""You generated invalid core Python logic for a HumanStudyBench StudyConfig.

STUDY ID: {study_id}
VALIDATION ERROR: {validation_error}

Repair the code while preserving its study design, canonical material IDs, trial structure,
and response format. Escape line breaks inside Python string literals as \\n. The corrected
code must define a class that subclasses BaseStudyConfig and implements create_trials().
Treat n_trials as a total package-level trial budget: when it is at least the number of
canonical materials, the returned trials must cover every material. Emit exactly one
line-starting `RESPONSE_SPEC:` header in every rendered prompt; explanatory prose may
refer to the response format without repeating that header.
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
4. **Respect runtime units** - Every real item carries `condition.runtime_unit_id` and `assignment_scope`; these are canonical routing metadata. Never show mutually exclusive group/condition units to the same participant. For repeated timepoints belonging to the same assigned group, preserve their source order and coupling.
5. **Do not crop full instruments** - When `preserve_full_instrument_for_runtime=true`, retain every item assigned to the selected runtime unit(s). Do not select items by keyword similarity.
6. **Keep runtime content source-only** - Render `instructions`, item questions, options, scales, and matrices from material JSON. Never expose researcher metadata, option-role labels, findings, means, or statistics.
7. **Use a package-level trial budget** - `n_trials` is the total number of returned trials, not a per-study count. Distribute that budget across canonical materials; when `n_trials >= number_of_materials`, return at least one trial for every material. Never append all trials for the first material and then truncate the combined list.
8. **Emit one response header** - Every rendered trial prompt must contain exactly one line-starting `RESPONSE_SPEC:` header. Do not repeat that exact header in prose, examples, mappings, or closing reminders.

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

### PACKAGE JSON CONTEXT
[[CONTEXT_SUMMARY]]

### MATERIALS (Compact executable contract)
[[MATERIAL_CONTEXT]]

IMPORTANT MATERIAL SCHEMA BOUNDARY:
- `runtime_unit_summary_prompt_only` exists only in this compact prompt to show item counts and routing. It is NOT a field in the JSON returned by `self.load_material()`.
- The real material has top-level `instructions`, `items`, `conditions`, and `response_schema`.
- At runtime, iterate over every object in `material["items"]` and group/filter it by `item.get("condition", {}).get("runtime_unit_id")` and `assignment_scope`.
- `sample_item` is descriptive prompt context only. Never load or execute it. Never reduce a runtime unit to its sample; preserve all item IDs listed for that unit.

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
- Treat `n_trials` as a total package-level budget and cover every canonical material when the budget permits
- Emit exactly one line-starting `RESPONSE_SPEC:` header per rendered prompt
- Use `self.prompt_builder` (not `self.prompt_builder_class()`) in `dump_prompts`
- If `n=0`, use default `n=50` to ensure experiments run
- Load every canonical `material_id` listed above. Do not invent aliases.
- Route the actual `material["items"]` using each item's `condition.runtime_unit_id`; use `assignment_scope` to keep one participant in one between-subject assignment while preserving its repeated timepoints.
- Build response labels from each item's actual response contract: options for multiple choice/ranking, anchors for scales/sliders, rows and columns for matrices, and free text otherwise.
- Do not mutate loaded material items while building prompts or trials; copy dictionaries before attaching runtime-only fields.

DO NOT write import statements or the registration decorator - these will be added automatically.

Generate the code now:
"""
        return template.replace("[[STUDY_ID]]", study_id)\
                       .replace("[[EXTRACTION_SUMMARY]]", extraction_summary)\
                       .replace("[[CONTEXT_SUMMARY]]", context_summary)\
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
