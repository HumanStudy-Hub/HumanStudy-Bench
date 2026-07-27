import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Sequence, Tuple
from pathlib import Path

from src.agents.prompt_builder import PromptBuilder


class BaseStudyConfig(ABC):
    prompt_builder_class = PromptBuilder
    SUPPORTED_SUB_STUDIES: Optional[Tuple[str, ...]] = None

    def __init__(self, study_path: Path, specification: Dict[str, Any]):
        study_root = Path(study_path)
        source_path = study_root / "source" if (study_root / "source").is_dir() else study_root
        self.study_root = study_root
        self.study_path = source_path
        self.source_path = source_path
        self.specification = specification
        self.study_id = specification["study_id"]
        self.selected_sub_studies: Tuple[str, ...] = ()

        self.prompt_builder = self.prompt_builder_class(self.study_path)

    def get_supported_sub_studies(self) -> Optional[Tuple[str, ...]]:
        return self.SUPPORTED_SUB_STUDIES

    def configure_sub_studies(
        self,
        sub_studies: Optional[Sequence[str]],
    ) -> Tuple[str, ...]:
        if not sub_studies:
            self.selected_sub_studies = ()
            return self.selected_sub_studies

        requested = [sub_studies] if isinstance(sub_studies, str) else list(sub_studies)
        normalized: List[str] = []
        for value in requested:
            sub_study_id = str(value).strip()
            if not sub_study_id:
                raise ValueError("sub-study identifiers cannot be empty")
            if sub_study_id not in normalized:
                normalized.append(sub_study_id)

        supported = self.get_supported_sub_studies()
        if supported is None:
            raise ValueError(
                f"{self.study_id} does not declare sub-study selection support"
            )
        invalid = [value for value in normalized if value not in supported]
        if invalid:
            raise ValueError(
                f"Unsupported sub-study selection for {self.study_id}: {invalid}. "
                f"Available: {list(supported)}"
            )

        requested_set = set(normalized)
        self.selected_sub_studies = tuple(
            value for value in supported if value in requested_set
        )
        return self.selected_sub_studies

    def load_material(self, sub_study_id: str) -> Dict[str, Any]:
        file_path = self.study_path / "materials" / f"{sub_study_id}.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Material not found: {file_path}")
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                return json.load(f)
        except UnicodeDecodeError as e:
            # Try to detect encoding and provide helpful error
            import chardet
            with open(file_path, "rb") as f:
                raw = f.read()
                detected = chardet.detect(raw)
            raise UnicodeDecodeError(
                'utf-8', raw, e.start, e.end,
                f"File encoding issue. Detected: {detected.get('encoding', 'unknown')} "
                f"(confidence: {detected.get('confidence', 0):.2f}). "
                f"Please ensure the file is UTF-8 encoded."
            )

    def load_metadata(self) -> Dict[str, Any]:
        """load metadata.json"""
        file_path = self.study_path / "metadata.json"
        with open(file_path, "r", encoding='utf-8') as f:
            return json.load(f)

    def load_specification(self) -> Dict[str, Any]:
        """load specification.json"""
        file_path = self.study_path / "specification.json"
        with open(file_path, "r", encoding='utf-8') as f:
            return json.load(f)

    def load_ground_truth(self) -> Dict[str, Any]:
        """load ground_truth.json"""
        file_path = self.study_path / "ground_truth.json"
        with open(file_path, "r", encoding='utf-8') as f:
            return json.load(f)

    def extract_numeric(self, text: str, default: float = 0.0) -> float:
        if text is None: return default
        import re
        match = re.search(r"(-?\d+\.?\d*)", str(text))
        return float(match.group(1)) if match else default

    def extract_choice(self, text: str, options: List[str] = None) -> Optional[int]:
        if text is None: return None
        import re
        text_s = str(text).strip()

        if options:
            for i, opt in enumerate(options):
                if opt.lower() in text_s.lower():
                    return i

        match = re.search(r"\b([A-Z])\b", text_s.upper())
        if match:
            # A->0, B->1...
            return ord(match.group(1)) - ord('A')

        return None

    @abstractmethod
    def create_trials(self, n_trials: Optional[int] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_prompt_builder(self) -> PromptBuilder:
        """Get prompt builder"""
        return self.prompt_builder

    def get_instructions(self) -> str:
        return self.prompt_builder.get_instructions()

    def aggregate_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        return raw_results

    def custom_scoring(
        self,
        results: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        return None

    def get_n_participants(self) -> int:
        return self.specification["participants"]["n"]

    def get_study_type(self) -> str:
        return self.specification.get("study_type", self.study_id)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(study_id='{self.study_id}')"


class GeneratedStudyPromptBuilder(PromptBuilder):
    """Prompt builder for AI-generated HumanStudy-Bench draft packages."""

    def build_trial_prompt(self, trial_data: Dict[str, Any]) -> str:
        material = trial_data.get("material") or trial_data
        sections: List[str] = []

        study_name = material.get("study_name") or material.get("sub_study_id") or trial_data.get("sub_study_id")
        if study_name:
            sections.append(f"STUDY:\n{study_name}")

        instructions = (material.get("instructions") or "").strip()
        if instructions and instructions != material.get("stimulus"):
            sections.append(f"INSTRUCTIONS:\n{instructions}")

        stimulus = (
            material.get("stimulus")
            or material.get("scenario")
            or material.get("vignette")
            or ""
        ).strip()
        if stimulus:
            sections.append(f"MATERIAL:\n{stimulus}")

        condition_text = self._format_condition_assignment(trial_data)
        if condition_text:
            sections.append(f"ASSIGNED CONDITION:\n{condition_text}")

        questions = self._format_questions(material)
        if questions:
            sections.append(f"QUESTIONS:\n{questions}")

        response_spec = self._format_response_spec(material)
        sections.append(f"RESPONSE_SPEC:\n{response_spec}")

        return "\n\n".join(sections)

    def _format_condition_assignment(self, trial_data: Dict[str, Any]) -> str:
        assignment = trial_data.get("condition_assignment") or {}
        if not assignment:
            return ""
        lines = []
        for name, payload in assignment.items():
            if isinstance(payload, dict):
                level = payload.get("level")
                description = payload.get("description")
                if description:
                    lines.append(f"- {name}: {level}. {description}")
                else:
                    lines.append(f"- {name}: {level}")
            else:
                lines.append(f"- {name}: {payload}")
        return "\n".join(lines)

    def _format_questions(self, material: Dict[str, Any]) -> str:
        items = material.get("items")
        if isinstance(items, list) and items:
            lines = []
            for idx, item in enumerate(items, start=1):
                if not isinstance(item, dict):
                    continue
                question = item.get("question") or item.get("text") or item.get("label")
                if not question:
                    continue
                lines.append(f"Q{idx}. {question}")
                options = item.get("options") or item.get("choices")
                if isinstance(options, list) and options:
                    lines.append("Options:")
                    for opt in options:
                        lines.append(f"- {opt}")
                scale = item.get("scale") or item.get("response_format") or {}
                anchors = scale.get("anchors") if isinstance(scale, dict) else None
                if isinstance(anchors, dict) and anchors:
                    lines.append("Scale anchors:")
                    for key, value in anchors.items():
                        lines.append(f"- {key}: {value}")
            return "\n".join(lines).strip()

        question = material.get("question")
        if question:
            return f"Q1. {question}"
        return ""

    def _format_response_spec(self, material: Dict[str, Any]) -> str:
        response_format = material.get("response_format") or {}
        if not isinstance(response_format, dict):
            response_format = {}

        answer_type = response_format.get("answer_type") or response_format.get("type") or "free_text"
        lines = [
            "Output only answer lines.",
            "Use Qk=<value>, one line per question.",
            f"Answer type: {answer_type}.",
        ]

        scale_min = response_format.get("scale_min")
        scale_max = response_format.get("scale_max")
        if scale_min is not None and scale_max is not None:
            lines.append(f"Allowed numeric range: {scale_min} to {scale_max}.")

        options = response_format.get("options")
        if isinstance(options, list) and options:
            lines.append("Allowed options:")
            for opt in options:
                lines.append(f"- {opt}")

        anchors = response_format.get("anchors")
        if isinstance(anchors, dict) and anchors:
            lines.append("Anchors:")
            for key, value in anchors.items():
                lines.append(f"- {key}: {value}")

        return "\n".join(lines)


class GenericGeneratedStudyConfig(BaseStudyConfig):
    """Fallback config for Stage4-generated study packages without custom adapters."""

    prompt_builder_class = GeneratedStudyPromptBuilder
    DEFAULT_TRIAL_COUNT = 30

    def get_supported_sub_studies(self) -> Optional[Tuple[str, ...]]:
        sub_study_ids: List[str] = []
        for material in self._load_ready_materials(apply_scope=False):
            sub_study_id = material.get("sub_study_id")
            if sub_study_id and sub_study_id not in sub_study_ids:
                sub_study_ids.append(str(sub_study_id))
        return tuple(sub_study_ids)

    def create_trials(self, n_trials: Optional[int] = None) -> List[Dict[str, Any]]:
        materials = self._load_ready_materials()
        if not materials:
            raise ValueError(f"No ready material JSON files found in {self.study_path / 'materials'}")

        requested_total = self._resolve_requested_trial_count(materials, n_trials)
        trials: List[Dict[str, Any]] = []
        trial_number = 1

        for material, count in self._allocate_trials(materials, requested_total, n_trials):
            for local_idx in range(count):
                trials.append(
                    {
                        "trial_number": trial_number,
                        "study_type": "generated_human_study",
                        "trial_type": "generated_material",
                        "sub_study_id": material.get("sub_study_id"),
                        "material_id": material.get("material_id") or material.get("target_id"),
                        "target_id": material.get("target_id"),
                        "scenario_id": material.get("material_id") or material.get("target_id"),
                        "scenario": material.get("stimulus") or material.get("scenario"),
                        "stimulus": material.get("stimulus"),
                        "instructions": material.get("instructions"),
                        "question": material.get("question"),
                        "items": material.get("items", []),
                        "response_format": material.get("response_format", {}),
                        "condition_assignment": self._condition_assignment(material, local_idx),
                        "metadata": material.get("metadata", {}),
                        "material": material,
                    }
                )
                trial_number += 1

        return trials

    def get_instructions(self) -> str:
        instructions = super().get_instructions()
        if instructions != "No instructions provided.":
            return instructions
        return (
            "You are a participant in a behavioral research study. "
            "Read the material carefully and answer the study questions in the requested format."
        )

    def aggregate_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        individual_data = raw_results.get("individual_data", []) if isinstance(raw_results, dict) else []
        by_material: Dict[str, int] = {}
        for row in individual_data:
            trial_info = row.get("trial_info", {}) if isinstance(row, dict) else {}
            material_id = trial_info.get("material_id") or trial_info.get("target_id") or "unknown"
            by_material[material_id] = by_material.get(material_id, 0) + 1

        result = dict(raw_results) if isinstance(raw_results, dict) else {"individual_data": individual_data}
        result["descriptive_statistics"] = {
            "n_responses": len(individual_data),
            "responses_by_material": by_material,
        }
        result.setdefault("inferential_statistics", {})
        return result

    def _load_ready_materials(
        self,
        *,
        apply_scope: bool = True,
    ) -> List[Dict[str, Any]]:
        materials = []
        for path in sorted((self.study_path / "materials").glob("*.json")):
            material = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(material, dict):
                continue
            readiness = material.get("readiness")
            if isinstance(readiness, dict) and readiness.get("ready") is False:
                continue
            material.setdefault("material_id", path.stem)
            if (
                apply_scope
                and self.selected_sub_studies
                and material.get("sub_study_id") not in self.selected_sub_studies
            ):
                continue
            materials.append(material)
        return materials

    def _resolve_requested_trial_count(self, materials: List[Dict[str, Any]], n_trials: Optional[int]) -> int:
        if n_trials is not None:
            return max(int(n_trials), 1)

        by_sub_study = self.specification.get("participants", {}).get("by_sub_study")
        if isinstance(by_sub_study, dict) and by_sub_study:
            total = 0
            for material in materials:
                sub_study_id = material.get("sub_study_id")
                sub_spec = by_sub_study.get(sub_study_id, {}) if sub_study_id else {}
                total += int(sub_spec.get("n") or 1)
            return max(total, len(materials))

        participants = self.specification.get("participants", {})
        return max(int(participants.get("n") or self.DEFAULT_TRIAL_COUNT), 1)

    def _allocate_trials(
        self,
        materials: List[Dict[str, Any]],
        total: int,
        n_trials: Optional[int],
    ) -> List[Tuple[Dict[str, Any], int]]:
        allocations: Dict[int, int] = {idx: 0 for idx in range(len(materials))}

        by_sub_study = self.specification.get("participants", {}).get("by_sub_study")
        if n_trials is None and isinstance(by_sub_study, dict) and by_sub_study:
            for idx, material in enumerate(materials):
                sub_study_id = material.get("sub_study_id")
                sub_spec = by_sub_study.get(sub_study_id, {}) if sub_study_id else {}
                allocations[idx] = max(int(sub_spec.get("n") or 1), 1)
            return [(materials[idx], count) for idx, count in allocations.items() if count > 0]

        for idx in range(total):
            allocations[idx % len(materials)] += 1
        return [(materials[idx], count) for idx, count in allocations.items() if count > 0]

    def _condition_assignment(self, material: Dict[str, Any], local_idx: int) -> Dict[str, Any]:
        assignment: Dict[str, Any] = {}
        conditions = material.get("conditions")
        if not isinstance(conditions, list):
            return assignment

        for condition in conditions:
            if not isinstance(condition, dict):
                continue
            name = condition.get("name") or condition.get("label") or "condition"
            levels = condition.get("levels")
            if not isinstance(levels, list) or not levels:
                continue
            level = levels[local_idx % len(levels)]
            descriptions = condition.get("level_descriptions") or {}
            description = descriptions.get(level) if isinstance(descriptions, dict) else None
            assignment[name] = {"level": level, "description": description}
        return assignment


class StudyConfigRegistry:
    _configs: Dict[str, type] = {}

    @classmethod
    def register(cls, study_id: str):
        def decorator(config_class):
            cls._configs[study_id] = config_class
            return config_class
        return decorator

    @classmethod
    def get_config_class(cls, study_id: str) -> Optional[type]:
        """Get the config class"""
        return cls._configs.get(study_id)

    @classmethod
    def create_config(
        cls,
        study_id: str,
        study_path: Path,
        specification: Dict[str, Any]
    ) -> Optional[BaseStudyConfig]:
        config_class = cls.get_config_class(study_id)
        if config_class:
            return config_class(study_path, specification)
        return None

    @classmethod
    def list_registered_studies(cls) -> List[str]:
        return list(cls._configs.keys())


def _looks_like_generated_stage4_package(study_path: Path) -> bool:
    package_path = study_path / "source" if (study_path / "source").is_dir() else study_path
    audit_path = package_path / "audit.json"
    if audit_path.exists():
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if str(audit.get("stage4_version", "")).startswith("human-study-bench"):
                return True
        except Exception:
            pass
    materials_path = package_path / "materials"
    return materials_path.is_dir() and any(materials_path.glob("*.json"))


def _load_hub_script_config(
    study_id: str,
    study_path: Path,
    specification: Dict[str, Any],
) -> Optional[Any]:
    """Load `studies/<study_id>/scripts/config.py` when a Hub package provides one."""
    import importlib.util
    import sys

    config_path = Path(study_path) / "scripts" / "config.py"
    if not config_path.exists():
        return None

    scripts_dir = str(config_path.parent)
    inserted = False
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
        inserted = True

    module_name = f"{study_id}_hub_config"
    try:
        spec = importlib.util.spec_from_file_location(module_name, config_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load config module spec: {config_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if inserted:
            try:
                sys.path.remove(scripts_dir)
            except ValueError:
                pass

    candidates = []
    for obj in vars(module).values():
        if not isinstance(obj, type) or obj.__name__ in {"BaseStudyConfig", "PromptBuilder"}:
            continue
        if hasattr(obj, "create_trials") and obj.__name__.lower().endswith("config"):
            candidates.append(obj)

    if not candidates:
        raise ValueError(f"No config class with create_trials found in {config_path}")

    config_class = candidates[0]
    return config_class(Path(study_path), specification)


def get_study_config(
    study_id: str,
    study_path: Path,
    specification: Dict[str, Any]
) -> BaseStudyConfig:
    config = _load_hub_script_config(study_id, Path(study_path), specification)
    if config is not None:
        return config

    try:
        import importlib
        import pkgutil
        import src.studies

        for _, name, _ in pkgutil.iter_modules(src.studies.__path__, src.studies.__name__ + "."):
            try:
                importlib.import_module(name)
            except Exception as e:
                print(f"Warning: Could not import study config {name}: {e}")
    except ModuleNotFoundError:
        pass

    config = StudyConfigRegistry.create_config(study_id, study_path, specification)

    if config is None:
        if _looks_like_generated_stage4_package(Path(study_path)):
            return GenericGeneratedStudyConfig(study_path, specification)
        raise ValueError(
            f"No configuration found for {study_id}. "
            f"Available: {StudyConfigRegistry.list_registered_studies()}"
        )

    return config
