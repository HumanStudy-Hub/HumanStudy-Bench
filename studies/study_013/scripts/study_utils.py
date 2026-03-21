"""
Standalone study utilities: BaseStudyConfig and PromptBuilder.
No dependency on src/; for use within each study's scripts/.
"""
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional


RESPONSE_LINE_PATTERN = re.compile(r"Q(\d+)\s*[:=]\s*(.+)")
EXPLICIT_SKIP_RESPONSES = {
    "",
    "skip",
    "n/a",
    "na",
}
NO_ADDITIONAL_INFO_RESPONSES = {
    "none",
    "none needed",
    "nothing else",
    "nothing additional",
    "no additional info",
    "no additional information",
    "no additional information needed",
    "no more information",
    "no more information needed",
}
NO_ADDITIONAL_INFO_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"\bno (?:additional|further|more) info(?:rmation)? (?:is )?needed\b",
        r"\bno (?:additional|further|more) (?:data|research) (?:is )?needed\b",
        r"\b(?:the|this) information given is enough\b",
        r"\bthe case provides enough information\b",
        r"\benough information (?:is )?(?:provided|given)\b",
    )
]
STATISTICAL_REASONING_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"\bmarket research\b",
        r"\bmarket study\b",
        r"\bsurvey\b",
        r"\blarger sample\b",
        r"\bsample size\b",
        r"\brepresentative sample\b",
        r"\bstatistical data\b",
        r"\bindustry statistics?\b",
        r"\bmarket (?:size|demand|growth) data\b",
        r"\bcustomer (?:data|survey|research)\b",
        r"\bdemand (?:data|research)\b",
        r"\btrend data\b",
        r"\bmore (?:data|research|surveys?|samples?)\b",
    )
]


def parse_question_responses(response_text: str) -> Dict[int, str]:
    """Parse Qk=value or Qk: value lines into a question-number map."""
    responses: Dict[int, str] = {}
    for line in str(response_text or "").splitlines():
        match = RESPONSE_LINE_PATTERN.match(line.strip())
        if match:
            responses[int(match.group(1))] = match.group(2).strip()
    return responses


def extract_numeric_value(text: Any, default: Optional[float] = None) -> Optional[float]:
    """Extract the first numeric value from text, tolerating commas and units."""
    if text is None:
        return default
    clean_text = str(text).replace(",", "")
    match = re.search(r"(-?\d+(?:\.\d+)?)", clean_text)
    return float(match.group(1)) if match else default


def code_belief_in_small_numbers(text: Any) -> Optional[int]:
    """
    Code the open-ended vignette response.

    -1: respondent asks for broader evidence such as market research / survey / data
    +1: respondent relies on the vignette and anecdotal cues without that request,
        including statements that no additional information is needed
    None: blank / skipped response
    """
    if text is None:
        return None

    normalized = re.sub(r"\s+", " ", str(text).strip().lower())
    canonical = normalized.strip(" .,!?:;")
    if canonical in EXPLICIT_SKIP_RESPONSES:
        return None
    if canonical in NO_ADDITIONAL_INFO_RESPONSES:
        return 1

    for pattern in STATISTICAL_REASONING_PATTERNS:
        if pattern.search(normalized):
            return -1

    for pattern in NO_ADDITIONAL_INFO_PATTERNS:
        if pattern.search(normalized):
            return 1

    return 1


def iter_response_records(results: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Yield flat response records regardless of whether results are already flattened
    or grouped under participant summaries.
    """
    individual_data = results.get("individual_data") or []
    if individual_data:
        first = individual_data[0]
        if isinstance(first, dict) and "response_text" in first:
            yield from individual_data
            return
        if isinstance(first, dict) and "responses" in first:
            for participant in individual_data:
                for response in participant.get("responses", []):
                    yield response
            return

    for participant in results.get("participant_summaries", []) or []:
        for response in participant.get("responses", []):
            yield response


def compute_construct_scores(response_text: str, trial_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Compute participant-level construct scores from one completed questionnaire."""
    responses = parse_question_responses(response_text)
    items_a = trial_info.get("items_a", [])
    items_b = trial_info.get("items_b", [])
    items_c = trial_info.get("items_c", [])
    items_d = trial_info.get("items_d", [])

    risk_propensity = 0
    risk_answered = 0
    for item in items_a:
        q_idx = item.get("q_idx")
        if q_idx and q_idx in responses:
            choice_text = responses[q_idx].strip().lower()
            choice_match = re.search(r"\b([ab])\b", choice_text)
            if choice_match:
                choice = choice_match.group(1)
                risk_answered += 1
                if choice == item.get("metadata", {}).get("risky_option", "a"):
                    risk_propensity += 1

    planning_fallacy = 0
    planning_count = 0
    illusion_of_control = 0
    ioc_count = 0
    for item in items_b:
        q_idx = item.get("q_idx")
        if not q_idx or q_idx not in responses:
            continue
        value = extract_numeric_value(responses[q_idx], default=None)
        if value is None or not 1 <= value <= 7:
            continue
        construct = item.get("metadata", {}).get("construct")
        if construct == "planning_fallacy":
            planning_fallacy += value
            planning_count += 1
        elif construct == "illusion_of_control":
            illusion_of_control += value
            ioc_count += 1

    overconfidence = 0
    oc_count = 0
    for item in items_c:
        q_lower = item.get("q_idx_lower")
        q_upper = item.get("q_idx_upper")
        correct_answer = item.get("correct_answer")
        if not q_lower or not q_upper or correct_answer is None:
            continue
        if q_lower not in responses or q_upper not in responses:
            continue
        lower = extract_numeric_value(responses[q_lower], default=None)
        upper = extract_numeric_value(responses[q_upper], default=None)
        if lower is None or upper is None:
            continue
        if lower > upper:
            lower, upper = upper, lower
        oc_count += 1
        if correct_answer < lower or correct_answer > upper:
            overconfidence += 1

    risk_perception = 0
    rp_count = 0
    opportunity_evaluation = 0
    oe_count = 0
    small_numbers = None
    for item in items_d:
        q_idx = item.get("q_idx")
        if not q_idx or q_idx not in responses:
            continue

        construct = item.get("metadata", {}).get("construct")
        if construct == "belief_in_small_numbers":
            small_numbers = code_belief_in_small_numbers(responses[q_idx])
            continue

        value = extract_numeric_value(responses[q_idx], default=None)
        if value is None or not 1 <= value <= 7:
            continue
        if construct == "risk_perception":
            risk_perception += value
            rp_count += 1
        elif construct == "opportunity_evaluation":
            opportunity_evaluation += value
            oe_count += 1

    profile = trial_info.get("profile", {}) or {}
    age = extract_numeric_value(profile.get("age"), default=None)

    if (
        risk_answered < 5
        or planning_count < 2
        or ioc_count < 3
        or oc_count < 10
        or rp_count < 4
        or oe_count < 3
        or age is None
    ):
        return None

    return {
        "risk_propensity": risk_propensity,
        "planning_fallacy": planning_fallacy,
        "illusion_of_control": illusion_of_control,
        "overconfidence": overconfidence,
        "risk_perception": risk_perception,
        "opportunity_evaluation": opportunity_evaluation,
        "small_numbers": small_numbers,
        "age": age,
        "profile": profile,
    }


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
