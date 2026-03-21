import numpy as np

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from study_utils import BaseStudyConfig, PromptBuilder, compute_construct_scores, iter_response_records

import random


AGE_DISTRIBUTION = [
    (range(30, 40), 0.222),   # Less than 40
    (range(40, 61), 0.715),   # 40 to 60
    (range(61, 70), 0.063),   # More than 60
]

SEX_OPTIONS = ["male", "female"]
SEX_WEIGHTS = [0.97, 0.03]

RACE_OPTIONS = ["Chinese", "Indian", "Other"]
RACE_WEIGHTS = [0.924, 0.045, 0.031]

EDUCATION_OPTIONS = ["secondary", "postsecondary", "primary/other"]
EDUCATION_WEIGHTS = [0.061, 0.864, 0.075]

BUSINESS_SIZE_OPTIONS = [
    "Less than S$1m",
    "Between S$1m and S$25m",
    "Between S$25m and S$50m",
    "More than S$50m",
]

BUSINESS_SIZE_WEIGHTS = [0.028, 0.486, 0.444, 0.042]


def weighted_age_sample():
    """Sample an age from the Table 2 age distribution."""
    r = random.random()
    cumulative = 0
    for age_range, prob in AGE_DISTRIBUTION:
        cumulative += prob
        if r < cumulative:
            return random.choice(list(age_range))
    return random.randint(40, 60)


def weighted_choice(options, weights):
    """Draw one option according to the reported sample proportions."""
    return random.choices(options, weights=weights, k=1)[0]


class CustomPromptBuilder(PromptBuilder):
    """Builds the full Keh, Foo & Lim (2002) questionnaire prompt."""

    def build_trial_prompt(self, trial_metadata):
        profile = trial_metadata.get("profile") or trial_metadata.get("participant_profile", {})
        items_a = trial_metadata.get("items_a", [])
        items_b = trial_metadata.get("items_b", [])
        items_c = trial_metadata.get("items_c", [])
        items_d = trial_metadata.get("items_d", [])
        vignette_text = trial_metadata.get("vignette_text", "")

        lines = []
        optional_question_numbers = []

        # --- Persona Introduction ---
        age = profile.get("age", 47)
        sex = profile.get("sex", "male")
        race = profile.get("race", "Chinese")
        education = profile.get("education", "postsecondary")
        business_size = profile.get("business_size", "Between S$1m and S$25m")
        founder = profile.get("is_founder", True)

        lines.append("You are participating in a research study on entrepreneurial decision-making.")
        lines.append(
            "Answer as one of the Singapore SME founders/owners described in the original paper."
        )
        lines.append(
            f"Imagine you are a {age}-year-old {sex} entrepreneur in Singapore, "
            f"{race}, with {education} education, who {'founded' if founder else 'bought over'} "
            f"the business you run (annual revenue: {business_size})."
        )
        lines.append("Please answer all questions honestly from that participant's perspective.\n")

        q_counter = 1

        # --- Section A: Risk Propensity (5 forced-choice items) ---
        lines.append("=" * 60)
        lines.append("SECTION A: RISK PREFERENCES")
        lines.append("=" * 60)
        lines.append("Please answer the following five items by choosing the alternative (\"a\" or \"b\") you would feel most comfortable with.\n")

        for item in items_a:
            options = item.get("options", [])
            lines.append(f"Q{q_counter}: Which would you prefer?")
            lines.append(f"  a) {options[0]}")
            lines.append(f"  b) {options[1]}")
            lines.append(f"  (Answer Q{q_counter}=a or Q{q_counter}=b)\n")
            item["q_idx"] = q_counter
            q_counter += 1

        # --- Section B: Cognitive Biases (7 Likert items) ---
        lines.append("=" * 60)
        lines.append("SECTION B: BUSINESS ATTITUDES")
        lines.append("=" * 60)
        lines.append("Please indicate how much you agree with each statement.")
        lines.append("Scale: 1 = Strongly Disagree, 2 = Disagree, 3 = Slightly Disagree, 4 = Neutral, 5 = Slightly Agree, 6 = Agree, 7 = Strongly Agree\n")

        for item in items_b:
            lines.append(f"Q{q_counter}: {item['question']}")
            lines.append(f"  (Answer Q{q_counter}=1 to Q{q_counter}=7)\n")
            item["q_idx"] = q_counter
            q_counter += 1

        # --- Section C: Overconfidence (10 confidence-interval items) ---
        lines.append("=" * 60)
        lines.append("SECTION C: GENERAL KNOWLEDGE")
        lines.append("=" * 60)
        lines.append("For each question below, provide a LOWER LIMIT and UPPER LIMIT such that you are 90% confident the correct answer falls within your range.")
        lines.append("If you have absolutely no idea, provide the widest reasonable range.\n")

        for item in items_c:
            unit = item.get("unit", "")
            lines.append(f"Q{q_counter} (Lower Limit) and Q{q_counter + 1} (Upper Limit): {item['question']}")
            lines.append(f"  Unit: {unit}")
            lines.append(f"  (Answer Q{q_counter}=<lower> Q{q_counter + 1}=<upper>)\n")
            item["q_idx_lower"] = q_counter
            item["q_idx_upper"] = q_counter + 1
            q_counter += 2

        # --- Section D: Case Vignette + Risk Perception + Opportunity Evaluation ---
        lines.append("=" * 60)
        lines.append("SECTION D: BUSINESS CASE EVALUATION")
        lines.append("=" * 60)
        lines.append("Please read the following case study carefully, then answer the questions.\n")
        lines.append(vignette_text)
        lines.append("")
        lines.append("Based on the case above, please indicate how much you agree with each statement.")
        lines.append("Scale: 1 = Strongly Disagree, 2 = Disagree, 3 = Slightly Disagree, 4 = Neutral, 5 = Slightly Agree, 6 = Agree, 7 = Strongly Agree\n")

        for item in items_d:
            if item["type"] == "likert_7":
                lines.append(f"Q{q_counter}: {item['question']}")
                lines.append(f"  (Answer Q{q_counter}=1 to Q{q_counter}=7)\n")
                item["q_idx"] = q_counter
                q_counter += 1
            elif item["type"] == "open_ended":
                lines.append(f"Q{q_counter}: {item['question']}")
                lines.append("  Focus on the issues that actually drive your judgment from the case as written.")
                lines.append("  Mention extra information only if you genuinely need it.")
                lines.append(f"  (Optional. Answer Q{q_counter}=<brief response>, write Q{q_counter}=No additional information needed, or omit Q{q_counter} to skip.)\n")
                item["q_idx"] = q_counter
                optional_question_numbers.append(q_counter)
                q_counter += 1

        # --- Response format ---
        lines.append("=" * 60)
        lines.append("RESPONSE FORMAT (MANDATORY)")
        lines.append("=" * 60)
        lines.append("Output ONLY answer lines in the format: Qk=<value>")
        lines.append("One answer per line. Do not include explanations.")
        if optional_question_numbers:
            optional_labels = ", ".join(f"Q{idx}" for idx in optional_question_numbers)
            required_answers = (q_counter - 1) - len(optional_question_numbers)
            lines.append(f"All numbered items except {optional_labels} are required.")
            lines.append(
                f"For {optional_labels}, respond with the issues influencing your judgment, "
                "or state that no additional information is needed."
            )
            lines.append(f"Expected number of answer lines: {required_answers} to {q_counter - 1}")
        else:
            lines.append(f"Expected number of answer lines: {q_counter - 1}")

        return "\n".join(lines)


class StudyStudy013Config(BaseStudyConfig):
    """Study config for Keh, Foo & Lim (2002) — Opportunity Evaluation under Risky Conditions."""

    prompt_builder_class = CustomPromptBuilder
    PROMPT_VARIANT = "v1"

    def create_trials(self, n_trials=None):
        spec = self.load_specification()
        n = n_trials if n_trials is not None else spec["participants"]["n"]

        # Load all materials
        mat_a = self.load_material("section_a_risk_propensity")
        mat_b = self.load_material("section_b_cognitive_biases")
        mat_c = self.load_material("section_c_overconfidence")
        mat_d = self.load_material("section_d_case_vignette")

        vignette_text = mat_d.get("vignette_text", "")

        trials = []
        for i in range(n):
            # Generate entrepreneur profiles only from demographics reported in Table 2.
            age = weighted_age_sample()
            sex = weighted_choice(SEX_OPTIONS, SEX_WEIGHTS)
            race = weighted_choice(RACE_OPTIONS, RACE_WEIGHTS)
            education = weighted_choice(EDUCATION_OPTIONS, EDUCATION_WEIGHTS)
            business_size = random.choices(BUSINESS_SIZE_OPTIONS, weights=BUSINESS_SIZE_WEIGHTS, k=1)[0]
            is_founder = random.random() < 0.79

            profile = {
                "age": age,
                "sex": sex,
                "race": race,
                "education": education,
                "business_size": business_size,
                "is_founder": is_founder,
            }

            # Deep copy items to avoid mutation across trials
            import copy
            trial = {
                "sub_study_id": "keh_foo_lim_opportunity_evaluation",
                "scenario_id": "mr_tan_vignette",
                "scenario": "mr_tan_vignette",
                "profile": profile,
                "items_a": copy.deepcopy(mat_a["items"]),
                "items_b": copy.deepcopy(mat_b["items"]),
                "items_c": copy.deepcopy(mat_c["items"]),
                "items_d": copy.deepcopy(mat_d["items"]),
                "vignette_text": vignette_text,
                "variant": self.PROMPT_VARIANT,
            }
            trials.append(trial)

        return trials

    def aggregate_results(self, raw_results):
        """Parse Qk=value responses and compute per-participant construct scores."""
        participants = []

        for record in iter_response_records(raw_results):
            participant_scores = compute_construct_scores(
                record.get("response_text", ""),
                record.get("trial_info", {}),
            )
            if participant_scores is not None:
                participants.append(participant_scores)

        # Compute descriptive statistics
        if not participants:
            return {"participants": [], "descriptive_statistics": {}, "n_valid": 0}

        constructs = [
            "risk_propensity",
            "planning_fallacy",
            "illusion_of_control",
            "overconfidence",
            "risk_perception",
            "opportunity_evaluation",
            "small_numbers",
            "age",
        ]

        desc_stats = {}
        for c in constructs:
            values = [p[c] for p in participants if p.get(c) is not None]
            if not values:
                continue
            desc_stats[c] = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": len(values),
            }

        return {
            "participants": participants,
            "descriptive_statistics": desc_stats,
            "n_valid": len(participants),
        }
