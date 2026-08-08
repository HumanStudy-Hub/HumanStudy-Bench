"""Shared playground constants: run limits, prompt presets, and OpenRouter routing."""

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# A run on the shared HumanStudy-Hub key is capped so one playground session
# cannot spend the project's OpenRouter budget. A contributor who supplies their
# own key is allowed the larger budget.
SHARED_KEY_MAX_TRIALS = 60
OWN_KEY_MAX_TRIALS = 600

SHARED_KEY_MAX_PER_SCENARIO = 10
OWN_KEY_MAX_PER_SCENARIO = 80

DEFAULT_PER_SCENARIO = 8
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_PRESET = "v3_human_plus_demo"
DEFAULT_WORKERS = 8

# Presets shipped by src/agents/custom_methods/. "custom" is the playground's own
# marker for a prompt the researcher wrote in the prompt designer.
PROMPT_PRESETS = ("v1_empty", "v2_human", "v3_human_plus_demo", "v4_background")
CUSTOM_PRESET = "custom"

# Demographic fields the prompt designer can override on every participant.
DEMOGRAPHIC_FIELDS = ("age", "gender", "education", "background", "population", "persona")


def trial_limits(has_own_key: bool) -> tuple[int, int]:
    """Return (max total trials, max trials per scenario) for this run."""
    if has_own_key:
        return OWN_KEY_MAX_TRIALS, OWN_KEY_MAX_PER_SCENARIO
    return SHARED_KEY_MAX_TRIALS, SHARED_KEY_MAX_PER_SCENARIO
