"""Runnable adapter for the decentralized two-sided matching-market experiment
(Echenique, Robinson-Cortes & Yariv, "An Experimental Study of Decentralized Matching").

Every food-side and color-side participant in the original experiment is played here
by an independent call to the injected `llm(prompt) -> str`. No participant's action
is scripted, sampled from a fixed distribution, or pre-computed -- see task/task.json
for the full fidelity rationale and the (documented) real-time-clock discretization.

Runs fully offline: all payoffs and instructions are read from local JSON files under
materials/; no network access is performed or required.
"""

import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
MATERIALS_DIR = PACKAGE_ROOT / "materials"

TASK_JSON_PATH = HERE / "task.json"
PAYOFF_MATRICES_PATH = MATERIALS_DIR / "payoff_matrices_verbatim.json"
INSTRUCTIONS_PATH = MATERIALS_DIR / "instructions_script.json"


class BlockedError(RuntimeError):
    """Raised when a required researcher decision or material is missing."""


def _load_json(path):
    if not path.exists():
        raise BlockedError(
            "Required material file not found: %s. See audit/missing_information.json." % path
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_conditions():
    task = _load_json(TASK_JSON_PATH)
    return {c["arm"]: c for c in task["conditions"]}


def load_payoff_treatment(index):
    data = _load_json(PAYOFF_MATRICES_PATH)
    for t in data["treatments"]:
        if t["index"] == index:
            return t
    raise BlockedError("payoff_treatment_index %s not found in payoff_matrices_verbatim.json" % index)


def build_instructions(condition):
    """Return the participant-facing instructions text for a given condition arm.

    The main (bilateral, 8-per-side) arms get the verbatim CASSEL script. Unilateral
    and large arms get that same verbatim script with the group-size number and (for
    unilateral arms) an explicit proposing-rule sentence substituted in -- these
    substitutions are marked 'derived', not verbatim; see materials/instructions_script.json
    'not_covered_by_this_script' and audit/missing_information.json.
    """
    doc = _load_json(INSTRUCTIONS_PATH)
    slides = [s["text"] for s in doc["slides"]]
    text = "\n\n".join(slides)
    n = condition["n_per_side"]
    if n != 8:
        text = text.replace("There will be 8 colors and 8 fruit in your group.",
                             "There will be %d colors and %d fruit in your group." % (n, n))
    if condition["proposing"] == "unilateral_food_only":
        text = text.replace(
            "You make an offer by clicking on the name you want to match with (right panel).",
            "Only participants assigned the role of \"fruit\" (food) may make offers in this "
            "session; if you are a \"color\" you may only accept or reject offers you receive. "
            "You make an offer by clicking on the name you want to match with (right panel)."
        )
    return text


def format_matrix(matrix, n):
    foods = ["food-%d" % (i + 1) for i in range(n)]
    colors = ["color-%d" % (j + 1) for j in range(n)]
    header = "            " + "  ".join("%9s" % c for c in colors)
    lines = [header]
    for i, f in enumerate(foods):
        row = matrix[f]
        cells = ["%4d,%4d" % (row[j][0], row[j][1]) for j in range(n)]
        lines.append("%-10s  " % f + "  ".join(cells))
    return "\n".join(lines)


def describe_state(matches, n):
    foods = ["food-%d" % (i + 1) for i in range(n)]
    parts = []
    for f in foods:
        c = matches.get(f)
        parts.append("%s: %s" % (f, c if c else "unmatched"))
    return "; ".join(parts)


def describe_history(history):
    if not history:
        return "(none yet)"
    lines = []
    for h in history:
        if h["type"] == "made":
            lines.append("You offered %s; it was %s." % (h["target"], h["outcome"]))
        elif h["type"] == "received":
            lines.append("You received an offer from %s and you %s." % (h["source"], h["outcome"]))
    return " ".join(lines)


PROPOSE_INSTRUCTIONS = (
    "You are participant '%(role)s' (a %(side)s) in a decentralized matching market. "
    "You and everyone else can see every possible match's payoff below. A match pays "
    "the two people involved the two numbers shown for their pair; if you end the market "
    "unmatched you earn 0. You may, right now, either propose a match to any one member "
    "of the opposite side (whether or not they are currently matched), or pass. Offers are "
    "not binding until the market ends, and you may later receive or make other offers.\n\n"
    "Full payoff matrix (food_payoff, color_payoff for each food-color pair):\n%(matrix)s\n\n"
    "Current matches (all tentative): %(state)s\n\n"
    "Your own history so far: %(history)s\n\n"
    "Reply with exactly one line: either 'PASS' or 'OFFER: <role>' (e.g. 'OFFER: color-3'). "
    "%(target_note)s"
)

RESPOND_INSTRUCTIONS = (
    "You are participant '%(role)s' (a %(side)s) in a decentralized matching market. "
    "You have just received a match offer from %(source)s. You may accept (breaking your "
    "current match, if any) or reject. You have full information about everyone's payoffs.\n\n"
    "Full payoff matrix (food_payoff, color_payoff for each food-color pair):\n%(matrix)s\n\n"
    "Current matches (all tentative, before this offer): %(state)s\n\n"
    "Your own history so far: %(history)s\n\n"
    "Reply with exactly one line: either 'ACCEPT' or 'REJECT'."
)


def _parse_propose(reply, valid_targets):
    if not reply:
        return ("PASS", None, True)
    m = re.search(r"OFFER\s*:\s*([a-zA-Z]+-\d+)", reply, re.IGNORECASE)
    if m:
        target = m.group(1).lower()
        for v in valid_targets:
            if v.lower() == target:
                return ("OFFER", v, False)
        return ("PASS", None, True)
    if re.search(r"\bPASS\b", reply, re.IGNORECASE):
        return ("PASS", None, False)
    return ("PASS", None, True)


def _parse_respond(reply):
    if not reply:
        return ("REJECT", True)
    if re.search(r"\bACCEPT\b", reply, re.IGNORECASE):
        return ("ACCEPT", False)
    if re.search(r"\bREJECT\b", reply, re.IGNORECASE):
        return ("REJECT", False)
    return ("REJECT", True)


def run_market(llm, condition, rng, seed_label):
    n = condition["n_per_side"]
    treatment = load_payoff_treatment(condition["payoff_treatment_index"])
    matrix = treatment["matrix"]
    foods = ["food-%d" % (i + 1) for i in range(n)]
    colors = ["color-%d" % (j + 1) for j in range(n)]
    bilateral = condition["proposing"] == "bilateral"
    proposers = foods + colors if bilateral else list(foods)

    matches = {}  # food_role -> color_role, and reverse maintained via lookup
    rev_matches = {}  # color_role -> food_role
    history = {a: [] for a in foods + colors}
    events = []

    task = _load_json(TASK_JSON_PATH)
    td = task["timing_discretization"]
    size_key = condition["size"]
    inactivity_ticks = td["default_inactivity_ticks"][size_key]
    max_ticks = td["default_max_ticks"][size_key]

    no_offer_streak = 0
    tick = 0
    terminated_by = "max_ticks"
    while tick < max_ticks:
        tick += 1
        actor = rng.choice(proposers)
        role_side = "food" if actor.startswith("food") else "color"
        valid_targets = colors if role_side == "food" else foods
        matrix_text = format_matrix(matrix, n)
        state_text = describe_state(matches, n)
        history_text = describe_history(history[actor])
        prompt = PROPOSE_INSTRUCTIONS % {
            "role": actor,
            "side": role_side,
            "matrix": matrix_text,
            "state": state_text,
            "history": history_text,
            "target_note": "Valid targets: %s." % ", ".join(valid_targets),
        }
        reply = llm(prompt)
        action, target, fallback = _parse_propose(reply, valid_targets)
        event = {"tick": tick, "actor": actor, "action": action, "target": target,
                 "parse_fallback": fallback}

        if action == "PASS":
            no_offer_streak += 1
            events.append(event)
        else:
            no_offer_streak = 0
            recipient = target
            matrix_text2 = format_matrix(matrix, n)
            state_text2 = describe_state(matches, n)
            recipient_side = "color" if role_side == "food" else "food"
            history_text2 = describe_history(history[recipient])
            prompt2 = RESPOND_INSTRUCTIONS % {
                "role": recipient,
                "side": recipient_side,
                "source": actor,
                "matrix": matrix_text2,
                "state": state_text2,
                "history": history_text2,
            }
            reply2 = llm(prompt2)
            decision, fallback2 = _parse_respond(reply2)
            event["response"] = decision
            event["response_parse_fallback"] = fallback2
            events.append(event)

            outcome = "accepted" if decision == "ACCEPT" else "rejected"
            history[actor].append({"type": "made", "target": recipient, "outcome": outcome})
            history[recipient].append({"type": "received", "source": actor, "outcome": outcome})

            if decision == "ACCEPT":
                if role_side == "food":
                    f, c = actor, recipient
                else:
                    f, c = recipient, actor
                old_c = matches.get(f)
                if old_c:
                    rev_matches.pop(old_c, None)
                old_f = rev_matches.get(c)
                if old_f:
                    matches.pop(old_f, None)
                matches[f] = c
                rev_matches[c] = f

        if no_offer_streak >= inactivity_ticks:
            terminated_by = "inactivity"
            break

    final_matching = {f: matches.get(f) for f in foods}

    return {
        "arm": condition["arm"],
        "structure": condition["structure"],
        "proposing": condition["proposing"],
        "size": condition["size"],
        "n_per_side": n,
        "payoff_treatment_index": condition["payoff_treatment_index"],
        "seed_label": seed_label,
        "events": events,
        "final_matching": final_matching,
        "n_ticks_used": tick,
        "terminated_by": terminated_by,
        "payoff_matrix": matrix,
    }


def run_sessions(llm, seed, n, arms=None, on_session=None):
    """Run n sessions (experimental markets) for every condition arm in task.json,
    or only the arms named in `arms` when it is provided.

    Returns a flat list of session-log dicts, each carrying its own condition/arm
    labels so evaluation.py can group and compare across arms.
    """
    conditions = load_conditions()
    if arms:
        conditions = {name: condition for name, condition in conditions.items() if name in arms}
    base_rng = random.Random(seed)
    sessions = []
    for arm_name, condition in conditions.items():
        for i in range(n):
            session_seed = base_rng.randint(0, 2**31 - 1)
            rng = random.Random(session_seed)
            seed_label = "%s#%d(seed=%d)" % (arm_name, i, session_seed)
            session = run_market(llm, condition, rng, seed_label)
            sessions.append(session)
            if on_session:
                on_session(session)
    return sessions


def _stub_llm(prompt):
    """Deterministic offline stub used only by --smoke-test: a proposer offers the
    first untried valid target (or passes if all have been tried), and a recipient
    alternates accept/reject based on the prompt's own hash. This is NOT a research
    participant model -- it only exists to exercise run_sessions()/evaluate() without
    a real model plugged in."""
    if "Reply with exactly one line: either 'PASS' or 'OFFER" in prompt:
        targets = re.findall(r"Valid targets: (.+)\.", prompt)
        options = [t.strip() for t in targets[0].split(",")] if targets else []
        tried = set(re.findall(r"You offered ([a-zA-Z]+-\d+)", prompt))
        remaining = [o for o in options if o not in tried]
        return "OFFER: %s" % remaining[0] if remaining else "PASS"
    return "ACCEPT" if hash(prompt) % 2 == 0 else "REJECT"


def _smoke_test():
    conditions = load_conditions()
    print("Loaded %d condition arms:" % len(conditions), ", ".join(conditions))
    sessions = run_sessions(_stub_llm, seed=1, n=1)
    print("Ran %d smoke-test sessions with a deterministic stub LLM (no network)." % len(sessions))
    for s in sessions:
        matched = sum(1 for v in s["final_matching"].values() if v)
        print(" - %-24s ticks=%3d terminated_by=%-10s matched=%d/%d" % (
            s["arm"], s["n_ticks_used"], s["terminated_by"], matched, s["n_per_side"]))
    import sys
    sys.path.insert(0, str(HERE.parent / "evaluation"))
    try:
        from evaluation import evaluate  # type: ignore
    except Exception:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "evaluation", str(HERE.parent / "evaluation" / "evaluation.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        evaluate = mod.evaluate
    result = evaluate(sessions)
    print("evaluate() returned keys:", list(result.keys()))
    print("OK: adapter runs end-to-end without network access.")


if __name__ == "__main__":
    import sys
    if "--smoke-test" in sys.argv:
        _smoke_test()
    else:
        print("Usage: python adapter.py --smoke-test")
