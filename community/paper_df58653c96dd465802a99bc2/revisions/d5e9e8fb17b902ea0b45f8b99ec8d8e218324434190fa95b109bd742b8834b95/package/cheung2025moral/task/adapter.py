#!/usr/bin/env python3
"""Replay adapter for Cheung, Maier & Lieder: moral dilemmas (Study 1 moral part, Study 2).

Offline. Every original human participant is one AI participant: `llm(prompt) -> str` is injected and
called once per participant (Study 2) or once per vignette of a sequential session (Study 1).
Nothing here answers for a participant.

    python adapter.py --smoke-test
"""
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAT = json.loads((HERE.parent / "materials" / "materials.json").read_text(encoding="utf-8"))
SYSTEM = MAT["system_prompts"]["participant_moral_dilemmas"]
INSTRUCTION = MAT["response_instruction"]
S1 = MAT["studies"]["s1"]["vignettes"]
S2 = MAT["studies"]["s2"]["vignettes"]
S2_KEYS = list(S2)
FRAMING = {"s2_original": "original", "s2_yes_no": "yes_no", "s2_omission": "omission"}
CONDITION = {"original": "c2_orig", "yes_no": "c2_yn", "omission": "c2_om"}
ARMS = ["s1_moral_sequential", "s2_original", "s2_yes_no", "s2_omission"]


def _item(study, condition):
    """The material for a condition id: 's1' + vignette key, or 's2' + 'framing/vignette'."""
    if study == "s1":
        v = S1[condition]
        return v["text"], v["answer_key"], v["cbr_coincides_with"] == "action"
    framing, key = condition.split("/")
    v = S2[key][framing]
    return v["text"], v["answer_key"], v["cbr_is_action"]


def build_prompt(study, condition, history=None):
    """Exactly what one AI participant reads. `history` (Study 1): [(condition, answer), ...] earlier in this session."""
    text, _, _ = _item(study, condition)
    parts = [SYSTEM, ""]
    for prev, answer in history or []:
        parts += [_item(study, prev)[0], INSTRUCTION, "Your answer: %s" % (answer or "(no valid answer)"), ""]
    parts += [text, INSTRUCTION]
    return "\n".join(parts)


def parse_reply(study, condition, reply):
    """First standalone yes/no in the reply, mapped to the CBR option through the vignette's answer key."""
    _, key, cbr_is_action = _item(study, condition)
    m = re.search(r"\b(yes|no)\b", reply or "", re.I)
    answer = m.group(1).lower() if m else None
    return {"answer": answer, "cbr": None if answer is None else key[answer] == "CBR",
            "yes_is_cbr": key["yes"] == "CBR", "cbr_is_action": cbr_is_action}


def _trial(study, condition, vignette, framing, llm, history=None):
    prompt = build_prompt(study, condition, history)
    reply = llm(prompt)
    out = parse_reply(study, condition, reply)
    out.update({"vignette": vignette, "framing": framing, "reply": reply})
    return out


def run_sessions(llm, seed, n, arms=None):
    """n AI participants per arm. arms=None runs every arm: ARMS."""
    arms = list(arms) if arms else list(ARMS)
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise ValueError("unknown arm(s) %s; available: %s" % (unknown, ARMS))
    rng = random.Random(seed)
    sessions = []
    for arm in arms:
        if arm == "s1_moral_sequential":
            for p in range(n):
                order = list(S1)
                rng.shuffle(order)
                history, trials = [], []
                for key in order:
                    t = _trial("s1", key, key, "action" if S1[key]["cbr_coincides_with"] == "action" else "omission", llm, history)
                    trials.append(t)
                    history.append((key, t["answer"]))
                sessions.append({"study": "s1", "arm": arm, "condition": "c1_action|c1_omission", "participant": p, "trials": trials})
        else:
            framing = FRAMING[arm]
            keys = [S2_KEYS[i % len(S2_KEYS)] for i in range(n)]
            rng.shuffle(keys)
            for p, key in enumerate(keys):
                t = _trial("s2", "%s/%s" % (framing, key), key, framing, llm)
                sessions.append({"study": "s2", "arm": arm, "condition": CONDITION[framing], "participant": p, "trials": [t]})
    return sessions


def _smoke():
    # A stub stands in for the injected model only to exercise the code path; it is not a participant.
    stub = lambda prompt: "Yes" if len(prompt) % 2 else "No."
    sessions = run_sessions(stub, seed=1, n=2)
    assert len(sessions) == 2 + 3 * 2 and len(sessions[0]["trials"]) == 13
    assert all(t["answer"] in ("yes", "no") for s in sessions for t in s["trials"])
    prompt = build_prompt("s2", "yes_no/suicide")
    assert prompt.endswith(INSTRUCTION) and SYSTEM in prompt
    sys.path.insert(0, str(HERE.parent / "evaluation"))
    import evaluation
    out = evaluation.evaluate(sessions)
    json.dumps(out)
    print("smoke test ok:", ", ".join(sorted(k for k in out if k.startswith("study"))))


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        _smoke()
    else:
        print(__doc__)
