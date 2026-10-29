#!/usr/bin/env python3
"""Offline replay adapter: Hoogeveen, Sarafoglou & Wagenmakers (2020), laypeople predicting replicability.

Every original participant is an AI participant: `run_sessions(llm, seed, n, arms=None)` calls the injected
`llm(prompt) -> str` once per participant. Nothing here answers for a participant.
`--smoke-test` only checks prompt building and reply parsing with a format-only stub reply; it is not a simulation.
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MATERIALS = json.loads((HERE.parent / "materials" / "materials.json").read_text(encoding="utf-8"))
ARMS = {"c-desc": "description_only", "c-evid": "description_plus_evidence"}
KEYS = [i["key"] for i in MATERIALS["items"]]
BY_KEY = {i["key"]: i for i in MATERIALS["items"]}


def _page(item, k, total, condition):
    head, _, rest = item["description"].partition("?")
    lines = ["Page %d / %d" % (k, total), "", "Research question", head + "?", "",
             "How was this measured? / What was found?", rest.strip()]
    if condition == "c-evid":
        lines += ["", "Evidence strength in Bayes factor", "BF = %s" % item["bf"],
                  "This qualifies as %s evidence." % item["verbal"]]
    lines += ["", "[ ] " + MATERIALS["response_items"]["tick"],
              "", MATERIALS["response_items"]["replicate"] + "  (Yes / No)",
              "", MATERIALS["response_items"]["confidence"] + "  (slider from 0% through 50% to 100%; the handle starts at 50%)"]
    return "\n".join(lines)


def build_prompt(study_id="s1", condition_id="c-desc", order=None):
    """The complete prompt for ONE participant. `order` is the participant's randomized list of study keys."""
    order = list(order or KEYS)
    parts = ["You are taking part in an online survey as a participant. Respond as an actual participant would.",
             "", MATERIALS["instructions"]["text"]]
    if condition_id == "c-evid":
        parts += ["", MATERIALS["bayes_factor_explanation"]["text"]]
    parts += ["", "You will now see %d study pages, one at a time, in this order. Give your own judgment for each page." % len(order)]
    for k, key in enumerate(order, 1):
        parts += ["", "-----", _page(BY_KEY[key], k, len(order), condition_id)]
    parts += ["", "-----", "",
              "Reply with only a JSON list with one object per page, in page order, like "
              '[{"page": 1, "not_understood": false, "replicate": "yes", "confidence": 60}, ...]. '
              '"replicate" is "yes" or "no"; "confidence" is a number from 0 to 100.']
    return "\n".join(parts)


def parse_reply(study_id, condition_id, reply, order=None):
    """Parse a reply into per-page answers; returns {"valid": bool, "answers": [...], "error": str|None}."""
    order = list(order or KEYS)
    text = reply if isinstance(reply, str) else json.dumps(reply)
    a, b = text.find("["), text.rfind("]")
    try:
        rows = json.loads(text[a:b + 1])
        answers = []
        for k, key in enumerate(order, 1):
            row = rows[k - 1]
            rep = str(row["replicate"]).strip().lower()
            if rep not in ("yes", "no"):
                raise ValueError("replicate must be yes/no on page %d" % k)
            conf = float(row["confidence"])
            if not 0 <= conf <= 100:
                raise ValueError("confidence out of range on page %d" % k)
            answers.append({"key": key, "page": k, "replicate": rep, "confidence": conf,
                            "not_understood": bool(row.get("not_understood", False))})
        if len(rows) != len(order):
            raise ValueError("expected %d rows, got %d" % (len(order), len(rows)))
        return {"valid": True, "answers": answers, "error": None}
    except Exception as exc:  # parse failure is recorded, never repaired
        return {"valid": False, "answers": [], "error": "%s: %s" % (type(exc).__name__, exc)}


def run_sessions(llm, seed=20261015, n=5, arms=None):
    """n AI participants per arm (the sampling unit is the participant). Each sees a random page order."""
    arms = list(arms) if arms else list(ARMS)
    rng = random.Random(seed)
    sessions = []
    for arm in arms:
        cond = next(c for c, a in ARMS.items() if a == arm or c == arm)
        for p in range(int(n)):
            order = KEYS[:]
            rng.shuffle(order)
            prompt = build_prompt("s1", cond, order)
            reply = llm(prompt)
            parsed = parse_reply("s1", cond, reply, order)
            sessions.append({"study": "s1", "condition": cond, "arm": ARMS[cond], "participant": "%s-%d" % (ARMS[cond], p + 1),
                             "order": order, "reply": reply, "valid": parsed["valid"], "error": parsed["error"],
                             "answers": parsed["answers"]})
    return sessions


def _smoke_test():
    order = KEYS[:]
    random.Random(1).shuffle(order)
    for cond in ARMS:
        prompt = build_prompt("s1", cond, order)
        assert ("Bayes factor (BF)" in prompt) == (cond == "c-evid")
        assert prompt.count("Page ") >= len(KEYS) and "Will this study replicate?" in prompt
    fmt = json.dumps([{"page": k, "not_understood": False, "replicate": "yes", "confidence": 50} for k in range(1, len(KEYS) + 1)])
    ok = parse_reply("s1", "c-desc", fmt, order)
    assert ok["valid"] and len(ok["answers"]) == len(KEYS)
    assert not parse_reply("s1", "c-desc", "no answer", order)["valid"]
    print("smoke test passed: prompts build for both arms and a format-only reply parses (no simulation was run)")


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        _smoke_test()
    else:
        print(__doc__)
