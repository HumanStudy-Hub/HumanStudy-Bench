#!/usr/bin/env python3
"""Replay adapter for Tessler & Franke (2018), Experiments 1 and 2. Offline; no network.

run_sessions(llm, seed, n, arms=None): one AI participant per original participant, `n` per arm.
Participants rate statements trial by trial; each model call sees only that participant's own earlier
statements and ratings (never other participants' responses, never the hypothesis).
build_prompt(study_id, condition_id, trial=None, history=None) / parse_reply(study_id, condition_id, reply)
are module-level so a simulation can run participants in parallel.
"""
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAT = json.loads((HERE.parent / "materials/materials.json").read_text(encoding="utf-8"))

ADJ_TYPES = ["positive", "antonym", "negated_positive", "negated_antonym"]
ARMS = {
    "exp1": {"study": "exp1", "condition": None, "antonym_type": None, "context": "single"},
    "exp2-morphological-single": {"study": "exp2", "condition": "c2-morph-single", "antonym_type": "morphological", "context": "single"},
    "exp2-lexical-single": {"study": "exp2", "condition": "c2-lex-single", "antonym_type": "lexical", "context": "single"},
    "exp2-morphological-multiple": {"study": "exp2", "condition": "c2-morph-multi", "antonym_type": "morphological", "context": "multiple"},
    "exp2-lexical-multiple": {"study": "exp2", "condition": "c2-lex-multi", "antonym_type": "lexical", "context": "multiple"},
}
CONDITION_ARM = {a["condition"]: k for k, a in ARMS.items() if a["condition"]}
NAMES = MAT["replay_adaptations"]["names"]


def _sets(study, antonym_type):
    m = MAT["studies"][study]
    if study == "exp1":
        rows = m["morphological_sets"] if antonym_type == "morphological" else m["lexical_sets"]
        out = []
        for r in rows:
            pos, ant = [w.strip() for w in r.split(",")]
            out.append({"positive": pos, "antonym": ant})
        return out
    key = "morphological_antonym" if antonym_type == "morphological" else "lexical_antonym"
    return [{"positive": r["positive"], "antonym": r[key]} for r in m["sets"]]


def _word(adj_set, adj_type):
    return {"positive": adj_set["positive"], "antonym": adj_set["antonym"],
            "negated_positive": "not " + adj_set["positive"], "negated_antonym": "not " + adj_set["antonym"]}[adj_type]


def _statement(name, adj_set, adj_type):
    return "%s is %s." % (name, _word(adj_set, adj_type))


def _endpoints(study, adj_set):
    if study == "exp1":
        return ("the most %s person" % adj_set["antonym"], "the most %s person" % adj_set["positive"])
    return ("the most negative person in the world", "the most positive person in the world")


def plan_trials(arm, rng):
    """Trial plan for one participant. Single: one dict per trial. Multiple: one dict per display."""
    a = ARMS[arm]
    trials = []
    if arm == "exp1":
        for atype in ("morphological", "lexical"):
            sets = rng.sample(_sets("exp1", atype), 8)
            types = ADJ_TYPES * 2
            rng.shuffle(types)
            for s, t in zip(sets, types):
                trials.append({"antonym_type": atype, "set": s, "adjective_type": t})
        rng.shuffle(trials)
    elif a["context"] == "single":
        sets = rng.sample(_sets("exp2", a["antonym_type"]), 12)
        types = ADJ_TYPES * 3
        rng.shuffle(types)
        for s, t in zip(sets, types):
            trials.append({"antonym_type": a["antonym_type"], "set": s, "adjective_type": t})
    else:
        for s in rng.sample(_sets("exp2", a["antonym_type"]), 3):
            order = ADJ_TYPES[:]
            rng.shuffle(order)
            names = rng.sample(NAMES, 4)
            trials.append({"antonym_type": a["antonym_type"], "set": s, "adjective_types": order, "names": names})
    for t in trials:
        if "adjective_type" in t:
            t["name"] = rng.choice(NAMES)
    return trials


def _history_text(history):
    if not history:
        return ""
    lines = ["Earlier in this session you rated:"]
    for h in history:
        lines.append("- %s -> %s" % (" / ".join(h["statements"]), ", ".join(str(r) for r in h["ratings"])))
    return "\n".join(lines) + "\n\n"


def build_prompt(study_id, condition_id=None, trial=None, history=None):
    arm = CONDITION_ARM.get(condition_id, "exp1" if study_id == "exp1" else "exp2-morphological-single")
    a = ARMS[arm]
    if trial is None:  # canonical first trial of this arm, used for display and testing
        trial = plan_trials(arm, random.Random(0))[0]
    scale = MAT["replay_adaptations"]["scale"]
    lo, hi = _endpoints(a["study"], trial["set"])
    head = MAT["replay_adaptations"]["intro"] + "\n\n" + _history_text(history)
    if "adjective_type" in trial:
        st = _statement(trial["name"], trial["set"], trial["adjective_type"])
        return (head + "Statement: \"%s\"\n\nPlace the person on a slider from 0 (%s) to 100 (%s).\n%s\n"
                "Reply with only JSON like {\"rating\": 50}." % (st, lo, hi, scale))
    sts = [_statement(n, trial["set"], t) for n, t in zip(trial["names"], trial["adjective_types"])]
    lines = "\n".join("%d. \"%s\"" % (i + 1, s) for i, s in enumerate(sts))
    return (head + "The four statements below are shown together on one screen, each about a different person:\n%s\n\n"
            "Place each person on a slider from 0 (%s) to 100 (%s).\n%s\n"
            "Reply with only JSON like {\"ratings\": [50, 50, 50, 50]} in the order shown." % (lines, lo, hi, scale))


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return min(100.0, max(0.0, v))


def parse_reply(study_id, condition_id, reply):
    """Return {"rating": float|None} or {"ratings": [float|None]*4} parsed from a reply, else None values."""
    text = reply if isinstance(reply, str) else json.dumps(reply)
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if "ratings" in d and isinstance(d["ratings"], list):
                return {"ratings": [_num(v) for v in d["ratings"]][:4] + [None] * (4 - len(d["ratings"]))}
            if "rating" in d:
                return {"rating": _num(d["rating"])}
        except ValueError:
            pass
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    arm = CONDITION_ARM.get(condition_id)
    if arm and ARMS[arm]["context"] == "multiple":
        vals = [_num(v) for v in nums[:4]]
        return {"ratings": vals + [None] * (4 - len(vals))}
    return {"rating": _num(nums[0]) if nums else None}


def run_sessions(llm, seed, n, arms=None):
    sessions = []
    for arm in (arms or list(ARMS)):
        a = ARMS[arm]
        for i in range(n):
            rng = random.Random("%s|%s|%d" % (seed, arm, i))
            history, rows = [], []
            for tr in plan_trials(arm, rng):
                prompt = build_prompt(a["study"], a["condition"], tr, history)
                reply = llm(prompt)
                parsed = parse_reply(a["study"], a["condition"], reply)
                if "adjective_type" in tr:
                    stmts = [_statement(tr["name"], tr["set"], tr["adjective_type"])]
                    ratings = [parsed.get("rating")]
                    types = [tr["adjective_type"]]
                else:
                    stmts = [_statement(n_, tr["set"], t) for n_, t in zip(tr["names"], tr["adjective_types"])]
                    ratings = parsed.get("ratings", [None] * 4)
                    types = tr["adjective_types"]
                for t, s, r in zip(types, stmts, ratings):
                    rows.append({"adjective_type": t, "antonym_type": tr["antonym_type"], "positive": tr["set"]["positive"],
                                 "statement": s, "rating": r, "reply": reply})
                history.append({"statements": stmts, "ratings": ratings})
            sessions.append({"study": a["study"], "condition": a["condition"], "arm": arm, "context": a["context"],
                             "participant": i, "trials": rows})
    return sessions


def _smoke_test():
    """Plumbing check only: a stub reply function exercises prompts, parsing and session structure."""
    def stub(prompt):
        return '{"ratings": [50, 50, 50, 50]}' if "shown together" in prompt else '{"rating": 50}'
    ss = run_sessions(stub, 1, 1)
    assert len(ss) == len(ARMS), "one session per arm"
    want = {"exp1": 16, "exp2-morphological-single": 12, "exp2-lexical-single": 12,
            "exp2-morphological-multiple": 12, "exp2-lexical-multiple": 12}
    for s in ss:
        assert len(s["trials"]) == want[s["arm"]], (s["arm"], len(s["trials"]))
        assert all(t["rating"] == 50.0 for t in s["trials"])
        counts = {t: sum(1 for r in s["trials"] if r["adjective_type"] == t) for t in ADJ_TYPES}
        assert len(set(counts.values())) == 1, counts
    for a in ARMS.values():
        p = build_prompt(a["study"], a["condition"])
        assert "hypothes" not in p.lower() and "partial ordering" not in p.lower()
    assert parse_reply("exp1", None, "I'd say 72")["rating"] == 72.0
    print("smoke test ok: %d arms" % len(ss))


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        _smoke_test()
    else:
        print(__doc__)
