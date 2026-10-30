#!/usr/bin/env python3
"""Offline adapter for Franke & Degen (2016), Experiments 1 (comprehension) and 2 (production).

Every original participant is one AI participant: `llm(prompt) -> str` is called once per trial and the
prompt contains only the instructions, that participant's own earlier trials and choices, and the current
trial. Stimulus displays follow the generation rules of the paper (see materials/materials.json); the
messages shown in comprehension are generated from the design, not taken from other AI participants.

    python adapter.py --smoke-test        offline plumbing test with a constant stub reply (not a simulation)
"""
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAT = json.loads((HERE.parent / "materials" / "materials.json").read_text(encoding="utf-8"))

KINDS = ["robot", "green monster", "purple monster"]
ACCS = ["scarf", "blue hat", "red hat"]
EXPR_KINDS = ["green monster", "purple monster"]
EXPR_ACCS = ["blue hat", "red hat"]
MSGS = EXPR_KINDS + EXPR_ACCS
STUDIES = ("exp1", "exp2")
LETTERS = "ABC"


def mk(a, b):
    """Object (kind, accessory) from two features of different dimensions, in either order."""
    return (a, b) if a in KINDS else (b, a)


def other_dim(f):
    return EXPR_ACCS if f in KINDS else EXPR_KINDS


def inexpressible_for(f):
    """The inexpressible feature along the dimension other than f's."""
    return "scarf" if f in KINDS else "robot"


def describe(o):
    return "a %s with %s" % (o[0], "a scarf" if o[1] == "scarf" else "a " + o[1])


def true_of(msg, o):
    return msg in o


def unique_msgs(i, objs):
    """Messages true of objs[i] and of no other object."""
    return [m for m in MSGS if true_of(m, objs[i]) and not any(true_of(m, o) for j, o in enumerate(objs) if j != i)]


# ---------------------------------------------------------------- comprehension displays (Exp 1)
def comp_display(rng, kind):
    """Return objs [target, competitor, distractor] for the simple/complex implicature displays."""
    m = rng.choice(MSGS)
    if kind == "simple":
        t = mk(m, inexpressible_for(m))
        c = mk(m, rng.choice(other_dim(m)))
        kl = [k for k in KINDS if k not in (t[0], c[0])]
        al = [a for a in ACCS if a not in (t[1], c[1])]
        d = (rng.choice(kl), rng.choice(al))
    else:
        xs = other_dim(m)
        xt = rng.choice(xs)
        xc = [x for x in xs if x != xt][0]
        t, c = mk(m, xt), mk(m, xc)
        d = mk(xt, inexpressible_for(xt))
    return m, [t, c, d]


def comp_trial(rng, cond):
    """cond: simple | complex | f_csimple | f_dsimple | f_ccomplex | unamb | amb."""
    if cond in ("simple", "complex"):
        m, objs = comp_display(rng, cond)
        roles = {0: "target", 1: "competitor", 2: "distractor"}
        trig, tgt = m, 0
    elif cond.startswith("f_"):
        while True:
            kind = "complex" if cond == "f_ccomplex" else "simple"
            m, objs = comp_display(rng, kind)
            tgt = {"f_csimple": 1, "f_dsimple": 2, "f_ccomplex": 1}[cond]
            u = unique_msgs(tgt, objs)
            if u:
                trig = rng.choice(u)
                break
        roles = {i: ("target" if i == tgt else "distractor") for i in range(3)}
    elif cond == "unamb":
        while True:
            objs = [(rng.choice(KINDS), rng.choice(ACCS)) for _ in range(3)]
            if len(set(objs)) < 3:
                continue
            tgt = rng.randrange(3)
            u = unique_msgs(tgt, objs)
            if u:
                trig = rng.choice(u)
                break
        roles = {i: ("target" if i == tgt else "distractor") for i in range(3)}
    else:  # amb: two identical objects both bearing the message feature, third object lacks it
        trig = rng.choice(MSGS)
        o1 = mk(trig, rng.choice(ACCS if trig in KINDS else KINDS))
        third = rng.choice([(k, a) for k in KINDS for a in ACCS if not true_of(trig, (k, a))])
        objs = [o1, o1, third]
        roles = {0: "target", 1: "competitor", 2: "distractor"}
    order = list(range(3))
    rng.shuffle(order)
    shown = [objs[i] for i in order]
    shown_roles = [roles[i] for i in order]
    return {"cond": cond, "analysis_cond": cond if cond in ("simple", "complex", "unamb", "amb") else "filler",
            "objects": [list(o) for o in shown], "roles": shown_roles, "message": trig}


# ---------------------------------------------------------------- production displays (Exp 2)
def prod_display(rng, kind):
    m = rng.choice(MSGS)
    x = rng.choice(other_dim(m))
    trig = mk(m, x)
    second = mk(x, inexpressible_for(x))
    rest = [f for f in MSGS if f not in (m, x)]
    if kind == "simple":
        third = mk(rest[0], rest[1])
    else:
        xprime = [f for f in other_dim(m) if f != x][0]
        third = mk(m, xprime)
    return m, x, [trig, second, third]


def prod_trial(rng, cond):
    """cond: simple | complex | f_csimple | f_dsimple | f_dcomplex | unamb | amb."""
    if cond in ("simple", "complex"):
        m, x, objs = prod_display(rng, cond)
        hi, target, comp = 0, [m], [x]
    elif cond.startswith("f_"):
        kind = "complex" if cond == "f_dcomplex" else "simple"
        m, x, objs = prod_display(rng, kind)
        if cond == "f_csimple":      # second object (robot/scarf with the competitor feature): only true message is x
            hi, target, comp = 1, [x], []
        elif cond == "f_dsimple":    # third object: both of its (unique) features are equally good messages
            hi, target, comp = 2, list(objs[2]), []
            target = [f for f in target if f in MSGS]
        else:                        # third object of complex display: x' is unique, m is shared with the trigger
            hi = 2
            xprime = [f for f in objs[2] if f != m][0]
            target, comp = [xprime], [m]
    elif cond == "unamb":
        e = rng.choice(MSGS)
        hl = mk(e, inexpressible_for(e))
        pool = [(k, a) for k in KINDS for a in ACCS if not true_of(e, (k, a))]
        objs = [hl] + rng.sample(pool, 2)
        hi, target, comp = 0, [e], []
    else:  # amb
        k, a = rng.choice(EXPR_KINDS), rng.choice(EXPR_ACCS)
        pool = [(kk, aa) for kk in KINDS for aa in ACCS if kk != k and aa != a]
        objs = [(k, a)] + rng.sample(pool, 2)
        pair = [k, a]
        rng.shuffle(pair)
        hi, target, comp = 0, [pair[0]], [pair[1]]
    order = list(range(3))
    rng.shuffle(order)
    shown = [objs[i] for i in order]
    hi_pos = order.index(hi)
    msgs = list(MSGS)
    rng.shuffle(msgs)
    return {"cond": cond, "analysis_cond": cond if cond in ("simple", "complex", "unamb", "amb") else "filler",
            "objects": [list(o) for o in shown], "highlight": hi_pos, "messages": msgs,
            "target": target, "competitor": comp}


COMP_PLAN = [("simple", 12), ("complex", 12), ("f_csimple", 6), ("f_dsimple", 6), ("f_ccomplex", 12), ("unamb", 9), ("amb", 9)]
PROD_PLAN = [("simple", 12), ("complex", 12), ("f_csimple", 6), ("f_dsimple", 6), ("f_dcomplex", 12), ("unamb", 9), ("amb", 9)]


def make_trials(study, rng):
    plan, gen = (COMP_PLAN, comp_trial) if study == "exp1" else (PROD_PLAN, prod_trial)
    trials = [gen(rng, c) for c, k in plan for _ in range(k)]
    rng.shuffle(trials)
    for i, t in enumerate(trials, 1):
        t["trial"] = i
        t["phase"] = "main"
    if study == "exp1":   # four speaker trials first: three unambiguous, one ambiguous (production format)
        pr = [prod_trial(rng, c) for c in ("unamb", "unamb", "unamb", "amb")]
        rng.shuffle(pr)
        for i, t in enumerate(pr, 1):
            t["trial"] = i
            t["phase"] = "speaker"
        trials = pr + trials
    return trials


# ---------------------------------------------------------------- prompts
def objects_text(t):
    lines = []
    for i, o in enumerate(t["objects"]):
        mark = "  <-- highlighted with a yellow rectangle" if t.get("highlight") == i else ""
        lines.append("  Object %s: %s%s" % (LETTERS[i], describe(o), mark))
    return "\n".join(lines)


def trial_text(t):
    if "message" in t:
        return ("The three objects:\n%s\nThe message you received: %s\nWhich object do you pick? Reply with the letter A, B or C only."
                % (objects_text(t), t["message"]))
    msgs = "\n".join("  %d. %s" % (i + 1, m) for i, m in enumerate(t["messages"]))
    return ("The three objects:\n%s\nThe messages you can send:\n%s\nWhich message do you send? Reply with its number (1-4) only."
            % (objects_text(t), msgs))


def history_line(t, choice_text):
    if "message" in t:
        return "Trial %d (%s): objects %s; message %s; you picked %s." % (
            t["trial"], t["phase"], "; ".join("%s=%s" % (LETTERS[i], describe(o)) for i, o in enumerate(t["objects"])), t["message"], choice_text)
    return "Trial %d (%s): objects %s; highlighted %s; you sent %s." % (
        t["trial"], t["phase"], "; ".join("%s=%s" % (LETTERS[i], describe(o)) for i, o in enumerate(t["objects"])), LETTERS[t["highlight"]], choice_text)


def intro_for(study, t):
    if t["phase"] == "speaker":
        return MAT["instructions"]["exp1_speaker"]
    return MAT["instructions"]["exp1_comprehension" if study == "exp1" else "exp2_production"]


def build_prompt(study, condition=None, trial=None, history=None, seed=0):
    """Prompt for one trial of one participant. Without `trial`, a sample trial of `condition` is built
    (condition: simple | complex | unamb | amb)."""
    if trial is None:
        rng = random.Random(seed)
        trial = (comp_trial if study == "exp1" else prod_trial)(rng, condition or "simple")
        trial.update(trial=1, phase="main")
    parts = [intro_for(study, trial)]
    if history:
        parts.append("Your earlier trials in this session:\n" + "\n".join(history))
    parts.append("Trial %d.\n%s" % (trial["trial"], trial_text(trial)))
    return "\n\n".join(parts)


def parse_reply(study, condition, reply, trial=None):
    """Return {'choice': index or None, 'raw': reply}. Comprehension/speaker display letters map to indexes."""
    text = (reply or "").strip()
    if trial is None:
        m = re.search(r"\b([ABC]|[1-4])\b", text)
        return {"choice": m.group(1) if m else None, "raw": text}
    if "message" in trial:
        m = re.search(r"\b([ABC])\b", text) or re.fullmatch(r"\s*([abc])\W*", text)
        return {"choice": LETTERS.index(m.group(1).upper()) if m else None, "raw": text}
    m = re.search(r"\b([1-4])\b", text)
    return {"choice": int(m.group(1)) - 1 if m else None, "raw": text}


def score(trial, choice):
    """Role of the choice: target | competitor | distractor | None (no valid reply)."""
    if choice is None:
        return None
    if "message" in trial:
        return trial["roles"][choice]
    msg = trial["messages"][choice]
    if msg in trial["target"]:
        return "target"
    if msg in trial["competitor"]:
        return "competitor"
    return "distractor"


def run_participant(llm, study, rng):
    trials = make_trials(study, rng)
    history, rows = [], []
    for t in trials:
        reply = llm(build_prompt(study, None, trial=t, history=history))
        # speaker trials are in production format even in Experiment 1
        p = parse_reply(study, None, reply, trial=t)
        role = score(t, p["choice"])
        if "message" in t:
            ctext = "object " + LETTERS[p["choice"]] if p["choice"] is not None else "no valid answer"
        else:
            ctext = repr(t["messages"][p["choice"]]) if p["choice"] is not None else "no valid answer"
        history.append(history_line(t, ctext))
        rows.append({"trial": t["trial"], "phase": t["phase"], "cond": t["cond"], "analysis_cond": t["analysis_cond"],
                     "objects": t["objects"], "stimulus": t.get("message") or t["messages"],
                     "choice": p["choice"], "role": role, "reply": p["raw"][:200]})
    return rows


def run_sessions(llm, seed, n, arms=None):
    sessions = []
    for study in (arms or STUDIES):
        if study not in STUDIES:
            continue
        for i in range(n):
            rng = random.Random("%s-%s-%d" % (seed, study, i))
            sessions.append({"study": study, "arm": study, "condition": "within-participant: all trial types",
                             "participant": i, "trials": run_participant(llm, study, rng)})
    return sessions


def smoke_test():
    calls = []

    def stub(prompt, *a, **k):  # constant reply; exercises generation, parsing and scoring only
        calls.append(prompt)
        return "A" if "Reply with the letter" in prompt else "1"
    sessions = run_sessions(stub, 1, 2)
    assert len(sessions) == 4
    for s in sessions:
        main = [t for t in s["trials"] if t["phase"] == "main"]
        assert len(main) == 66, len(main)
        from collections import Counter
        c = Counter(t["cond"] for t in main)
        assert c["simple"] == 12 and c["complex"] == 12 and sum(v for k, v in c.items() if k.startswith("f_")) == 24, c
        assert c["unamb"] + c["amb"] == 18, c
        if s["study"] == "exp1":
            assert sum(1 for t in s["trials"] if t["phase"] == "speaker") == 4
        assert all(t["role"] is not None for t in s["trials"])
    assert "Object A" in build_prompt("exp1", "simple") and "Object A" in build_prompt("exp2", "complex")
    assert len(calls) == 2 * (70 + 66)
    import importlib.util
    spec = importlib.util.spec_from_file_location("hs_eval", HERE.parent / "evaluation" / "evaluation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.evaluate(sessions)
    assert "studies" in out and out["model_comparison"]["status"] == "not_ready"
    print("smoke test ok: %d prompts, %d sessions" % (len(calls), len(sessions)))


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
    else:
        print(__doc__)
