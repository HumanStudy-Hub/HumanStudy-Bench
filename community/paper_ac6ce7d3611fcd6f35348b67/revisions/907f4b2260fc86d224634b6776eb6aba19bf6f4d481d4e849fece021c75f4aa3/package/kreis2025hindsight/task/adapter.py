#!/usr/bin/env python3
"""Offline adapter for Kreis et al. (2025), study s1: hindsight bias and knowledge updating (sugar content).

Every participant is an AI participant (llm(prompt) -> str is injected). A participant is a short
conversation: the OJ task, then the information phase and the ROJ task, then the NJ task. Each call
receives the participant's whole transcript so far (what a person would remember of the session).

    python task/adapter.py --smoke-test
"""
import itertools
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUDY = "s1"
ARMS = {"control": "c_control", "feedback": "c_feedback", "domain": "c_domain", "irrelevant": "c_irrelevant"}
COND_TO_ARM = {v: k for k, v in ARMS.items()}
FRAME = ("You are taking part in an online study as a participant. Answer from your own knowledge and "
         "from what you see in this study. For each numbered question reply with one line in the form "
         "`<question number>: <number of grams>` and nothing else.\n")
_MAT = None


class NotReady(RuntimeError):
    """A required printed material is missing; the paper does not contain it."""


def materials():
    global _MAT
    if _MAT is None:
        _MAT = json.loads((ROOT / "materials/materials.json").read_text(encoding="utf-8"))["studies"][STUDY]
    return _MAT


def _arm(condition):
    return COND_TO_ARM.get(condition, condition)


def _phrase(name):
    m = re.match(r"^1 (\w)(.*)$", name)
    return ("an " if m.group(1) in "aeiou" else "a ") + m.group(1) + m.group(2) if m else name


def question(item, kind):
    base = "How many grams of sugar does %s that weighs %d grams contain?" % (_phrase(item["name"]), item["size"])
    return base + (" What was your ORIGINAL answer?" if kind == "roj" else "")


def statement(item):
    s = "%s that weighs %d grams contains %d grams of sugar." % (_phrase(item["name"]), item["size"], item["sugar"])
    return s[0].upper() + s[1:]


def plan_participant(arm, k, seed):
    """Sets and item orders for participant k of an arm (rotation over the 6 role assignments of sets A, B, C)."""
    rng = random.Random("%s-%s-%d" % (seed, arm, k))
    oj_set, dom_set, nj_set = list(itertools.permutations("ABC"))[k % 6]
    items = materials()["items"]
    by = lambda s: [i for i in items if i["set"] == s]
    oj_items = by(oj_set)[:]
    rng.shuffle(oj_items)
    nj_items = by(nj_set)[:]
    rng.shuffle(nj_items)
    info_set = dom_set if arm == "domain" else oj_set
    info_items = by(info_set)[:]
    rng.shuffle(info_items)
    return {"oj_set": oj_set, "info_set": info_set, "nj_set": nj_set, "oj": oj_items, "nj": nj_items, "info": info_items}


def questions_block(title, items, kind):
    return "%s\n%s\n" % (title, "\n".join("%d. %s" % (n + 1, question(i, kind)) for n, i in enumerate(items)))


def info_block(arm, plan):
    m = materials()
    if arm == "control":
        if not m.get("control_filler_text"):
            raise NotReady("control arm: the filler text is not printed in the paper (issue iss_filler); put it in materials.json control_filler_text")
        return "Phase 2 (shown for 256 seconds):\n%s\n" % m["control_filler_text"]
    if arm == "irrelevant":
        vals = sorted(i["sugar"] for i in plan["info"])
        cities = m["cities"]
        lines = []
        for c, v in zip(cities, vals):
            lines.append("Basel, a major city in Switzerland, is located at longitude %d." % v if c["city"] == "Basel"
                         else "%s is located at longitude %d." % (c["city"], v))
        order = random.Random(len(lines) + sum(vals)).sample(lines, len(lines))
        return "Phase 2. Each of these statements was shown for 8 seconds, one after another:\n%s\n" % "\n".join(order)
    return "Phase 2. Each of these statements was shown for 8 seconds, one after another:\n%s\n" % "\n".join(statement(i) for i in plan["info"])


def build_prompt(study, condition, phase="oj", k=0, seed=0, transcript=""):
    """Prompt for one call of one participant. phase: oj | roj | nj. transcript: earlier blocks and replies."""
    arm = _arm(condition)
    plan = plan_participant(arm, k, seed)
    if phase == "oj":
        return FRAME + "\n" + questions_block("Phase 1. Please estimate the sugar content of these food items.", plan["oj"], "oj")
    if phase == "roj":
        return FRAME + "\n" + transcript + "\n" + info_block(arm, plan) + "\n" + questions_block(
            "Phase 3. Please recall the answers you gave in Phase 1 for the same items, in the same order.", plan["oj"], "roj")
    if phase == "nj":
        return FRAME + "\n" + transcript + "\n" + questions_block("Phase 4. Please estimate the sugar content of these new food items.", plan["nj"], "oj")
    raise ValueError(phase)


def parse_reply(study, condition, reply, n=32):
    """List of n floats (None where the reply gave no number)."""
    out = [None] * n
    for m in re.finditer(r"^\s*\W*(\d+)\s*[:.)=\-]\s*(-?\d+(?:[.,]\d+)?)", reply or "", re.M):
        i = int(m.group(1)) - 1
        if 0 <= i < n and out[i] is None:
            out[i] = float(m.group(2).replace(",", "."))
    if all(v is None for v in out):
        nums = re.findall(r"-?\d+(?:[.,]\d+)?", reply or "")
        for i, v in enumerate(nums[:n]):
            out[i] = float(v.replace(",", "."))
    return out


def run_participant(llm, arm, k, seed):
    cond = ARMS[arm]
    plan = plan_participant(arm, k, seed)
    p1 = build_prompt(STUDY, cond, "oj", k, seed)
    r1 = llm(p1)
    oj = parse_reply(STUDY, cond, r1)
    t1 = p1[len(FRAME) + 1:] + "\nYour answers:\n" + r1.strip() + "\n"
    p3 = build_prompt(STUDY, cond, "roj", k, seed, t1)
    r3 = llm(p3)
    roj = parse_reply(STUDY, cond, r3)
    t3 = t1 + "\n" + p3[len(FRAME) + 1 + len(t1) + 1:] + "\nYour answers:\n" + r3.strip() + "\n"
    p4 = build_prompt(STUDY, cond, "nj", k, seed, t3)
    r4 = llm(p4)
    nj = parse_reply(STUDY, cond, r4)
    rec = lambda items, vals, extra=None: [dict(item=i["id"], name=i["name"], weight=i["size"], actual=i["sugar"], subcategory=i["subcategory"], estimate=v) for i, v in zip(items, vals)]
    return {"participant": "%s-%d" % (arm, k + 1), "condition": cond, "arm": arm, "seed": seed,
            "sets": {"oj": plan["oj_set"], "info": plan["info_set"], "nj": plan["nj_set"]},
            "oj": rec(plan["oj"], oj), "roj": rec(plan["oj"], roj), "nj": rec(plan["nj"], nj),
            "raw": {"oj": r1, "roj": r3, "nj": r4}}


def run_sessions(llm, seed, n, arms=None):
    """n AI participants per arm. arms: names (control, feedback, domain, irrelevant) or condition ids; None = all four."""
    chosen = [_arm(a) for a in (arms or ARMS)]
    sessions = []
    for arm in chosen:
        for k in range(n):
            sessions.append(run_participant(llm, arm, k, seed))
    return sessions


def smoke_test():
    def fake(prompt):
        qs = re.findall(r"^(\d+)\. How many grams", prompt.split("Phase 4")[-1] if "Phase 4" in prompt else prompt.split("Phase 3")[-1] if "Phase 3" in prompt else prompt, re.M)
        return "\n".join("%s: %d" % (q, 3 + int(q) % 7) for q in qs)
    sessions = run_sessions(fake, 7, 2, ["feedback", "domain", "irrelevant"])
    assert len(sessions) == 6 and all(len(s["oj"]) == len(s["roj"]) == len(s["nj"]) == 32 for s in sessions)
    assert all(None not in [t["estimate"] for t in s["oj"] + s["roj"] + s["nj"]] for s in sessions)
    p = build_prompt(STUDY, "c_feedback", "roj", 0, 7, "earlier\n")
    assert "kiwi" in p or "grams of sugar" in p and "ORIGINAL" in p
    assert not re.search(r"hindsight|knowledge updating|transfer", p, re.I)
    irr = build_prompt(STUDY, "c_irrelevant", "roj", 0, 7, "")
    assert "longitude" in irr and "contains" not in irr.split("Phase 3")[0].split("Phase 2")[1]
    try:
        run_sessions(fake, 7, 1, ["control"])
        raise SystemExit("control arm should be not ready while the filler text is missing")
    except NotReady:
        pass
    sys.path.insert(0, str(ROOT / "evaluation"))
    import evaluation
    out = evaluate_smoke(evaluation, sessions)
    print("smoke test ok:", out)


def evaluate_smoke(evaluation, sessions):
    res = evaluation.evaluate(sessions)
    assert "groups" in res and "bayes_factors" in res and res["bayes_factors"]["status"] == "not_ready"
    return sorted(res["groups"])


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
    else:
        print(__doc__)
