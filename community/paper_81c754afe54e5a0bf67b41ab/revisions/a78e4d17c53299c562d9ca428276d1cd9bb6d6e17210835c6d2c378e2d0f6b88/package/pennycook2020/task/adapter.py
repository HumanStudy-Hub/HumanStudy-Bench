#!/usr/bin/env python3
"""Offline adapter for Pennycook et al. (2020), Studies 1 and 2.

Every simulated participant is one call to the injected `llm(prompt) -> str`. Nothing here answers
for a participant. Standard library only.

  build_prompt(study, condition) -> str     one participant's whole session as one prompt
  parse_reply(study, condition, reply)      reply text -> {"valid", "responses", "pretest", "errors"}
  run_sessions(llm, seed, n, arms=None)     n AI participants per arm; arms are "study/condition" strings
  python task/adapter.py --smoke-test       offline structure check with a constant stub (not a simulation)

Arms: study1/accuracy, study1/sharing, study2/control, study2/treatment. The treatment arm shows one of four
pretest headlines first (text, question and framing transcribed from sources/study2_materials). The paper
does not state how the headline was chosen (iss_s2_pretest_assign); the replay draws one at random per participant.
It is skipped if the pretest materials are ever missing.
"""
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MATERIALS = json.loads((HERE.parent / "materials" / "materials.json").read_text(encoding="utf-8"))
HEADLINES = MATERIALS["headlines"]
BY_ID = {h["id"]: h for h in HEADLINES}
# Neutral codes H01..H30, assigned by a fixed hash of the item id, so a code does not reveal veracity.
_ORDERED = sorted(HEADLINES, key=lambda h: hashlib.sha1(h["id"].encode()).hexdigest())
CODE = {h["id"]: "H%02d" % (i + 1) for i, h in enumerate(_ORDERED)}
ID_OF = {v: k for k, v in CODE.items()}
ARMS = ["study1/accuracy", "study1/sharing", "study2/control", "study2/treatment"]


class NotReady(RuntimeError):
    pass


def arm_status():
    """{arm: None when runnable, else the reason}."""
    out = {a: None for a in ARMS}
    pre = MATERIALS["studies"]["study2"]["conditions"]["treatment"]["pretest"]
    missing = [k for k in ("headlines", "question", "framing") if not pre.get(k)]
    if missing:
        out["study2/treatment"] = "pretest %s not printed in the paper (issues %s)" % (", ".join(missing), ", ".join(pre["issue_ids"]))
    return out


def _spec(study, condition):
    try:
        return MATERIALS["studies"][study]["conditions"][condition]
    except KeyError:
        raise ValueError("unknown arm %s/%s" % (study, condition))


def _rng(*parts):
    return random.Random("|".join(str(p) for p in parts))


def _options(study, seed):
    spec = MATERIALS["studies"][study]
    if study == "study1":
        yn = ["yes", "no"] if _rng(study, "opt", seed).random() < 0.5 else ["no", "yes"]
        return " / ".join(yn)
    return spec["scale_text"]


def _build(study, condition, seed):
    reason = arm_status()["%s/%s" % (study, condition)]
    if reason:
        raise NotReady(reason)
    spec = _spec(study, condition)
    frame = MATERIALS["replay_frame"]
    order = [h["id"] for h in HEADLINES]
    _rng(study, condition, "order", seed).shuffle(order)
    question = spec["question"]
    options = _options(study, seed)
    lines = [frame["intro"].replace("{n}", str(len(order)))]
    keys = []
    pre = spec.get("pretest")
    if pre:
        headline = _rng(study, condition, "pretest", seed).choice(pre["headlines"])
        pre_options = " / ".join(pre["options"])
        lines += ["", pre["framing"], "", "[P] " + headline["text"], pre["question"], "Options: " + pre_options]
        keys.append("pretest")
    lines += ["", frame["task_header"]]
    for hid in order:
        lines += ["", "[%s] %s" % (CODE[hid], BY_ID[hid]["text"]), question, "Options: " + options]
        keys.append(CODE[hid])
    lines += ["", frame["reply_format"].replace("{keys}", ", ".join(keys))]
    return "\n".join(lines), order


def build_prompt(study, condition, seed=0):
    return _build(study, condition, seed)[0]


def _answer(study, value):
    if study == "study1":
        v = str(value).strip().lower().rstrip(".")
        return {"yes": 1, "no": 0}.get(v)
    try:
        v = int(str(value).strip())
    except ValueError:
        return None
    return v if 1 <= v <= 6 else None


def parse_reply(study, condition, reply):
    errors, responses, pretest = [], {}, None
    blob = None
    match = re.search(r"\{.*\}", reply or "", re.S)
    if match:
        try:
            blob = json.loads(match.group(0))
        except ValueError:
            blob = None
    if not isinstance(blob, dict):
        return {"valid": False, "responses": {}, "pretest": None, "errors": ["no JSON object in reply"]}
    blob = {str(k).strip().strip("[]").upper() if k != "pretest" else k: v for k, v in blob.items()}
    for code, hid in ID_OF.items():
        value = _answer(study, blob.get(code)) if code in blob else None
        if value is None:
            errors.append("missing or invalid answer for " + code)
        else:
            responses[hid] = value
    if _spec(study, condition).get("pretest"):
        pretest = blob.get("pretest")
        pretest = {"yes": 1, "no": 0}.get(str(pretest).strip().lower().rstrip(".")) if pretest is not None else None
        if pretest is None:
            errors.append("missing or invalid pretest answer")
    return {"valid": not errors, "responses": responses, "pretest": pretest, "errors": errors}


def run_sessions(llm, seed, n, arms=None):
    status = arm_status()
    if arms is None:
        arms = [a for a in ARMS if status[a] is None]
    sessions = []
    for arm in arms:
        if status.get(arm):
            raise NotReady(arm + ": " + status[arm])
        study, condition = arm.split("/")
        for i in range(int(n)):
            pseed = "%s-%s-%d" % (seed, arm, i)
            prompt, order = _build(study, condition, pseed)
            reply = llm(prompt)
            parsed = parse_reply(study, condition, reply)
            sessions.append({"study": study, "condition": condition, "arm": arm, "participant": "%s#%d" % (arm, i),
                             "seed": pseed, "order": order, "reply": reply, **parsed})
    return sessions


def _smoke():
    status = arm_status()
    assert len(HEADLINES) == 30 and sum(h["veracity"] == "true" for h in HEADLINES) == 15
    assert len(set(CODE.values())) == 30
    for arm in ARMS:
        study, condition = arm.split("/")
        if status[arm]:
            try:
                build_prompt(study, condition)
            except NotReady:
                continue
            raise AssertionError(arm + " should not build while its materials are missing")
        prompt = build_prompt(study, condition)
        assert _spec(study, condition)["question"] in prompt, arm
        assert all("[%s]" % c in prompt for c in CODE.values()), arm
        assert "False-" not in prompt and "True-" not in prompt, "item ids leak veracity"

    def stub(prompt):  # constant reply, only to exercise the parser and session bookkeeping
        out = {}
        for key, rest in re.findall(r"\[(H\d\d|P)\][^\n]*\n[^\n]*\nOptions: ([^\n]*)", prompt):
            out["pretest" if key == "P" else key] = "no" if rest in ("yes / no", "no / yes") else "3"
        return json.dumps(out)

    sessions = run_sessions(stub, 1, 2)
    assert sessions and all(s["valid"] for s in sessions), [s["errors"] for s in sessions if not s["valid"]][:2]
    print("smoke test ok: %d arms ready, %d skipped (%s), %d sessions parsed"
          % (sum(v is None for v in status.values()), sum(v is not None for v in status.values()),
             "; ".join("%s: %s" % kv for kv in status.items() if kv[1]), len(sessions)))


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        _smoke()
    else:
        print(__doc__)
