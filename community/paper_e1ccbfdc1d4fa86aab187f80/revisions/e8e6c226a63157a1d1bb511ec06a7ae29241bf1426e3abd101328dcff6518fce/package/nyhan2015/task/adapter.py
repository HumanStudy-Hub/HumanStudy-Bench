#!/usr/bin/env python3
"""Offline replay adapter for Nyhan & Reifler, "Displacing misinformation about events".

Each AI participant makes two model calls, as the survey had two stages separated by the dispatches:
  1. the pre-treatment 'crooked officials' question (the blocking question, asked before the experiment);
  2. the survey pages as the respondent saw them: that question with their own answer, the scenario
     introduction, the dispatches of their condition (one per page, in order), then the outcome questions.
The delay tasks and the recall questions are not administered (see task.json adaptations).

    python task/adapter.py --smoke-test        offline format checks, no model call
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MATERIALS = json.loads((HERE.parent / "materials" / "materials.json").read_text(encoding="utf-8"))
STUDY_ID = "S1"
CONDITIONS = ["control", "innuendo", "denial", "causal"]
PAGE_BREAK = "--- page break ---"
FRAME = ("You are a respondent in an online survey. The survey pages you saw are shown below in order. "
         "Answer as yourself, as you would in this survey.")


def _study(study_id=None):
    return MATERIALS["studies"][study_id or STUDY_ID]


def _cond(condition_id):
    c = str(condition_id)
    c = c[2:] if c.startswith("c-") else c
    if c not in CONDITIONS:
        raise ValueError("unknown condition %r; expected one of %s" % (condition_id, CONDITIONS))
    return c


def _norm(text):
    return re.sub(r"\s+", " ", re.sub(r"\[[^\]]*\]", " ", str(text))).strip(" -\t.").lower()


def _options_block(options):
    return "\n".join("- " + o for o in options)


def build_pre_prompt(study_id=None, condition_id=None):
    """Call 1: the pre-treatment question, exactly as printed."""
    m = _study(study_id)["pre_treatment"]
    return "\n\n".join([
        FRAME,
        m["text"] + "\n" + _options_block(m["options"]),
        "Reply with exactly one of the options above, copying its text and nothing else.",
    ])


def build_prompt(study_id, condition_id, crooked=None):
    """Call 2: everything one participant reads after the pre-treatment question, in the survey's order."""
    st = _study(study_id)
    cond = _cond(condition_id)
    pre = st["pre_treatment"]
    pages = [pre["text"] + "\n" + _options_block(pre["options"]) + ("\nYour answer: " + crooked if crooked else "")]
    pages.append(st["intro"]["text"])
    by_id = {d["id"]: d for d in st["dispatches"]}
    for did in st["by_condition"][cond]:
        d = by_id[did]
        pages.append(d["timestamp"] + "\n" + d["text"])
    qs = [q for q in st["questions"] if cond in q["conditions"]]
    q_pages = [st["question_intro"]["text"]]
    for i, q in enumerate(qs, 1):
        q_pages.append("Question %d. %s\n%s" % (i, q["text"], _options_block([o["label"] for o in q["options"]])))
    body = ("\n" + PAGE_BREAK + "\n").join(pages + q_pages)
    fmt = "Reply with one line per question in the form 'Q1: <option text>', copying the text of the option you choose."
    return FRAME + "\n\n" + body + "\n\n" + fmt


def parse_crooked(study_id, reply):
    options = _study(study_id)["pre_treatment"]["options"]
    r = _norm(reply)
    for o in options:
        if _norm(o) == r:
            return o
    hits = [o for o in options if _norm(o) in r]
    return max(hits, key=len) if hits else None


def parse_reply(study_id, condition_id, reply):
    """Map the reply to {variable: code or None} using the printed response options."""
    st = _study(study_id)
    cond = _cond(condition_id)
    qs = [q for q in st["questions"] if cond in q["conditions"]]
    lines = {}
    for line in str(reply).splitlines():
        m = re.match(r"\s*\**Q(\d+)\**\s*[:.)-]\s*(.+)", line, re.I)
        if m:
            lines[int(m.group(1))] = m.group(2).strip()
    out = {}
    for i, q in enumerate(qs, 1):
        ans = _norm(lines.get(i, ""))
        code = None
        for o in q["options"]:
            if _norm(o["label"]) == ans:
                code = o["code"]
        if code is None and ans:
            hits = [o for o in q["options"] if ans.startswith(_norm(o["label"]))]
            if hits:
                code = max(hits, key=lambda o: len(o["label"]))["code"]
        out[q["variable"]] = code
    return out


def run_sessions(llm, seed, n, arms=None):
    """n AI participants per arm (the respondent is the sampling unit); one condition per arm."""
    arms = list(arms) if arms else list(CONDITIONS)
    sessions = []
    for arm in arms:
        cond = _cond(arm)
        for i in range(int(n)):
            raw1 = llm(build_pre_prompt(STUDY_ID, cond))
            crooked = parse_crooked(STUDY_ID, raw1)
            raw2 = llm(build_prompt(STUDY_ID, cond, crooked))
            sessions.append({"study": STUDY_ID, "condition": cond, "arm": cond, "participant": i, "seed": seed,
                             "crooked": crooked, "responses": parse_reply(STUDY_ID, cond, raw2),
                             "raw": [raw1, raw2]})
    return sessions


def smoke_test():
    st = _study()
    expect = {"control": 3, "innuendo": 10, "denial": 11, "causal": 11}
    by_id = {d["id"]: d for d in st["dispatches"]}
    for cond in CONDITIONS:
        ids = st["by_condition"][cond]
        assert len(ids) == expect[cond], (cond, len(ids))
        p = build_prompt(STUDY_ID, cond, st["pre_treatment"]["options"][0])
        for did in ids:
            assert by_id[did]["text"] in p, (cond, did)
        for did in by_id:
            if did not in ids and not any(by_id[did]["text"] in by_id[k]["text"] for k in ids):  # d11 contains d10
                assert by_id[did]["text"] not in p, (cond, did, "shown to the wrong condition")
        assert ("investigation for bribery?" in p) == (cond != "control"), cond
        qs = [q for q in st["questions"] if cond in q["conditions"]]
        fixture = "\n".join("Q%d: %s" % (i, q["options"][0]["label"]) for i, q in enumerate(qs, 1))
        parsed = parse_reply(STUDY_ID, cond, fixture)  # parser format check only, not a response
        assert all(v is not None for v in parsed.values()) and len(parsed) == len(qs), (cond, parsed)
    assert parse_crooked(STUDY_ID, "Quite a few are crooked") == "Quite a few are crooked"
    sys.path.insert(0, str(HERE.parent / "evaluation"))
    import evaluation
    assert evaluation.evaluate([]).get("status") == "not_ready"
    print("smoke test ok: materials, prompts, parser and evaluation load offline")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-test", action="store_true")
    a = ap.parse_args()
    if a.smoke_test:
        smoke_test()
    else:
        ap.print_help()
