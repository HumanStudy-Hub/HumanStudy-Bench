#!/usr/bin/env python3
"""Offline adapter for Many Labs 2 (Klein et al.).

run_sessions(llm, seed, n, arms=None): n AI participants per slate (arms = slate ids, default both).
Every participant is a separate sequence of llm(prompt) calls; the adapter never answers for a participant.
A participant sees only its own earlier pages and answers. Effects run in a fresh random order per participant and
each effect's version is drawn at random per participant. Effects whose stimulus is a picture (materials.json imageRequired) are run only when the injected llm has the attribute
accepts_images = True and takes llm(prompt, images=[paths]); otherwise they are not run and are reported as such. The
runner's text-only CLI and API participants do not set it.

    python adapter.py --smoke-test     offline structure test with a stub (not a participant, produces no data)
"""
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MATERIALS = json.loads((HERE.parent / "materials/materials.json").read_text(encoding="utf-8"))
TASK = json.loads((HERE / "task.json").read_text(encoding="utf-8"))

INSTRUCTIONS = (
    "You are taking part in an online survey on a computer, as a participant who answers for yourself. "
    "The survey has several unrelated tasks. You see one page at a time and cannot go back. "
    "Answer each page as you would honestly answer it."
)
FORMAT = (
    "How to reply: if the page lists numbered options, reply with the number of the option you choose and nothing else. "
    "If the page asks for a number or percentage, reply with the number(s) only. "
    "If the page asks for text, reply with your text only. If the page asks for nothing, reply: continue"
)
PIPE = re.compile(r"\$\{[^}]*\}")


def build_prompt(history, page, goal=None):
    # Picture files of a page are passed to a vision-capable llm separately; the text of the page is unchanged.
    """Prompt for one participant at one page. history = [(page_text, answer_text), ...] of this participant only."""
    text = PIPE.sub(goal or "[your answer on the earlier page]", page["text"])
    seen = "".join("\n--- Earlier page ---\n%s\n--- Your answer ---\n%s\n" % (p, a) for p, a in history)
    return "%s\n%s\n%s\n--- Current page ---\n%s\n--- Your answer ---\n" % (INSTRUCTIONS, FORMAT, seen, text)


def parse_reply(page, reply):
    """Return {'raw', 'code', 'number'} for one page. code = chosen option code if the page lists options."""
    raw = (reply or "").strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", raw)
    out = {"raw": raw, "code": None, "number": float(nums[0]) if nums else None}
    codes = [o["code"] for o in page.get("options", [])]
    if codes and nums:
        for n in nums:
            if n.lstrip("-").isdigit() and int(n) in codes:
                out["code"] = int(n)
                break
    if codes and out["code"] is None:
        low = raw.lower()
        for o in page["options"]:
            if o["label"] and o["label"].lower() in low:
                out["code"] = o["code"]
                break
    return out


def _run_pages(llm, pages, history, goal_holder):
    answers = {}
    for page in pages:
        prompt = build_prompt(history, page, goal_holder.get("goal"))
        reply = llm(prompt, images=[str(HERE.parent / "materials/stimuli" / f) for f in page["images"]]) if page.get("images") else llm(prompt)
        parsed = parse_reply(page, reply)
        answers[page["id"]] = parsed
        history.append((page["text"], parsed["raw"]))
        if not page.get("options") and parsed["raw"] and "goal" not in goal_holder and page["id"].endswith(".1") and "goal" in page["text"].lower():
            goal_holder["goal"] = parsed["raw"]
    return answers


def _runnable(eff, llm):
    return not eff["blocked"] and (not eff.get("imageRequired") or getattr(llm, "accepts_images", False))


def run_participant(llm, slate, rng, participant):
    study = MATERIALS["studies"][slate]
    history, sessions = [], []
    _run_pages(llm, study.get("intro", []), history, {})
    effects = [k for k, v in study["effects"].items() if _runnable(v, llm)]
    rng.shuffle(effects)
    for position, k in enumerate(effects, 1):
        eff = study["effects"][k]
        cond = rng.choice(sorted(eff["conditions"]))
        answers = _run_pages(llm, eff["conditions"][cond]["pages"], history, {})
        sessions.append({"arm": slate, "participant": participant, "effect": k, "condition": cond, "position": position, "answers": answers})
    answers = _run_pages(llm, study.get("after", []), history, {})
    sessions.append({"arm": slate, "participant": participant, "effect": "_after", "condition": "all", "position": len(effects) + 1, "answers": answers})
    return sessions


def run_sessions(llm, seed, n, arms=None):
    """n participants per arm (slate). llm(prompt) -> str is injected. Returns a list of session dicts."""
    rng = random.Random(seed)
    out = []
    for slate in (arms or [a["id"] for a in TASK["arms"]]):
        for p in range(n):
            out.extend(run_participant(llm, slate, random.Random(rng.random()), p))
        for k, v in MATERIALS["studies"][slate]["effects"].items():
            if not _runnable(v, llm):
                why = v["blocked"] or "picture stimulus; this participant is text-only (%s)" % v["imageRequired"]
                out.append({"arm": slate, "participant": None, "effect": k, "condition": None, "status": "not_run", "reason": why})
    return out


def _smoke_test():
    # A stub that returns a fixed string. It only proves the plumbing; it is not a participant and its output is not data.
    sessions = run_sessions(lambda prompt: "1", seed=1, n=1)
    runs = [s for s in sessions if s.get("status") != "not_run"]
    assert runs and all("arm" in s and "condition" in s for s in sessions)
    assert any(s.get("status") == "not_run" for s in sessions)
    p = build_prompt([("earlier", "2")], {"text": "Question?"})
    assert "Question?" in p and "earlier" in p
    assert parse_reply({"options": [{"code": 1, "label": "Yes"}, {"code": 2, "label": "No"}]}, "2")["code"] == 2
    print("smoke test ok: %d sessions" % len(sessions))


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        _smoke_test()
    else:
        print(__doc__)
