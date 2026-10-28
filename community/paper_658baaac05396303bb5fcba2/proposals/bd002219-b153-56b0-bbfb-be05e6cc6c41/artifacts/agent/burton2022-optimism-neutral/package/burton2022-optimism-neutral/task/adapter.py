#!/usr/bin/env python3
"""Offline adapter for Burton, Harris, Shah & Hahn: update method with valence-neutral life events.

Every original participant is an AI participant. One participant works through the whole session
turn by turn: each screen is one call to llm(prompt); the prompt holds only that participant's own
earlier screens and answers, so what they have seen and said so far is exactly what they read.

  build_prompt(study, condition, history=None, screen=None)  first screen (or a later one) for one participant
  parse_reply(study, condition, reply, screen=None)          number 0-100 (estimates) or 1-5 (valence)
  run_sessions(llm, seed, n, arms=None)                      n AI participants per study, in sequence

The 51 events and base rates (SI Table S1) are not in the paper, so run_sessions refuses to run until
materials/materials.json carries them (status "missing" is the blocking issue iss-stimuli).
`--smoke-test` checks the mechanics with a stub reply and the four quoted events; it is not a study run.
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUDIES = ("study1", "study2", "study3")
ORDERS = ("e1first", "ebrfirst")
ARMS = [f"{s}:{o}" for s in STUDIES for o in ORDERS]
# reported age distribution per study (paper p. 6): range, mean, sd
AGES = {"study1": (18, 81, 32.82, 11.52), "study2": (18, 67, 33.29, 10.31), "study3": (19, 73, 36.09, 12.06)}


def materials():
    return json.loads((ROOT / "materials/materials.json").read_text(encoding="utf-8"))


def stimuli(allow_partial=False):
    m = materials()["stimuli"]
    if m["status"] != "complete" and not allow_partial:
        raise RuntimeError("not_ready: the 51 life events and base rates (SI Table S1) are missing; "
                           "see issue iss-stimuli. Add them to materials/materials.json.")
    return m["events"]


def _screen_text(study, kind, event, br=None):
    """Screen wording. Only the valence item and scale are the paper's; the rest is implementation wording."""
    m = materials()
    w = m["screens"]
    ev = event["text"]
    if kind == "e1":
        return w["e1"].format(event=ev)
    if kind == "ebr":
        return w["ebr"].format(event=ev)
    if kind == "br":
        return w["br"].format(event=ev, br=br)
    if kind == "recall":
        return w["recall"].format(event=ev)
    if kind == "recall_fix":
        return w["recall_fix"].format(event=ev, br=br)
    if kind == "e2":
        return w["e2"].format(event=ev)
    if kind == "valence":
        return w["valence"].format(event=ev)
    raise ValueError(kind)


def persona(study, rng):
    lo, hi, mean, sd = AGES[study]
    age = None
    while age is None or not lo <= age <= hi:
        age = int(round(rng.gauss(mean, sd)))
    return f"You are a {age}-year-old adult living in the United Kingdom, taking part in an online survey on Prolific."


def build_prompt(study, condition, history=None, screen=None, who=None):
    """Prompt for one screen. history is the participant's own list of (screen text, answer) so far."""
    w = materials()["screens"]
    lines = [who or w["intro"]]
    for text, answer in (history or []):
        lines.append(f"[Earlier screen] {text}\n[Your answer] {answer}")
    lines.append(f"[Current screen] {screen or w['first_screen_placeholder']}")
    lines.append(w["answer_format"])
    return "\n\n".join(lines)


def parse_reply(study, condition, reply, screen=None):
    """First number in the reply; estimates are clipped to the 0-100 scale, valence must be 1-5."""
    mt = re.search(r"-?\d+(?:\.\d+)?", str(reply))
    if not mt:
        return None
    v = float(mt.group())
    if screen == "valence":
        v = int(round(v))
        return v if 1 <= v <= 5 else None
    return min(100.0, max(0.0, v))


def _ask(llm, study, condition, history, kind, event, who, br=None, retries=2):
    text = _screen_text(study, kind, event, br)
    for _ in range(retries + 1):
        reply = llm(build_prompt(study, condition, history, text, who))
        value = parse_reply(study, condition, reply, "valence" if kind == "valence" else kind)
        if value is not None:
            history.append((text, str(int(value)) if float(value).is_integer() else str(value)))
            return value
    history.append((text, "no valid answer"))
    return None


def run_participant(llm, study, order, events, rng, pid):
    """One AI participant, one study, screens in the order the paper describes."""
    who = persona(study, rng)
    history, trials = [], {}
    first = list(events)
    rng.shuffle(first)
    if study == "study1":
        part1 = ("e1", "ebr", "br", "recall", "e2")
    else:
        part1 = ("e1", "ebr", "br", "recall") + (("valence",) if study == "study3" else ())
    for ev in first:
        t = trials.setdefault(ev["id"], {"event": ev["id"], "BR": ev["br"]})
        seq = list(part1)
        if order == "ebrfirst":
            seq[0], seq[1] = "ebr", "e1"
        for kind in seq:
            if kind == "br":
                history.append((_screen_text(study, "br", ev, ev["br"]), "(read)"))
            elif kind == "recall":
                r = _ask(llm, study, order, history, "recall", ev, who)
                t["BR_recall"] = r
                if r is None or abs(r - ev["br"]) > 1e-9:
                    history.append((_screen_text(study, "recall_fix", ev, ev["br"]), str(ev["br"])))
            else:
                key = {"e1": "E1", "ebr": "eBR", "e2": "E2", "valence": "valence"}[kind]
                t[key] = _ask(llm, study, order, history, kind, ev, who)
    later = []
    if study == "study2":
        later = [("e2", "E2"), ("valence", "valence")]
    elif study == "study3":
        later = [("e2", "E2")]
    elif study == "study1":
        later = [("valence", "valence")]
    for kind, key in later:
        second = list(events)
        rng.shuffle(second)
        for ev in second:
            trials[ev["id"]][key] = _ask(llm, study, order, history, kind, ev, who)
    return {"participant": pid, "study": study, "condition": f"{study}:{order}", "arm": f"{study}:{order}",
            "order": order, "persona": who, "trials": list(trials.values())}


def run_sessions(llm, seed, n, arms=None, allow_partial=False):
    """n AI participants for each study in arms (default: all three), orders assigned at random per participant.

    llm(prompt) -> str is injected. Participants run one after another, each with their own history."""
    events = stimuli(allow_partial)
    studies = [a.split(":")[0] for a in (arms or STUDIES)]
    studies = [s for i, s in enumerate(studies) if s in STUDIES and s not in studies[:i]]
    out = []
    for study in studies:
        rng = random.Random(f"{seed}:{study}")
        orders = [ORDERS[k % 2] for k in range(n)]
        rng.shuffle(orders)
        for k, order in enumerate(orders):
            out.append(run_participant(llm, study, order, events, random.Random(rng.random()), f"{study}-{k + 1}"))
    return out


def _smoke():
    mats = materials()
    assert mats["stimuli"]["status"] in ("missing", "complete")
    prompt = build_prompt("study1", "e1first", [], _screen_text("study1", "e1", mats["stimuli"]["events"][0]))
    assert "[Current screen]" in prompt
    assert parse_reply("study1", "e1first", "I'd say 35%", "e1") == 35.0
    assert parse_reply("study1", "e1first", "4", "valence") == 4 and parse_reply("study1", "e1first", "9", "valence") is None
    try:
        run_sessions(lambda p: "50", 1, 1)
        refused = False
    except RuntimeError as exc:
        refused = str(exc).startswith("not_ready")
    if mats["stimuli"]["status"] != "complete":
        assert refused, "must refuse to run with missing stimuli"
    calls = []

    def stub(prompt):
        calls.append(prompt)
        return "3" if "feel about experiencing" in prompt.split("[Current screen]")[-1] else "40"

    for study in STUDIES:
        sess = run_sessions(stub, 7, 2, [study], allow_partial=True)
        assert len(sess) == 2 and {s["order"] for s in sess} == {"e1first", "ebrfirst"}, "orders not balanced"
        for s in sess:
            assert all(t["E1"] is not None and "valence" in t for t in s["trials"])
            if study == "study1":
                assert all("E2" in t for t in s["trials"])
    print("smoke test ok: mechanics only, no study run (%d stub calls)" % len(calls))


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        _smoke()
    else:
        print(__doc__)
