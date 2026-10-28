#!/usr/bin/env python3
"""Offline adapter for the Many Labs 1 replay (Klein et al., 2014).

One AI participant completes the bundled session: the studies in a random order (the math IAT, which a
text participant cannot do, is excluded), each answered in turn while seeing its own earlier exchange.
The adapter never answers for a participant. `llm(prompt) -> str` is injected by the caller.

Participant-visible wording comes only from ../materials/materials.json. Where that file has no wording
(participant_text is null) the study is reported as not_ready instead of being invented.

    python task/adapter.py --smoke-test
"""
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MATERIALS_PATH = HERE.parent / "materials" / "materials.json"
TASK_PATH = HERE / "task.json"

# Implementation text, not part of any printed instrument.
PREFACE = ("You are taking part in an online study as a participant. Answer each item as you would if you "
           "were really taking part. End every answer with one line that starts with 'ANSWER:' followed by "
           "your answer in the requested format.")


class MissingMaterial(Exception):
    pass


def load_materials(path=MATERIALS_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def study_ready(materials, study):
    entry = materials["studies"][study]
    return bool(entry.get("runnable", True)) and all(c.get("participant_text") for c in entry["conditions"].values())


def build_prompt(study, condition, history=(), materials=None):
    """Prompt for one participant on one study in one condition; `history` is [(prompt_text, reply), ...]."""
    materials = materials or load_materials()
    entry = materials["studies"][study]
    text = entry["conditions"][condition].get("participant_text")
    if not text:
        raise MissingMaterial(f"{study}/{condition}: participant-visible wording is missing")
    parts = [PREFACE]
    for earlier_text, earlier_reply in history:
        parts.append(f"[Earlier in this session]\n{earlier_text}\nYour reply: {earlier_reply}")
    parts.append(text)
    fmt = entry.get("response_format", {})
    if fmt.get("instruction"):
        parts.append(fmt["instruction"])
    return "\n\n".join(parts)


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers(text):
    out = []
    for token in _NUM.findall(text.replace("−", "-")):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            pass
    return out


def _answer_line(reply):
    found = re.findall(r"ANSWER:\s*(.*)", reply or "", re.I)
    return found[-1].strip() if found else None


def parse_reply(study, condition, reply, materials=None):
    """Turn a reply into the response format the paper used; None when it cannot be parsed."""
    materials = materials or load_materials()
    fmt = materials["studies"][study].get("response_format", {})
    answer = _answer_line(reply)
    if answer is None:
        return None
    kind = fmt.get("kind")
    if kind == "rating":
        nums = _numbers(answer)
        lo, hi = fmt.get("min"), fmt.get("max")
        if not nums or (lo is not None and not lo <= nums[0] <= hi):
            return None
        return {"value": nums[0]}
    if kind == "number":
        nums = _numbers(answer)
        if re.search(r"million", answer, re.I) and nums:
            nums = [nums[0] * 1e6]
        return {"values": nums[: fmt.get("count", 1)]} if len(nums) >= fmt.get("count", 1) else None
    if kind == "yesno":
        m = re.match(r"\s*(yes|no)\b", answer, re.I)
        return {"yes": m.group(1).lower() == "yes"} if m else None
    if kind == "choice":
        options = fmt.get("options", {}).get(condition) or fmt.get("options", {}).get("all") or []
        for opt in options:
            letter = re.fullmatch(r"Program ([A-D])", opt["label"])
            pattern = r"(?:program\s+)?" + letter.group(1) if letter else re.escape(opt["label"])
            if re.match(r"\s*" + pattern + r"\b", answer, re.I):
                return {"choice": opt["label"], **{k: v for k, v in opt.items() if k != "label"}}
        return None
    if kind == "yesno_pair":
        found = [w.lower() == "yes" for w in re.findall(r"\b(yes|no)\b", answer, re.I)]
        order = fmt["item_order"][condition]
        if len(found) != len(order):
            return None
        by_item = dict(zip(order, found))
        return {"yes": by_item["us_lets_nk"], "yes_nk_lets_us": by_item["nk_lets_us"]}
    if kind == "items":
        nums = _numbers(answer)[: fmt.get("count") or None]
        need = fmt.get("count")
        lo, hi = fmt.get("min"), fmt.get("max")
        if not need or len(nums) < need or (lo is not None and not all(lo <= v <= hi for v in nums)):
            return None
        return {"items": nums}
    return None


def _assign(rng, conditions, k):
    """Balanced random assignment (implementation choice; the paper does not state the rule)."""
    order = list(conditions)
    rng.shuffle(order)
    return order[k % len(order)]


def run_sessions(llm, seed, n, arms=None, materials=None):
    """Run n AI participants. `arms` optionally restricts which studies are run (list of study ids).

    Returns one dict per participant with `responses` (one per completed study; each has `study`,
    `condition`, `arm`, `parsed`) and `not_ready` (studies skipped because wording is missing)."""
    materials = materials or load_materials()
    studies = [s for s in materials["studies"] if arms is None or s in arms]
    rng = random.Random(seed)
    sessions = []
    runnable = [s for s in studies if study_ready(materials, s)]
    skipped = {s: materials["studies"][s].get("not_ready_reason", "participant-visible wording is missing") for s in studies if s not in runnable}
    for k in range(n):
        order = runnable[:]
        rng.shuffle(order)
        history, responses = [], []
        for study in order:
            conds = list(materials["studies"][study]["conditions"])
            cond = _assign(rng, conds, k)
            prompt = build_prompt(study, cond, history, materials)
            reply = llm(prompt)
            history.append((materials["studies"][study]["conditions"][cond]["participant_text"], reply))
            responses.append({"study": study, "condition": cond, "arm": f"{study}:{cond}", "prompt": prompt,
                              "reply": reply, "parsed": parse_reply(study, cond, reply, materials)})
        sessions.append({"participant": k, "seed": seed, "responses": responses, "not_ready": skipped})
    return sessions


def _smoke_materials():
    """In-memory fixture used only to exercise the code path; it is never written or shown to participants."""
    def cond(label):
        return {"label": label, "participant_text": f"[smoke-test item for {label}]"}
    return {"studies": {
        "s01": {"conditions": {"paid": cond("paid"), "free": cond("free")},
                "response_format": {"kind": "rating", "min": 1, "max": 9, "instruction": "[smoke-test format]"}},
        "s07": {"conditions": {"allow": cond("allow"), "forbid": cond("forbid")},
                "response_format": {"kind": "yesno", "instruction": "[smoke-test format]"}}}}


def smoke_test():
    real = load_materials()
    assert set(real["studies"]) >= {f"s{k:02d}" for k in range(1, 13)}, "materials must list every study"
    calls = []
    def canned(prompt):
        calls.append(prompt)
        return "ANSWER: 5"
    out = run_sessions(canned, seed=1, n=2, materials=real)
    ready = [s for s in real["studies"] if study_ready(real, s)]
    assert ready and all(len(o["responses"]) == len(ready) for o in out), "every ready study gets one call per participant"
    assert len(calls) == 2 * len(ready)
    assert set(out[0]["not_ready"]) == {"s09", "s10", "s12"}, "image and IAT studies are reported, not run"
    for study in ready:   # no condition may be left without printed text
        assert all(c["participant_text"] for c in real["studies"][study]["conditions"].values())
    # parsers against the real response formats
    assert parse_reply("s01", "paid", "ANSWER: 7", real) == {"value": 7.0}
    assert parse_reply("s01", "paid", "ANSWER: 12", real) is None, "out-of-range rating is rejected"
    assert parse_reply("s02", "loss", "ANSWER: Program D", real)["sure"] is False
    assert parse_reply("s06", "us_first", "ANSWER: Yes; No", real) == {"yes": False, "yes_nk_lets_us": True}
    assert parse_reply("s05", "high", "ANSWER: Three to three and a half hours", real)["choice"] == "Three to three and a half hours"
    assert parse_reply("s11", "contact", "ANSWER: 1; 2; 3; 4", real) == {"items": [1.0, 2.0, 3.0, 4.0]}
    assert parse_reply("s03", "low", "ANSWER: 2,000; 500000; 3000; 10000", real)["values"][0] == 2000.0
    print("smoke test ok")


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
    else:
        print(__doc__)
