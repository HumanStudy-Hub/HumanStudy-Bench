#!/usr/bin/env python3
"""Offline adapter: builds one participant prompt per AI subject and parses the reply.

Usage: python task/adapter.py --smoke-test
run_sessions(llm, seed, n, arms=None): llm(prompt) -> str is injected; one call per AI participant.
Prompts are assembled from the verbatim questionnaire text in materials/materials.json
(sources/RG_questionnaire_items.pdf). If an entry is not verbatim, build_prompt raises
MaterialsMissing instead of inventing wording.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ["c00", "c10", "c01", "c11"]  # (tempting fate, load): 00, 10, 01, 11


class MaterialsMissing(RuntimeError):
    pass


def materials():
    return json.loads((HERE.parent / "materials/materials.json").read_text(encoding="utf-8"))["studies"]["s1"]


def _text(item):
    return item.get("text") if isinstance(item, dict) and item.get("status") == "verbatim" and item.get("text") else None


def missing_materials(condition):
    m, cond = materials(), materials()["conditions"][condition]
    need = {"scenario": _text(m["scenario"]["tempted_fate" if cond["tempting_fate"] else "not_tempted_fate"]),
            "likelihood_item": _text(m["likelihood_item"])}
    need["lead_in"] = _text(m["scenario"]["lead_in_load" if cond["cognitive_load"] else "lead_in_no_load"])
    if cond["cognitive_load"]:
        need["cognitive_load"] = _text(m["cognitive_load"])
    for k in secondary_keys(condition):
        need[k] = _text(m["secondary_items"][k])
    return [k for k, v in need.items() if not v]


def secondary_keys(condition):
    order = ["end_number", "effort_split", "difficulty", "effort", "importance", "negativity"]
    sec = materials()["secondary_items"]
    return [k for k in order if condition in sec[k]["conditions"]]


def _item_block(item):
    text = item["text"]
    if item.get("options"):
        anchors = [item.get("anchor_low"), item.get("anchor_mid"), item.get("anchor_high")]
        text += "\n" + "    ".join(a for a in anchors if a) + "\n" + item["options"]
    return text


def build_prompt(study_id, condition_id):
    if study_id != "s1" or condition_id not in ARMS:
        raise ValueError("unknown study or condition")
    gaps = missing_materials(condition_id)
    if gaps:
        raise MaterialsMissing("not printed in a source, supply it first: " + ", ".join(gaps))
    m, cond = materials(), materials()["conditions"][condition_id]
    parts = []
    if cond["cognitive_load"]:
        parts.append(_text(m["cognitive_load"]))
    parts.append(_text(m["scenario"]["lead_in_load" if cond["cognitive_load"] else "lead_in_no_load"]))
    parts.append(_text(m["scenario"]["tempted_fate" if cond["tempting_fate"] else "not_tempted_fate"]))
    parts.append(_item_block(m["likelihood_item"]))
    keys = ["likelihood"] + secondary_keys(condition_id)
    for k in secondary_keys(condition_id):
        parts.append(_item_block(m["secondary_items"][k]))
    parts.append("Reply only with JSON using these keys: " + ", ".join(keys) +
                 ". Each value is the whole number you choose (likelihood and ratings on the printed scale; end_number is the number you ended on).")
    return "\n\n".join(parts)


def parse_reply(study_id, condition_id, reply):
    keys = ["likelihood"] + secondary_keys(condition_id)
    out = {}
    try:
        data = json.loads(re.search(r"\{.*\}", reply, re.S).group(0))
    except Exception:
        data = {}
    for k in keys:
        try:
            v = int(round(float(data[k])))
            hi = 6 if k == "effort_split" else 10
            out[k] = v if (k == "end_number" or 0 <= v <= hi) else None
        except Exception:
            out[k] = None
    if out["likelihood"] is None:
        m = re.search(r"\b(10|[0-9])\b", reply or "")
        out["likelihood"] = int(m.group(1)) if m else None
    return out


def run_sessions(llm, seed, n, arms=None):
    sessions = []
    for arm in (arms or ARMS):
        for i in range(n):
            reply = llm(build_prompt("s1", arm))
            cond = materials()["conditions"][arm]
            sessions.append({"study": "s1", "condition": arm, "arm": arm, "participant": "%s-%d" % (arm, i),
                             "seed": seed, "tempting_fate": cond["tempting_fate"],
                             "cognitive_load": cond["cognitive_load"], "reply": reply,
                             "responses": parse_reply("s1", arm, reply)})
    return sessions


def smoke_test():
    task = json.loads((HERE / "task.json").read_text(encoding="utf-8"))
    assert task["arms"] == ARMS and not task["scripted_roles"]
    m = materials()
    assert set(m["conditions"]) == set(ARMS)
    for arm in ARMS:
        try:
            build_prompt("s1", arm)
            state = "prompt builds"
        except MaterialsMissing as exc:
            state = "blocked (%s)" % exc
        print(arm, state, "| secondary:", ",".join(secondary_keys(arm)))
    assert secondary_keys("c00") == ["importance", "negativity"]
    assert secondary_keys("c01") == ["end_number", "effort_split", "difficulty", "effort", "importance", "negativity"]
    assert parse_reply("s1", "c01", '{"likelihood": 4, "effort": 6, "difficulty": 5, "importance": 8, "negativity": 7}')["effort"] == 6
    assert parse_reply("s1", "c00", "no number here")["likelihood"] is None
    print("smoke test ok")


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
    else:
        print(__doc__)
