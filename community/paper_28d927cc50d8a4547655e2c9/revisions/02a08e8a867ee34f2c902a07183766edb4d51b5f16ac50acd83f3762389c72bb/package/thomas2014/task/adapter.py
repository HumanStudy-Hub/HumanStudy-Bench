#!/usr/bin/env python3
"""Offline adapter: builds AI-participant prompts for the three Thomas et al. (2014) experiments.

run_sessions(llm, seed, n, arms=None): llm(prompt) -> str is injected. Every original participant is
one AI participant (n per arm); nothing is scripted except the experimenter/interface. Pairing and
payment were done offline in the original, so sessions are independent. Prompts are built only from
../materials/materials.json. Run `python adapter.py --smoke-test` for an offline structure check.
"""
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "..", "materials", "materials.json"), encoding="utf-8") as _f:
    MAT = json.load(_f)

KNOWLEDGE = ["primary", "secondary", "tertiary", "common"]
PAYOFFS = [1, 2, 5, 10]
SIMILARITY = ["similar", "dissimilar"]
HEADER = ("You are a participant in an online decision-making study. Read each screen and answer "
          "it as you would if you were really taking part. Reply only in the requested format.")


def all_arms():
    arms = []
    for study in ("exp1", "exp2"):
        for k in KNOWLEDGE:
            for p in PAYOFFS:
                arms.append({"arm": f"{study}/{k}/{p}", "study": study, "knowledge": k, "payoff": p, "similarity": None})
    for k in KNOWLEDGE:
        for s in SIMILARITY:
            arms.append({"arm": f"exp3/{k}/2/{s}", "study": "exp3", "knowledge": k, "payoff": 2, "similarity": s})
    return arms


_SWAP = [("your butcher shop", "your bakery"), ("your bakery", "your butcher shop"), ("the butcher shop", "the bakery"),
         ("butcher shop", "bakery"), ("Butcher", "Baker"), ("Baker", "Butcher"), ("butcher", "baker"), ("baker", "butcher")]


def swap_role(text):
    """Mechanical Butcher<->Baker swap (the paper prints the baker's version only; see issue iss_e1_butcher)."""
    pattern = re.compile("|".join(re.escape(a) for a, _ in sorted(_SWAP, key=lambda x: -len(x[0]))))
    table = dict(_SWAP)
    return pattern.sub(lambda m: table[m.group(0)], text)


def money(x):
    return f"${x:.2f}"


def role_for(study, index, rng_offset):
    if study == "exp2":
        return "baker"  # only the baker perspective is printed (issue iss_e2_roles)
    return "baker" if (index + rng_offset) % 2 == 0 else "butcher"


def _message(study, knowledge, role, payoff):
    m = MAT["studies"][study]["messages"][knowledge]
    together = money(payoff + 0.10)
    private = m["private_box"]
    public = m.get("public_box") or MAT["common"]["silent_loudspeaker"]
    if knowledge == "common":
        public = public.replace("[today’s price]", together).replace("[earnings for working together]", together)
        private = private.replace("[today’s price]", together)
    if role == "butcher" and study != "exp2":
        private, public = swap_role(private), swap_role(public)
    return public, private


def build_prompt(study, condition, role="baker"):
    """First screen sequence of one AI participant: instructions, the day's information, decision request."""
    payoff = condition["payoff"]
    partner = "baker" if role == "butcher" else "butcher"
    parts = [MAT["studies"][study]["instructions"].format(role=role, partner=partner, alone=money(payoff), together=money(payoff + 0.10))]
    public, private = _message(study, condition["knowledge"], role, payoff)
    parts.append(MAT["common"]["reveal"].format(together=money(payoff + 0.10), alone=money(payoff)))
    parts.append("PUBLIC BOX (visible to you and your partner): " + public)
    parts.append("PRIVATE BOX (visible only to you): " + private)
    parts.append(MAT["common"]["decision"])
    return "\n\n".join(parts)


def parse_reply(study, condition, reply):
    """-> 'together' | 'alone' | None"""
    text = (reply or "").upper()
    m = re.search(r"DECISION\s*:\s*WORK\s+(TOGETHER|ALONE)", text)
    if m:
        return m.group(1).lower()
    t, a = "WORK TOGETHER" in text, "WORK ALONE" in text
    return "together" if t and not a else "alone" if a and not t else None


def _numbers(text, count):
    found = re.findall(r"(?:^|\n)\s*(?:Q?\d\s*[:.)-]\s*)?\$?\s*(-?\d+(?:\.\d+)?)", text or "")
    out = [float(x) for x in found][:count]
    return out + [None] * (count - len(out))


def parse_payoff_answers(reply):
    return _numbers(reply, 3)


def parse_knowledge_answers(reply):
    answers = []
    for line in (reply or "").splitlines():
        m = re.match(r"\s*Q?(\d)\s*[:.)-]\s*(.*)", line)
        if not m:
            continue
        v = m.group(2).strip().lower()
        answers.append("yes" if v.startswith("yes") else "no" if v.startswith("no") else "cant_tell" if ("tell" in v or "possible" in v) else None)
    return (answers + [None] * 4)[:4]


_PATTERN = {"primary": ("cant_tell",) * 3, "secondary": ("yes", "no", "no"), "tertiary": ("yes", "yes", "no"), "common": ("yes", "yes", "yes")}


def classify_level(answers):
    """Reported level of knowledge from the four knowledge answers (implementation reading; issue iss_e3_levelrule)."""
    if answers[0] != "yes":
        return "unclassifiable"
    for level, pattern in _PATTERN.items():
        if tuple(answers[1:]) == pattern:
            return level
    return "unclassifiable"


def _screen(prompt, reply_text, n):
    return f"[Screen {n}]\n{prompt}\n[Your answer]\n{reply_text}"


def run_session(llm, study, condition, role, rng):
    log = []

    def ask(prompt):
        history = "\n\n".join(_screen(p, r, i + 1) for i, (p, r) in enumerate(log))
        full = HEADER + ("\n\n" + history if history else "") + f"\n\n[Screen {len(log) + 1}]\n{prompt}\n[Your answer]"
        reply = llm(full)
        log.append((prompt, reply))
        return reply

    session = {"study": study, "condition": dict(condition), "arm": condition["arm"], "role": role}
    if study == "exp3":
        session["similarity_answers"] = ask(MAT["studies"]["exp3"]["similarity_questions"])
        told = MAT["studies"]["exp3"]["similarity_told"][condition["similarity"]] + "\n\n" + MAT["studies"]["exp3"]["rating_prompt"]
        rating = ask(told)
        nums = _numbers(rating, 1)[0]
        session["perceived_similarity"] = None if nums is None else max(0.0, min(100.0, nums))
    reply = ask(build_prompt(study, condition, role))
    session["decision"] = parse_reply(study, condition, reply)
    session["decision_raw"] = reply
    if study in ("exp1", "exp2"):
        session["explanation"] = ask(MAT["common"]["explain"])
        pay = ask(MAT["studies"][study]["payoff_questions"].format(partner="Baker" if role == "butcher" else "Butcher"))
        session["payoff_answers"] = parse_payoff_answers(pay)
        payoff = condition["payoff"]
        session["payoff_correct"] = session["payoff_answers"] == [round(payoff + 0.10, 2), 0.0, float(payoff)]
        know = ask(MAT["studies"][study]["knowledge_questions"].format(partner="Baker" if role == "butcher" else "Butcher", partner_lc="baker" if role == "butcher" else "butcher"))
        session["knowledge_answers"] = parse_knowledge_answers(know)
    else:
        session["big_five"] = None  # the 50 Big Five items are not in the paper (issue iss_e3_big5)
        know = ask(MAT["studies"]["exp3"]["knowledge_questions"].format(partner="Baker" if role == "butcher" else "Butcher", partner_lc="baker" if role == "butcher" else "butcher"))
        session["knowledge_answers"] = parse_knowledge_answers(know)
        session["reported_level"] = classify_level(session["knowledge_answers"])
    session["transcript"] = [{"prompt": p, "reply": r} for p, r in log]
    return session


def run_sessions(llm, seed, n, arms=None):
    """n AI participants per arm. arms: list of arm ids or arm dicts; None runs every arm of all three experiments."""
    rng = random.Random(seed)
    chosen = all_arms() if arms is None else [a if isinstance(a, dict) else next(x for x in all_arms() if x["arm"] == a) for a in arms]
    sessions = []
    for arm in chosen:
        offset = rng.randint(0, 1)
        for i in range(n):
            s = run_session(llm, arm["study"], arm, role_for(arm["study"], i, offset), rng)
            s["participant"] = f'{arm["arm"]}#{i + 1}'
            sessions.append(s)
    return sessions


def _smoke():
    calls = []

    def stub(prompt):  # offline structure check only; it does not produce study data
        calls.append(prompt)
        last = prompt.rsplit("[Screen", 1)[-1]
        if "DECISION:" in last:
            return "DECISION: WORK ALONE"
        if "Give a dollar amount" in last:
            return "1: $1.10\n2: $0\n3: $1.00"
        if "Yes, No, or Can't" in last:
            return "1: Yes\n2: Yes\n3: Yes\n4: Yes"
        if "0% to 100%" in last:
            return "50"
        return "ok"
    arms = all_arms()
    assert len(arms) == 40, len(arms)
    out = run_sessions(stub, 1, 1)
    assert len(out) == 40 and all(s["condition"] and s["arm"] and s["decision"] == "alone" for s in out)
    assert swap_role("The Butcher knows") == "The Baker knows"
    assert parse_reply("exp1", {}, "DECISION: WORK TOGETHER") == "together"
    print(f"smoke test ok: {len(arms)} arms, {len(calls)} prompts built with a stub, no study data produced")


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        _smoke()
    else:
        print(__doc__)
