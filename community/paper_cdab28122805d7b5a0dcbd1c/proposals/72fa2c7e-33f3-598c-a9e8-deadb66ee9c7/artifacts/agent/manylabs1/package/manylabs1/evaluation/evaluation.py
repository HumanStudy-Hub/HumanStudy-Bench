#!/usr/bin/env python3
"""Evaluation for the Many Labs 1 replay: the paper's own confirmatory comparison per study.

evaluate(sessions) -> dict. Every metric is computed from `sessions` alone (plus materials.json for the
wording/scoring flags). A study whose test cannot run returns {"status": "not_ready", "missing": [...]}.
Reference effect sizes are the weighted replication effect sizes in Table 2 of the paper (Cohen's d).
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "task"))

PAPER_WEIGHTED_D = {"s01": 0.27, "s02": 0.60, "s03:nyc": 1.17, "s03:chicago": 1.79, "s03:everest": 2.23, "s03:babies": 2.42,
                    "s04": 0.61, "s05": 0.49, "s06": 0.30, "s07": 1.96, "s08": 0.32, "s09": 0.03, "s10": -0.02, "s11": 0.13}
# first condition minus second condition, in the direction the original effect predicts
CONTRAST = {"s01": ("paid", "free"), "s02": ("gain", "loss"), "s03": ("high", "low"), "s04": ("three6", "two6one3"),
            "s05": ("high", "low"), "s06": ("us_first", "nk_first"), "s07": ("allow", "forbid"), "s08": ("liked", "disliked"),
            "s09": ("flag", "control"), "s10": ("money", "control"), "s11": ("contact", "control")}
ANCHOR_TOPICS = ["nyc", "chicago", "everest", "babies"]   # order the paper lists them


def _mean(x):
    return sum(x) / len(x)


def _var(x):
    m = _mean(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - 1)


def _betacf(a, b, x):
    tiny, qab, qap, qam = 1e-30, a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c; c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c; c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1) < 3e-12:
            break
    return h


def _betai(a, b, x):
    if x <= 0 or x >= 1:
        return float(x >= 1)
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return front * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - front * _betacf(b, a, 1 - x) / b


def t_test(a, b):
    """Pooled-variance independent samples t-test (a minus b) with Cohen's d."""
    if len(a) < 2 or len(b) < 2:
        return None
    df = len(a) + len(b) - 2
    sp2 = ((len(a) - 1) * _var(a) + (len(b) - 1) * _var(b)) / df
    if sp2 == 0:
        return None
    diff = _mean(a) - _mean(b)
    t = diff / math.sqrt(sp2 * (1 / len(a) + 1 / len(b)))
    p = _betai(df / 2, 0.5, df / (df + t * t))
    return {"test": "t", "t": t, "df": df, "p": p, "d": diff / math.sqrt(sp2), "n": [len(a), len(b)], "means": [_mean(a), _mean(b)]}


def chi_square(a_yes, a_n, b_yes, b_n):
    """2x2 chi-square (no correction) of condition A versus B; effect sizes: phi and log-odds d."""
    n = a_n + b_n
    if a_n < 1 or b_n < 1:
        return None
    cells = [[a_yes, a_n - a_yes], [b_yes, b_n - b_yes]]
    rows, cols = [a_n, b_n], [a_yes + b_yes, n - a_yes - b_yes]
    if 0 in cols:
        return None
    chi = sum((cells[i][j] - rows[i] * cols[j] / n) ** 2 / (rows[i] * cols[j] / n) for i in range(2) for j in range(2))
    p = math.erfc(math.sqrt(chi / 2))
    if min(min(r) for r in cells) == 0:
        d = None
    else:
        d = math.log(cells[0][0] * cells[1][1] / (cells[0][1] * cells[1][0])) * math.sqrt(3) / math.pi
    return {"test": "chi2", "chi2": chi, "df": 1, "p": p, "phi": math.sqrt(chi / n), "d_logodds": d, "n": [a_n, b_n],
            "proportions": [a_yes / a_n, b_yes / b_n],
            "note": "d_logodds = ln(OR)*sqrt(3)/pi; the paper does not state its chi-square to d conversion"}


def _rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks, i = [0.0] * len(values), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _gather(sessions, study):
    groups = {}
    for s in sessions:
        for r in s.get("responses", []):
            if r["study"] == study and r.get("parsed") is not None:
                groups.setdefault(r["condition"], []).append(r["parsed"])
    return groups


def _compare(study, groups, value):
    first, second = CONTRAST[study]
    a = [value(p) for p in groups.get(first, [])]
    b = [value(p) for p in groups.get(second, [])]
    a, b = [v for v in a if v is not None], [v for v in b if v is not None]
    return a, b


def _not_ready(reason, missing):
    return {"status": "not_ready", "reason": reason, "missing": missing}


def evaluate(sessions, materials=None):
    mats = materials or json.loads((HERE.parent / "materials" / "materials.json").read_text(encoding="utf-8"))
    out = {"status": "ok", "studies": {}, "paper_weighted_d": PAPER_WEIGHTED_D,
           "note": "A simulation result, to be read next to the paper's Table 2; agreement does not show that people would behave the same way."}
    skipped = {}
    for s in sessions or []:
        skipped.update(s.get("not_ready", {}))
    for study in mats["studies"]:
        entry = mats["studies"][study]
        if study == "s12":
            out["studies"][study] = _not_ready("The IAT needs reaction-time latencies that a text participant cannot produce.", ["IAT latencies"])
            continue
        if study in skipped:
            out["studies"][study] = _not_ready(skipped[study], entry.get("missing", []))
            continue
        groups = _gather(sessions or [], study)
        first, second = CONTRAST[study]
        if not groups.get(first) or not groups.get(second):
            out["studies"][study] = _not_ready("No parsed responses in both conditions.", [first, second])
            continue
        res = _study_test(study, groups, entry)
        out["studies"][study] = res if res else _not_ready("The test could not be computed from the responses.", ["variance or counts"])
    if all(v.get("status") == "not_ready" for v in out["studies"].values()):
        out["status"] = "not_ready"
    return out


def _study_test(study, groups, entry):
    if study in ("s01", "s08"):
        # s08 is printed 1 = strongly agree .. 9 = strongly disagree; reversed so higher = more agreement
        flip = (lambda p: 10 - p["value"]) if study == "s08" else (lambda p: p["value"])
        a, b = _compare(study, groups, flip)
        return _wrap(t_test(a, b))
    if study == "s02":
        a, b = _compare(study, groups, lambda p: 1 if p.get("sure") else 0)
        return _wrap(chi_square(sum(a), len(a), sum(b), len(b)))
    if study == "s03":
        res = {}
        for k, topic in enumerate(ANCHOR_TOPICS):
            high = [p["values"][k] for p in groups.get("high", []) if len(p.get("values", [])) > k]
            low = [p["values"][k] for p in groups.get("low", []) if len(p.get("values", [])) > k]
            ranks = _rank(high + low)
            res[topic] = _wrap(t_test(ranks[: len(high)], ranks[len(high):]))
        return {"status": "ok", "topics": res}
    if study == "s04":
        a, b = _compare(study, groups, lambda p: math.sqrt(p["values"][0]) if p["values"][0] >= 0 else None)
        pooled = a + b
        if len(pooled) < 3:
            return None
        m, sd = _mean(pooled), math.sqrt(_var(pooled))
        a, b = [v for v in a if abs(v - m) <= 3 * sd], [v for v in b if abs(v - m) <= 3 * sd]
        return _wrap(t_test(a, b))
    if study == "s05":
        thr = entry.get("response_format", {}).get("over_threshold_labels")
        if not thr:
            return _not_ready("The category labels above 2.5 hours are not in the materials.", ["category scale"])
        a, b = _compare(study, groups, lambda p: 1 if p.get("choice") in thr else 0)
        return _wrap(chi_square(sum(a), len(a), sum(b), len(b)))
    if study == "s06":
        a, b = _compare(study, groups, lambda p: 1 if p["yes"] else 0)
        return _wrap(chi_square(sum(a), len(a), sum(b), len(b)))
    if study == "s07":
        # restrictive answer: "No" to allowing, "Yes" to forbidding
        a = [1 if not p["yes"] else 0 for p in groups.get("allow", [])]
        b = [1 if p["yes"] else 0 for p in groups.get("forbid", [])]
        return _wrap(chi_square(sum(a), len(a), sum(b), len(b)))
    if study in ("s09", "s10", "s11"):
        if not entry.get("scoring_ready"):
            return _not_ready("Item scoring (direction and scale) is not in the materials.", ["scoring key"])
        rev, top = set(entry.get("reverse_items", [])), entry.get("response_format", {}).get("max", 0)
        score = lambda p: _mean([(top + 1 - v) if i + 1 in rev else v for i, v in enumerate(p["items"])]) if p["items"] else None
        a, b = _compare(study, groups, score)
        return _wrap(t_test(a, b))
    return None


def _wrap(r):
    return dict(r, status="ok") if r else None


def _smoke():
    import adapter
    out = evaluate([], materials=json.loads((HERE.parent / "materials" / "materials.json").read_text(encoding="utf-8")))
    assert out["status"] == "not_ready", "no sessions means not_ready"
    # known numbers: identical groups give t = 0; clearly separated groups give a large positive t
    assert abs(t_test([1, 2, 3, 4], [1, 2, 3, 4])["t"]) < 1e-9
    r = t_test([5, 6, 7, 8, 9], [1, 2, 3, 4, 5])
    assert abs(r["t"] - 4.0) < 1e-9 and 0.003 < r["p"] < 0.006
    c = chi_square(30, 100, 10, 100)
    assert c["chi2"] > 10 and c["p"] < 0.01
    mats = adapter._smoke_materials()
    tick = [0]
    def fake(prompt):
        item = prompt.split("\n\n")[-2]
        tick[0] += 1
        jitter = tick[0] % 2
        if "paid" in item:
            return f"ANSWER: {7 + jitter}"
        if "free" in item:
            return f"ANSWER: {3 + jitter}"
        if "allow" in item:
            return "ANSWER: No" if tick[0] % 3 else "ANSWER: Yes"
        return "ANSWER: Yes" if tick[0] % 5 == 0 else "ANSWER: No"
    sessions = adapter.run_sessions(fake, 3, 20, materials=mats)
    mats_for_eval = json.loads((HERE.parent / "materials" / "materials.json").read_text(encoding="utf-8"))
    mats_for_eval["studies"].update({k: dict(mats_for_eval["studies"][k], **{"response_format": mats["studies"][k]["response_format"]}) for k in mats["studies"]})
    res = evaluate(sessions, materials=mats_for_eval)
    assert res["studies"]["s01"]["status"] == "ok" and res["studies"]["s01"]["t"] > 0
    assert res["studies"]["s07"]["status"] == "ok"
    print("evaluation smoke ok")


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        _smoke()
    else:
        print(__doc__)
