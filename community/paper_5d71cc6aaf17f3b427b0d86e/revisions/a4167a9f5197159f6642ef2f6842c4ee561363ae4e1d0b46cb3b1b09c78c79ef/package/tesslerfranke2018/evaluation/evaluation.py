"""Evaluation for Tessler & Franke (2018): derives the paper's comparisons from sessions alone.

The paper fits lme4 mixed models. Those are not available offline, so each test is approximated with
participant-level contrasts (difference between the mean rating of antonyms and of negated positives per
participant) and t-tests. Statistics are therefore NOT numerically comparable to the paper's b and t.
"""
import math

REPORTED = {
    "exp1_interaction": "b = 0.029, t(16) = 2.4, p = 0.029",
    "exp2_interaction_single": "b = 0.011, t(565) = 2.68, p = 0.0076",
    "exp2_morphological_vs_negated_positive": "t(52) = 0.02, p = 0.98 (not different)",
    "exp2_lexical_vs_negated_positive": "t(280) = 3.66, p = 0.0003 (lexical more negative)",
    "exp2_context_morphological": "b = 0.032, t(6457) = 6.73",
    "exp2_three_way": "b = 0.012, t(469) = 1.64, p = 0.1 (not significant)",
}
ORDER = ["antonym", "negated_positive", "negated_antonym", "positive"]


def _mean(x):
    return sum(x) / len(x)


def _var(x):
    m = _mean(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - 1)


def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1 + aa / c
        c = c if abs(c) > 1e-30 else 1e-30
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1 + aa / c
        c = c if abs(c) > 1e-30 else 1e-30
        h *= d * c
        if abs(d * c - 1) < 1e-10:
            break
    return h


def _p_two_sided(t, df):
    if df <= 0 or not math.isfinite(t):
        return None
    x = df / (df + t * t)
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = math.lgamma(df / 2 + 0.5) - math.lgamma(df / 2) - math.lgamma(0.5)
    front = math.exp(lb + (df / 2) * math.log(x) + 0.5 * math.log(1 - x))
    if x < (df / 2 + 1) / (df / 2 + 0.5 + 2):
        return front * _betacf(df / 2, 0.5, x) / (df / 2)
    return 1 - front * _betacf(0.5, df / 2, 1 - x) / 0.5


def _one(x):
    if len(x) < 2 or _var(x) == 0:
        return None
    t = _mean(x) / math.sqrt(_var(x) / len(x))
    return {"mean": _mean(x), "t": t, "df": len(x) - 1, "p": _p_two_sided(t, len(x) - 1), "n": len(x)}


def _welch(x, y):
    if len(x) < 2 or len(y) < 2:
        return None
    vx, vy = _var(x) / len(x), _var(y) / len(y)
    if vx + vy == 0:
        return None
    t = (_mean(x) - _mean(y)) / math.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    return {"diff": _mean(x) - _mean(y), "t": t, "df": df, "p": _p_two_sided(t, df), "n": [len(x), len(y)]}


def _contrast(session, antonym_type=None):
    """Mean rating of antonyms minus negated positives for one participant (negative = antonym rated lower)."""
    a, np_ = [], []
    for r in session["trials"]:
        if r["rating"] is None or (antonym_type and r["antonym_type"] != antonym_type):
            continue
        if r["adjective_type"] == "antonym":
            a.append(r["rating"])
        elif r["adjective_type"] == "negated_positive":
            np_.append(r["rating"])
    return _mean(a) - _mean(np_) if a and np_ else None


def _cell_means(sessions):
    out = {}
    for arm in sorted({s["arm"] for s in sessions}):
        rows = {}
        for s in (x for x in sessions if x["arm"] == arm):
            vals = [r["rating"] for r in s["trials"] if r["rating"] is not None]
            if len(vals) < 2 or _var(vals) == 0:
                continue
            m, sd = _mean(vals), math.sqrt(_var(vals))
            for r in s["trials"]:
                if r["rating"] is not None:
                    key = (r["antonym_type"], r["adjective_type"]) if arm == "exp1" else r["adjective_type"]
                    rows.setdefault(key, []).append((r["rating"] - m) / sd)
        out[arm] = {("%s/%s" % k if isinstance(k, tuple) else k): round(_mean(v), 3) for k, v in sorted(rows.items())}
    return out


def evaluate(sessions):
    if not sessions:
        return {"status": "not_ready", "missing": ["no sessions"]}
    by = lambda arm: [s for s in sessions if s["arm"] == arm]
    tests, missing = {}, []

    def vals(arm, atype=None):
        return [c for c in (_contrast(s, atype) for s in by(arm)) if c is not None]

    if by("exp1"):
        diffs = []
        for s in by("exp1"):
            m, l = _contrast(s, "morphological"), _contrast(s, "lexical")
            if m is not None and l is not None:
                diffs.append(m - l)
        tests["exp1_interaction"] = _one(diffs)
    else:
        missing.append("exp1 sessions")
    ms, ls = vals("exp2-morphological-single"), vals("exp2-lexical-single")
    if ms and ls:
        tests["exp2_interaction_single"] = _welch(ms, ls)
        tests["exp2_morphological_vs_negated_positive"] = _one(ms)
        tests["exp2_lexical_vs_negated_positive"] = _one(ls)
    else:
        missing.append("exp2 single-utterance sessions (both antonym types)")
    mm, lm = vals("exp2-morphological-multiple"), vals("exp2-lexical-multiple")
    if ms and mm:
        tests["exp2_context_morphological"] = _welch(mm, ms)
    else:
        missing.append("exp2 morphological single and multiple sessions")
    if ms and ls and mm and lm and min(map(len, (ms, ls, mm, lm))) > 1:
        d = (_mean(lm) - _mean(ls)) - (_mean(mm) - _mean(ms))
        se = math.sqrt(sum(_var(x) / len(x) for x in (ms, ls, mm, lm)))
        if se > 0:
            df = min(len(x) for x in (ms, ls, mm, lm)) - 1
            tests["exp2_three_way"] = {"diff": d, "t": d / se, "df": df, "p": _p_two_sided(d / se, df)}
    else:
        missing.append("all four exp2 arms for the three-way contrast")
    tests = {k: v for k, v in tests.items() if v is not None}
    if not tests:
        return {"status": "not_ready", "missing": missing or ["variance in ratings"]}
    return {"status": "ok" if not missing else "partial", "missing": missing,
            "note": "Participant-level approximations of the paper's lme4 models; contrast = mean(antonym) - mean(negated positive) on a 0-100 slider.",
            "ordering_in_paper": {"morphological": "partial (antonym ~ negated positive) in isolation", "lexical": "full ordering"},
            "normalized_cell_means": _cell_means(sessions), "tests": tests, "reported_in_paper": REPORTED}
