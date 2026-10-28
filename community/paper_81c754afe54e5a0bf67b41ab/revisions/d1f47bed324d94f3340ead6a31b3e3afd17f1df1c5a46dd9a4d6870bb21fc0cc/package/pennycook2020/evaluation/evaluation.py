"""The paper's own comparisons, computed from sessions alone. Standard library only.

Study 1: rating-level linear regression of z-scored yes/no responses on veracity (-0.5 false, 0.5 true),
condition (-0.5 accuracy, 0.5 sharing) and their interaction, standard errors clustered on participants
and headline (two-way). Study 2: sharing ratings rescaled to 0-1, veracity 0/1, treatment 0/1, same clustering.
Item-level: correlation across headlines between Study 1 mean accuracy and the Study 2 treatment effect.
Returns `not_ready` entries (with the reason) when a comparison cannot run.
"""
import math

PAPER = {
    "study1": {"veracity_x_condition": {"beta": -0.126, "F": 42.24, "df": [1, 25586], "p": "<.0001"},
               "accuracy_d": 0.657, "sharing_d": 0.121},
    "study2": {"veracity_x_treatment": {"beta": 0.039, "F": 17.88, "df": [1, 25623], "p": "<.0001"},
               "control_d": 0.050, "treatment_d": 0.142, "discernment_ratio": 2.8},
    "study3": {"r": 0.76, "df": 28},
}


# ------------------------------------------------------------------ small numeric helpers
def _solve(a, b):
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-12:
            raise ZeroDivisionError("singular design")
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                for k in range(c, n + 1):
                    m[r][k] -= f * m[c][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def _inv(a):
    n = len(a)
    cols = [_solve(a, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def _betacf(a, b, x):
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1 - math.exp(lbeta) * _betacf(b, a, 1 - x) / b


def _p_two_sided(t, df):
    if df <= 0 or t != t:
        return float("nan")
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def _round(x, k=4):
    return round(x, k) if isinstance(x, float) and x == x else x


# ------------------------------------------------------------------ clustered regression
def _ols_cluster(X, y, g1, g2):
    """OLS with two-way cluster-robust variance (participant, item; their intersection is the row)."""
    n, k = len(X), len(X[0])
    xtx = [[sum(r[i] * r[j] for r in X) for j in range(k)] for i in range(k)]
    xty = [sum(r[i] * v for r, v in zip(X, y)) for i in range(k)]
    beta = _solve(xtx, xty)
    bread = _inv(xtx)
    resid = [v - sum(b * x for b, x in zip(beta, r)) for r, v in zip(X, y)]

    def meat(groups, adjust=True):
        sums = {}
        for r, e, gid in zip(X, resid, groups):
            s = sums.setdefault(gid, [0.0] * k)
            for i in range(k):
                s[i] += r[i] * e
        m = [[sum(s[i] * s[j] for s in sums.values()) for j in range(k)] for i in range(k)]
        G = len(sums)
        f = (G / (G - 1.0)) * ((n - 1.0) / (n - k)) if adjust and G > 1 else 1.0
        return [[f * v for v in row] for row in m], G

    m1, G1 = meat(g1)
    m2, G2 = meat(g2)
    m12, _ = meat(list(range(n)), adjust=False)
    mid = [[m1[i][j] + m2[i][j] - m12[i][j] for j in range(k)] for i in range(k)]
    tmp = [[sum(bread[i][a] * mid[a][j] for a in range(k)) for j in range(k)] for i in range(k)]
    v = [[sum(tmp[i][a] * bread[a][j] for a in range(k)) for j in range(k)] for i in range(k)]
    return beta, v, min(G1, G2) - 1, n, G1, G2


def _term(beta, v, i, df):
    se = math.sqrt(v[i][i]) if v[i][i] > 0 else float("nan")
    t = beta[i] / se if se == se and se > 0 else float("nan")
    return {"beta": _round(beta[i]), "se": _round(se), "F": _round(t * t), "df": [1, df], "p": _round(_p_two_sided(t, df), 6)}


def _zscore(values):
    m = sum(values) / len(values)
    sd = math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))
    if sd == 0:
        raise ZeroDivisionError("responses have no variance")
    return [(x - m) / sd for x in values]


def _valid(sessions, study):
    return [s for s in sessions if s.get("study") == study and s.get("valid")]


def _rows(sessions, code_condition, value):
    rows = []
    for s in sessions:
        for hid, resp in s["responses"].items():
            rows.append((s["participant"], hid, hid.startswith("True"), code_condition(s["condition"]), value(resp)))
    return rows


def _discernment(sessions, cond):
    per = []
    for s in sessions:
        if s["condition"] != cond:
            continue
        t = [v for h, v in s["responses"].items() if h.startswith("True")]
        f = [v for h, v in s["responses"].items() if h.startswith("False")]
        if t and f:
            per.append((sum(t) / len(t), sum(f) / len(f)))
    if not per:
        return None
    mt, mf = (sum(p[i] for p in per) / len(per) for i in (0, 1))
    return {"n": len(per), "true": _round(mt), "false": _round(mf), "discernment": _round(mt - mf)}


def _study1(sessions):
    ss = _valid(sessions, "study1")
    need = {"accuracy", "sharing"}
    have = {c: sum(s["condition"] == c for s in ss) for c in need}
    if min(have.values()) < 2:
        return {"status": "not_ready", "reason": "needs valid study1 sessions in both conditions (need >= 2 each)", "have": have}
    rows = _rows(ss, lambda c: 0.5 if c == "sharing" else -0.5, lambda v: float(v))
    try:
        y = _zscore([r[4] for r in rows])
        X = [[1.0, 0.5 if r[2] else -0.5, r[3], (0.5 if r[2] else -0.5) * r[3]] for r in rows]
        beta, v, df, n, g1, g2 = _ols_cluster(X, y, [r[0] for r in rows], [r[1] for r in rows])
    except ZeroDivisionError as exc:
        return {"status": "not_ready", "reason": "regression cannot run: %s" % exc, "have": have}
    out = {"status": "ok", "ratings": n, "participants": g1, "headlines": g2, "participants_per_condition": have,
           "veracity_x_condition": _term(beta, v, 3, df), "veracity": _term(beta, v, 1, df),
           "discernment": {c: _discernment(ss, c) for c in sorted(need)}}
    for cond in sorted(need):
        try:
            yy = _zscore([r[4] for r in rows])
            idx = [i for i, r in enumerate(rows) if (r[3] > 0) == (cond == "sharing")]
            b, vv, d, _, _, _ = _ols_cluster([[1.0, 0.5 if rows[i][2] else -0.5] for i in idx], [yy[i] for i in idx],
                                             [rows[i][0] for i in idx], [rows[i][1] for i in idx])
            out["veracity_within_" + cond] = _term(b, vv, 1, d)
        except ZeroDivisionError:
            out["veracity_within_" + cond] = {"status": "not_ready"}
    fa = [v for s in ss if s["condition"] == "accuracy" for h, v in s["responses"].items() if h.startswith("False")]
    fs = [v for s in ss if s["condition"] == "sharing" for h, v in s["responses"].items() if h.startswith("False")]
    if fa and fs:
        pa, ps = sum(fa) / len(fa), sum(fs) / len(fs)
        out["false_headlines"] = {"pct_judged_accurate": _round(100 * pa, 2), "pct_would_share": _round(100 * ps, 2),
                                  "difference_points": _round(100 * (ps - pa), 2)}
    out["direction_matches_paper"] = bool(out["veracity_x_condition"]["beta"] is not None and out["veracity_x_condition"]["beta"] < 0)
    return out


def _study2(sessions):
    ss = _valid(sessions, "study2")
    have = {c: sum(s["condition"] == c for s in ss) for c in ("control", "treatment")}
    out = {"participants_per_condition": have}
    if have["control"] >= 2:
        out["control_discernment"] = _discernment(ss, "control")
    if have["treatment"] >= 2:
        out["treatment_discernment"] = _discernment(ss, "treatment")
    if min(have.values()) < 2:
        out.update(status="not_ready", reason="needs valid study2 sessions in control and treatment (need >= 2 each)")
        return out
    rows = _rows(ss, lambda c: 1.0 if c == "treatment" else 0.0, lambda v: (v - 1) / 5.0)
    try:
        X = [[1.0, 1.0 if r[2] else 0.0, r[3], (1.0 if r[2] else 0.0) * r[3]] for r in rows]
        beta, v, df, n, g1, g2 = _ols_cluster(X, [r[4] for r in rows], [r[0] for r in rows], [r[1] for r in rows])
    except ZeroDivisionError as exc:
        out.update(status="not_ready", reason="regression cannot run: %s" % exc)
        return out
    out.update(status="ok", ratings=n, participants=g1, headlines=g2, veracity_x_treatment=_term(beta, v, 3, df))
    c, t = out["control_discernment"]["discernment"], out["treatment_discernment"]["discernment"]
    out["discernment_ratio_treatment_over_control"] = _round(t / c, 3) if c and c > 0 else None
    out["direction_matches_paper"] = bool(beta[3] > 0)
    return out


def _study3(sessions):
    s1 = [s for s in _valid(sessions, "study1") if s["condition"] == "accuracy"]
    s2 = _valid(sessions, "study2")
    ctrl = [s for s in s2 if s["condition"] == "control"]
    treat = [s for s in s2 if s["condition"] == "treatment"]
    if not (s1 and ctrl and treat):
        return {"status": "not_ready", "reason": "needs valid study1 accuracy sessions and study2 control and treatment sessions"}

    def mean(ss, h, f=lambda v: v):
        vals = [f(s["responses"][h]) for s in ss if h in s["responses"]]
        return sum(vals) / len(vals)

    heads = sorted(s1[0]["responses"])
    xs = [mean(s1, h) for h in heads]
    ys = [mean(treat, h, lambda v: (v - 1) / 5.0) - mean(ctrl, h, lambda v: (v - 1) / 5.0) for h in heads]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx, syy = sum((x - mx) ** 2 for x in xs), sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return {"status": "not_ready", "reason": "no variance across headlines"}
    r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)
    df = len(heads) - 2
    t = r * math.sqrt(df / max(1e-12, 1 - r * r))
    return {"status": "ok", "r": _round(r), "df": df, "p": _round(_p_two_sided(t, df), 6)}


def evaluate(sessions):
    sessions = list(sessions or [])
    invalid = [s.get("participant") for s in sessions if not s.get("valid")]
    out = {"paper_reported": PAPER, "sessions": len(sessions), "invalid_sessions": len(invalid),
           "study1": _study1(sessions), "study2": _study2(sessions), "study3_items": _study3(sessions),
           "note": "A simulation of AI participants, shown next to the paper's reported results; "
                   "agreement would not show that people behave the same way. p values use a t distribution with "
                   "min(participant clusters, headline clusters) - 1 degrees of freedom."}
    states = [out[k]["status"] for k in ("study1", "study2", "study3_items")]
    out["status"] = "ok" if all(s == "ok" for s in states) else "not_ready" if all(s == "not_ready" for s in states) else "partial"
    return out
