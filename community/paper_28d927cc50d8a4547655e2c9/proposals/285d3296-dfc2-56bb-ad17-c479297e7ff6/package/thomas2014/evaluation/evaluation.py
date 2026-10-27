#!/usr/bin/env python3
"""evaluate(sessions) -> dict: the paper's own comparisons computed from run_sessions output alone.

Tests (uncorrected Pearson chi-square, phi = sqrt(chi2/N), as reported in Tables 1-2):
  Experiments 1-2: omnibus (df 3) and adjacent knowledge-level comparisons within each payoff, payoff invariance
  within each knowledge level, expected earnings. Experiment 3: knowledge comparisons collapsed over similarity,
  similarity checks, confusion matrix. Agreeableness/Openness analyses return not_ready (Big Five items missing).
Participants whose payoff comprehension answers are not all correct are excluded (assumption, issue iss_e1_excl);
results with no exclusion are also given.
"""
import math

KNOW = ["primary", "secondary", "tertiary", "common"]
PAY = [1, 2, 5, 10]
PAPER = {
    "exp1_table1_chi2": {"all": ["63.70***", "62.79***", "56.71***", "70.41***"], "1o_vs_2o": ["21.26***", "23.27***", "2.66", "19.06***"],
                         "2o_vs_3o": ["0.69", "0.63", "10.88***", "0.01"], "3o_vs_ck": ["8.84**", "7.12**", "4.02*", "23.05***"]},
    "exp2_table2_chi2": {"all": ["68.24***", "45.32***", "46.36***", "24.72***"], "1o_vs_2o": ["13.59***", "14.95***", "1.77", "11.54***"],
                         "2o_vs_3o": ["2.50", "0.69", "2.26", "0.03"], "3o_vs_ck": ["10.14**", "6.21*", "12.39***", "4.23*"]},
    "exp3": {"common_vs_tertiary": "chi2(1,N=270)=22.28", "secondary_vs_primary": "chi2(1,N=280)=16.87", "secondary_vs_tertiary": "chi2(1,N=284)=0.72",
             "table3_tertiary_as_secondary": 0.230, "table3_diagonal": {"primary": .931, "secondary": .899, "tertiary": .637, "common": .837}},
}


def _p(chi2, df):
    if df == 1:
        return math.erfc(math.sqrt(chi2 / 2))
    if df == 3:
        return math.erfc(math.sqrt(chi2 / 2)) + math.sqrt(2 * chi2 / math.pi) * math.exp(-chi2 / 2)
    return None


def chi_square(groups):
    """groups: list of (together, total). Pearson chi-square of the 2 x k table."""
    total = sum(t for _, t in groups)
    yes = sum(y for y, _ in groups)
    if total == 0 or yes in (0, total) or any(t == 0 for _, t in groups):
        return {"chi2": None, "df": len(groups) - 1, "n": total, "p": None, "phi": None, "note": "undefined (empty or constant table)"}
    chi2 = 0.0
    for y, t in groups:
        for obs, exp in ((y, t * yes / total), (t - y, t * (total - yes) / total)):
            chi2 += (obs - exp) ** 2 / exp
    df = len(groups) - 1
    return {"chi2": round(chi2, 3), "df": df, "n": total, "p": round(_p(chi2, df), 4), "phi": round(math.sqrt(chi2 / total), 3)}


def _count(sessions, **kw):
    sel = [s for s in sessions if all(s["condition"].get(k) == v for k, v in kw.items())]
    return sum(1 for s in sel if s.get("decision") == "together"), len(sel)


def _kept(sessions, exclude):
    if not exclude:
        return sessions
    return [s for s in sessions if s["study"] == "exp3" or s.get("payoff_correct")]


def _knowledge_tests(sessions, study, **extra):
    cells = {k: _count([s for s in sessions if s["study"] == study], knowledge=k, **extra) for k in KNOW}
    out = {"proportions": {k: (round(y / t, 3) if t else None) for k, (y, t) in cells.items()}, "counts": cells,
           "all_levels": chi_square(list(cells.values()))}
    for name, (a, b) in (("1o_vs_2o", ("primary", "secondary")), ("2o_vs_3o", ("secondary", "tertiary")), ("3o_vs_ck", ("tertiary", "common"))):
        out[name] = chi_square([cells[a], cells[b]])
    return out


def _two_player_study(sessions, study):
    rows = [s for s in sessions if s["study"] == study]
    if not rows:
        return {"status": "not_ready", "missing": f"no {study} sessions"}
    res = {"by_payoff": {}, "payoff_invariance": {}, "expected_earnings_pct_of_max": {}}
    for p in PAY:
        res["by_payoff"][f"${p}"] = _knowledge_tests(rows, study, payoff=p)
    for k in KNOW:
        res["payoff_invariance"][k] = chi_square([_count(rows, knowledge=k, payoff=p) for p in PAY])
    exponent = 2 if study == "exp1" else 4  # four-person product is an assumption (issue iss_e2_edef)
    for k in KNOW:
        res["expected_earnings_pct_of_max"][k] = {}
        for p in PAY:
            y, t = _count(rows, knowledge=k, payoff=p)
            if t:
                q = y / t
                risky = p + 0.10
                res["expected_earnings_pct_of_max"][k][f"${p}"] = round(100 * ((1 - q) * p + (q ** exponent) * risky) / risky, 1)
    return res


def _confusion(rows):
    table = {}
    for k in KNOW:
        sel = [s for s in rows if s["condition"]["knowledge"] == k]
        counts = {lvl: sum(1 for s in sel if s.get("reported_level") == lvl) for lvl in KNOW + ["unclassifiable"]}
        table[k] = {lvl: (round(c / len(sel), 3) if sel else None) for lvl, c in counts.items()}
    return table


def _welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    return None if se == 0 else round((ma - mb) / se, 3)


def _wald_logistic(x, y):
    """Wald chi-square for one predictor in a logistic regression with intercept (Newton-Raphson)."""
    if len(x) < 5 or len(set(y)) < 2 or len(set(x)) < 2:
        return None
    b0 = b1 = 0.0
    for _ in range(50):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for xi, yi in zip(x, y):
            p = 1 / (1 + math.exp(-(b0 + b1 * xi)))
            w = p * (1 - p)
            g0 += yi - p; g1 += (yi - p) * xi
            h00 += w; h01 += w * xi; h11 += w * xi * xi
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            return None
        d0, d1 = (h11 * g0 - h01 * g1) / det, (h00 * g1 - h01 * g0) / det
        b0 += d0; b1 += d1
        if abs(d0) + abs(d1) < 1e-8:
            break
    se = math.sqrt(h00 / det)
    z = b1 / se
    return {"wald_chi2": round(z * z, 3), "p": round(_p(z * z, 1), 4)}


def _exp3(sessions):
    rows = [s for s in sessions if s["study"] == "exp3"]
    if not rows:
        return {"status": "not_ready", "missing": "no exp3 sessions"}
    res = {"knowledge_collapsed_over_similarity": _knowledge_tests(rows, "exp3"), "confusion_matrix": _confusion(rows)}
    sim = [s["perceived_similarity"] for s in rows if s["condition"]["similarity"] == "similar" and s.get("perceived_similarity") is not None]
    dis = [s["perceived_similarity"] for s in rows if s["condition"]["similarity"] == "dissimilar" and s.get("perceived_similarity") is not None]
    res["similarity"] = {"perceived_similarity_t_similar_minus_dissimilar": _welch(sim, dis),
                         "decision_by_similarity_condition": chi_square([_count(rows, similarity="similar"), _count(rows, similarity="dissimilar")]),
                         "decision_on_perceived_similarity": _wald_logistic([s["perceived_similarity"] for s in rows if s.get("perceived_similarity") is not None and s.get("decision")],
                                                                        [1 if s["decision"] == "together" else 0 for s in rows if s.get("perceived_similarity") is not None and s.get("decision")])}
    res["tertiary_mistaken_for_secondary"] = res["confusion_matrix"]["tertiary"].get("secondary")
    res["personality"] = {"status": "not_ready", "missing": "The 50 Big Five items, response scale and scoring are not in the paper (issues iss_e3_big5, iss_e3_big5score); Agreeableness and Openness analyses cannot run."}
    return res


def evaluate(sessions):
    if not sessions:
        return {"status": "not_ready", "missing": "no sessions"}
    out = {"status": "ok", "note": "Simulation output; agreement with the paper does not show people would behave the same way.", "paper_reported": PAPER}
    for label, exclude in (("with_comprehension_exclusion", True), ("no_exclusion", False)):
        kept = _kept(sessions, exclude)
        out[label] = {"n_sessions": len(kept), "exp1": _two_player_study(kept, "exp1"), "exp2": _two_player_study(kept, "exp2"), "exp3": _exp3(kept)}
    out["status"] = "partial"
    out["not_ready"] = ["exp3 Agreeableness and Openness analyses (Big Five items missing)"]
    return out


if __name__ == "__main__":
    import json, sys
    print(json.dumps(evaluate(json.load(open(sys.argv[1]))), indent=1) if len(sys.argv) > 1 else __doc__)
