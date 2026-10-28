"""Evaluation for Many Labs 2 replays: evaluate(sessions) -> dict, derived from the sessions alone.

Each effect returns {'status': 'computed' | 'not_ready' | 'blocked' | 'no_data', ...}. Only effects whose outcome and test are
fully stated by the paper and the codebook are computed; the others return not_ready with what is missing.
Statistics use the standard library only.
"""
import math
from statistics import mean, variance


def _welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = variance(a), variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    sp = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if se == 0 or sp == 0:
        return {"t": None, "d": None, "n": [len(a), len(b)], "means": [mean(a), mean(b)]}
    return {"t": (mean(a) - mean(b)) / se, "d": (mean(a) - mean(b)) / sp, "n": [len(a), len(b)], "means": [mean(a), mean(b)]}


def _prop(a, b):
    """a, b: lists of 0/1. Two-proportion z test and odds ratio (first minus second)."""
    if not a or not b:
        return None
    pa, pb, na, nb = mean(a), mean(b), len(a), len(b)
    p = (sum(a) + sum(b)) / (na + nb)
    se = math.sqrt(p * (1 - p) * (1 / na + 1 / nb))
    z = (pa - pb) / se if se else None
    return {"p_first": pa, "p_second": pb, "z": z, "n": [na, nb]}


def _get(sessions, effect, cond, page, field):
    out = []
    for s in sessions:
        if s.get("effect") == effect and s.get("condition") == cond and page in (s.get("answers") or {}):
            v = s["answers"][page].get(field)
            if v is not None:
                out.append(v)
    return out


def _per_participant(sessions, effect, cond, pages, field="code"):
    out = []
    for s in sessions:
        if s.get("effect") == effect and s.get("condition") == cond:
            vals = [(s.get("answers") or {}).get(p, {}).get(field) for p in pages]
            if vals and all(v is not None for v in vals):
                out.append(mean(vals))
    return out


def _pearson(x, y):
    if len(x) < 3:
        return None
    mx, my = mean(x), mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else None


# effect -> (kind, spec). Pages are codebook item ids.
CONT = {  # continuous outcome, difference = first condition minus second (the direction of the original finding)
    "bau": ("individual", "consumer", ["baudv.4"], "number", "trust is lower for consumer wording; d reported as individual minus consumer"),
    "kay": ("structure", "random", None, "number", "willingness higher after structure"),
    "hsee": ("scarf", "coat", ["hsee1.1"], "code", "scarf gift judged more generous"),
    "knob": ("harm", "help", None, "code", "harm judged more intentional"),
    "rise": ("tempt", "notempt", None, "number", "higher likelihood of being called on when tempting fate"),
    "crit": ("p97", "p17", ["crit1.1"], "number", "higher estimate with the high anchor"),
    "zav": ("heat", "cold", ["zav.dv.3"], "code", "more concern after heat priming"),
}
PAGES = {  # per-condition outcome pages
    "kay": {"structure": ["kay1.5", "kay1.6"], "random": ["kay2.5", "kay2.6"]},
    "hsee": {"scarf": ["hsee1.1"], "coat": ["hsee2.1"]},
    "knob": {"harm": ["knob2.3"], "help": ["knob1.3"]},
    "rise": {"tempt": ["rise1.3"], "notempt": ["rise2.3"]},
    "crit": {"p97": ["crit1.1"], "p17": ["crit2.1"]},
    "bau": {"individual": ["baudv.4"], "consumer": ["baudv.4"]},
    "zav": {"heat": ["zav.dv.3"], "cold": ["zav.dv.3"]},
}
BINARY = {  # effect: (cond_a, page_a, cond_b, page_b, code counted as 1)
    "rott": ("lottery", "rott2.1", "certain", "rott1.1", 1),
    "tver": ("cheap", "tver2.1", "costly", "tver1.1", 1),
    "haus12": ("side", "haus1.1", "means", "haus2.1", 1),
}
NOT_READY = {
    "huan": "the response is a click on the map picture and the codebook gives no coordinate format; test not implemented here",
    "geis": "picture stimulus is supplied, but the test needs the five power items and a vision-capable participant; not implemented here",
    "haus34": "diagram is supplied, but which scenario (Ned or Oscar) is the greater-good one is not stated (issue iss-map-haus34); test not implemented here",
    "gray": "pictures are supplied, but the test is not implemented here and needs a vision-capable participant",
    "zhon": "which of the ten rated products are the cleansing products is not stated in the paper or the codebook (issue iss-score-zhon)",
    "alt": "syllogism answer key and the font manipulation are not available to a text replay",
    "grah": "needs the ideology-by-foundation correlation with foundation coding of items from the paper (binding items 8-13 of the codebook); not implemented here",
    "inba": "needs DS-R scoring and a difference between two correlations; scoring key not stated in the paper",
    "miya": "paper's test controls for perceived constraint (covariate model) which is not implemented here",
    "van": "scoring is supplied (Murphy et al. 2011 angle from the six slider items), but the six slider choices sit on one page and the adapter parses a single choice per page; test not implemented here",
    "and": "composition of the subjective well-being score is not stated in the paper",
    "sava": "hierarchical logistic model with trial-level importance is not implemented here",
}


# Scoring keys from sources/ML2_Codebook_Primary_Analyses.xlsx (program evidence ev-cb-nore-key, ev-cb-gati-prom).
NORE_KEY = {k: (1 if k % 2 else 2) for k in range(1, 21)}  # rule-based group: odd items group 1, even items group 2, in both conditions
GATI_PROM_FIRST = {"gati1": {2, 3, 5, 11, 12, 14, 16, 17, 18, 19, 22}, "gati2": {4, 6, 7, 8, 9, 10, 13, 15, 20, 21}}  # items where the prominent country is shown first
GATI_PROM_SECOND = {"gati1": {4, 6, 7, 8, 9, 10, 13, 15, 20, 21}, "gati2": {2, 3, 5, 11, 12, 14, 16, 17, 18, 19, 22}}


def _nore(sessions):
    by = {"belong": [], "similar": []}
    for s in sessions:
        if s.get("effect") != "nore" or s.get("condition") not in by:
            continue
        pre = "nore1." if s["condition"] == "belong" else "nore2."
        a = s.get("answers") or {}
        codes = [a.get(pre + str(k), {}).get("code") for k in range(1, 21)]
        if all(c is not None for c in codes):
            by[s["condition"]].append(100.0 * sum(1 for k, c in enumerate(codes, 1) if c == NORE_KEY[k]) / 20)
    r = _welch(by["belong"], by["similar"])
    return {"status": "computed", "test": "Welch t on percentage of rule-based responses (belong minus similar)", **r} if r else {"status": "no_data", "reason": "need at least two participants per condition"}


def _gati(sessions):
    scores = {"sim": [], "diff": []}
    for s in sessions:
        c = s.get("condition") or ""
        if s.get("effect") != "gati" or c[:3] not in ("sim", "dif"):
            continue
        kind, ver = ("sim", c[3:]) if c.startswith("sim") else ("diff", c[4:])
        pre = "gati%s%s." % (ver, "s" if kind == "sim" else "d")
        a = s.get("answers") or {}
        first = [a.get(pre + str(k), {}).get("code") for k in sorted(GATI_PROM_FIRST["gati" + ver])]
        second = [a.get(pre + str(k), {}).get("code") for k in sorted(GATI_PROM_SECOND["gati" + ver])]
        if first and second and all(v is not None for v in first + second):
            scores[kind].append(mean(second) - mean(first))
    out = {}
    for kind, x in scores.items():
        if len(x) < 2:
            out[kind] = {"status": "no_data", "reason": "need at least two participants"}
        else:
            sd = math.sqrt(variance(x))
            out[kind] = {"mean_asymmetry": mean(x), "t": mean(x) / (sd / math.sqrt(len(x))) if sd else None, "n": len(x)}
    return {"status": "computed", "test": "one-sample t of the asymmetry score (mean rating with the prominent country second minus first) against 0", **out}


def _ross(sessions, effect, est_page, choice_page):
    first, second = [], []
    for s in sessions:
        if s.get("effect") != effect:
            continue
        a = s.get("answers") or {}
        est, ch = a.get(est_page, {}).get("number"), a.get(choice_page, {}).get("code")
        if est is None or ch is None or not 0 <= est <= 100:
            continue
        (first if ch == 1 else second).append(est)
    r = _welch(first, second)
    return {"status": "computed", "test": "Welch t on estimated % choosing option 1, by own choice", **r} if r else {"status": "no_data", "reason": "need at least two participants per own-choice group"}


def _schw(sessions):
    r = {}
    for cond, pages in (("specific", ("schw2.1", "schw2.2")), ("general", ("schw1.2", "schw1.1"))):  # (relationship, life)
        x, y = [], []
        for s in sessions:
            if s.get("effect") == "schw" and s.get("condition") == cond:
                a = s.get("answers") or {}
                u, v = a.get(pages[0], {}).get("code"), a.get(pages[1], {}).get("code")
                if u is not None and v is not None:
                    x.append(u); y.append(v)
        r[cond] = (_pearson(x, y), len(x))
    (r1, n1), (r2, n2) = r["specific"], r["general"]
    if r1 is None or r2 is None or min(n1, n2) < 4 or abs(r1) >= 1 or abs(r2) >= 1:
        return {"status": "no_data", "reason": "need correlations in both order conditions"}
    q = math.atanh(r1) - math.atanh(r2)
    z = q / math.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return {"status": "computed", "test": "Fisher z test of the difference between correlations (specific-first minus general-first)", "r_specific_first": r1, "r_general_first": r2, "q": q, "z": z, "n": [n1, n2]}


def _shaf(sessions):
    a = [1 if v == 2 else 0 for v in _get(sessions, "shaf", "award", "shaf1.1", "code")]
    d = [1 if v == 2 else 0 for v in _get(sessions, "shaf", "deny", "shaf2.1", "code")]
    if len(a) < 2 or len(d) < 2:
        return {"status": "no_data", "reason": "need participants in both conditions"}
    pa, pd = mean(a), mean(d)
    se = math.sqrt(pa * (1 - pa) / len(a) + pd * (1 - pd) / len(d))
    return {"status": "computed", "test": "sum of P(award extreme parent) and P(deny extreme parent) against 1", "p_award": pa, "p_deny": pd, "sum": pa + pd, "z": (pa + pd - 1) / se if se else None, "n": [len(a), len(d)]}


def evaluate(sessions):
    out = {}
    runs = [s for s in sessions if s.get("status") != "not_run"]
    for s in sessions:
        if s.get("status") == "not_run":
            out[s["effect"]] = {"status": "blocked", "reason": s["reason"]}
    effects = sorted({s["effect"] for s in runs if s.get("effect") and s["effect"] != "_after"})
    for k in effects:
        if k in out:
            continue
        if k in NOT_READY:
            out[k] = {"status": "not_ready", "reason": NOT_READY[k]}
        elif k in BINARY:
            ca, pa, cb, pb, one = BINARY[k]
            a = [1 if v == one else 0 for v in _get(runs, k, ca, pa, "code")]
            b = [1 if v == one else 0 for v in _get(runs, k, cb, pb, "code")]
            r = _prop(a, b)
            out[k] = {"status": "computed", "test": "two-proportion z (%s minus %s)" % (ca, cb), **r} if r else {"status": "no_data", "reason": "need participants in both conditions"}
        elif k in CONT:
            first, second, _, field, note = CONT[k]
            pg = PAGES[k]
            a = _per_participant(runs, k, first, pg[first], "number" if field == "number" else "code")
            b = _per_participant(runs, k, second, pg[second], "number" if field == "number" else "code")
            r = _welch(a, b)
            out[k] = {"status": "computed", "test": "Welch t (%s minus %s); %s" % (first, second, note), **r} if r else {"status": "no_data", "reason": "need at least two participants per condition"}
        elif k == "ross1":
            out[k] = _ross(runs, k, "ross.s1.1", "ross.s1.2")
        elif k == "ross2":
            out[k] = _ross(runs, k, "ross.s2.1", "ross.s2.2")
        elif k == "nore":
            out[k] = _nore(runs)
        elif k == "gati":
            out[k] = _gati(runs)
        elif k == "schw":
            out[k] = _schw(runs)
        elif k == "shaf":
            out[k] = _shaf(runs)
        else:
            out[k] = {"status": "not_ready", "reason": "no evaluation specified"}
    out.setdefault("zhon", {"status": "not_ready", "reason": NOT_READY["zhon"]})
    out["_note"] = "Simulation output next to the paper's reported results (program.json result nodes); agreement would not show that people behave the same way."
    return out


if __name__ == "__main__":
    print(evaluate([]))
