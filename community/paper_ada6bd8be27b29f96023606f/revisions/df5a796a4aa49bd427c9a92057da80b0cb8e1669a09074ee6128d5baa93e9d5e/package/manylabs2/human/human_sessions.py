"""Human sessions for Many Labs 2 (Klein et al.), built from the study's public data files.

load(data_dir)    -> list of sessions in the shape the adapter's run_sessions returns (plus the generic keys
                     study / parsed / valid / source), one per participant and effect they took part in.
key_effects(ev)   -> the paper's key tests from evaluation.evaluate(sessions), signed so positive = as predicted.
"""
import csv
import math
import re
from pathlib import Path

SOURCE = (
    "ML2_S1.csv (Slate 1) and ML2_S2.csv (Slate 2) with ML2_Codebook_Primary_Analyses.xlsx, the Many Labs 2 primary-analysis "
    "data and codebook shared by the ManyLabs project (Klein et al., 2018, https://osf.io/8cd4r/), as supplied in human-data/. "
    "Both files are the aggregate replication samples, one row per participant."
)

EXCLUSIONS = (
    "Exclusions the paper and codebook state, applied per effect (a participant dropped from one effect stays in the others): "
    "Kay - any of kay*.4-6 missing; Rottenstreich, Bauer, Hsee, Knobe, Risen, Tversky, Shafir, Schwarz - needs the outcome "
    "item(s) of the condition seen; Critcher - estimate must lie in 0-100; Ross 1 and 2 - all three items answered and the "
    "percentage estimate(s) in 0-100; Hauser trolley 1 - under 4 s on the target scenario page excluded (the paper's aggregate "
    "N = 6,842 keeps participants with prior experience; the no-prior-experience restriction applies only to the comparison with "
    "the original and is not applied); Zaval - zav.include.strict (excludes unscrambling errors). "
    "No global exclusion on the attention checks (IMC1/IMC2) is stated for the primary analyses, so none is applied. "
    "Paper-only restrictions (university samples for Savani, English sites for Zaval's direct test, sample-size cuts per lab) "
    "are not applied."
)

NOT_MAPPED = [
    {"effect": e, "reason": r} for e, r in [
        ("huan", "click position on a map picture; evaluation does not compute it (not_ready) and the vertical click coordinate is outside the page model"),
        ("alt", "evaluation marks it not_ready (syllogism key and font manipulation absent); not mapped"),
        ("grah", "evaluation marks it not_ready (needs ideology-by-foundation correlation); not mapped"),
        ("inba", "evaluation marks it not_ready (needs disgust-scale scoring); not mapped"),
        ("miya", "evaluation marks it not_ready (covariate model); not mapped"),
        ("van", "evaluation marks it not_ready (SVO slider scoring); not mapped"),
        ("and", "evaluation marks it not_ready (composition of the well-being index); not mapped"),
        ("geis", "picture stimulus; evaluation marks it not_ready; not mapped"),
        ("haus34", "picture stimulus; evaluation marks it not_ready; not mapped"),
        ("sava", "evaluation marks it not_ready (hierarchical logistic model); not mapped"),
        ("nore", "picture stimulus; scoring key not applied by evaluation; not mapped"),
        ("gray", "picture stimulus; evaluation marks it not_ready; not mapped"),
        ("zhon", "evaluation marks it not_ready (product classification); not mapped"),
        ("gati", "evaluation marks it not_ready (country prominence list); not mapped"),
    ]
]

# effect -> (arm, [(condition, [pages])], is_numeric_page). Every page id is a codebook variable name in the CSV
# (codebook: "Variables in Dataset for Primary Analysis"); the column named in the comment is the human response.
# rott: rott1.1 certainty, rott2.1 1% chance (1 = movie star, 2 = $50)
# kay: kay1.4/5/6 structure condition, kay2.4/5/6 random condition (1-7)
# bau: bau1.1 consumer / bau2.1 individual condition flags; baudv.4 trust (1-7)
# crit: crit1.1 P97 / crit2.1 P17 % sold (pencil sites: crit1.1_1_TEXT / crit2.1_1_TEXT)
# ross.s1: ross.s1.1_1_TEXT % of peers who would sign (option 1), ross.s1.1_2_TEXT % who would refuse, ross.s1.2 own choice (1 sign, 2 refuse)
# ross.s2: ross.s2.1_1_TEXT % of peers who would pay the fine, ross.s2.1_2_TEXT % who would go to court, ross.s2.2 own choice (1 pay, 2 court)
# haus12: haus1.1 side-effect scenario, haus2.1 greater-good scenario (1 = permissible), haus1.2/haus2.2 prior experience (1 yes),
#         haus1.1t_3 / haus2.1t_3 page timing in seconds
# hsee: hsee1.1 scarf generosity, hsee2.1 coat generosity
# knob: knob1.3 help condition, knob2.3 harm condition intentionality (1-7)
# rise: rise1.3 tempting-fate, rise2.3 control likelihood of being called on (1-10)
# zav: zav.dv.3 concern (1-4); condition = which prime block was seen (zav1.* cold, zav2.* heat, via block timings
#      zav1.1t_1 / zav2.1t_1); zav.include.strict = no unscrambling errors
# tver: tver1.1 costly item on sale, tver2.1 cheap item on sale (1 = go to the other branch)
# schw: schw1.1 life, schw1.2 relationship (general first); schw2.1 relationship, schw2.2 life (specific first)
# shaf: shaf1.1 award, shaf2.1 deny (1 = parent A, 2 = parent B / extreme parent is code 2 per evaluation)
SPECS = {
    "rott": ("slate1", [("certain", ["rott1.1"]), ("lottery", ["rott2.1"])]),
    "kay": ("slate1", [("structure", ["kay1.4", "kay1.5", "kay1.6"]), ("random", ["kay2.4", "kay2.5", "kay2.6"])]),
    "bau": ("slate1", [("consumer", ["baudv.4"]), ("individual", ["baudv.4"])]),
    "crit": ("slate1", [("p97", ["crit1.1"]), ("p17", ["crit2.1"])]),
    "haus12": ("slate1", [("side", ["haus1.1", "haus1.2"]), ("means", ["haus2.1", "haus2.2"])]),
    "ross1": ("slate1", [("all", ["ross.s1.1", "ross.s1.2"])]),
    "ross2": ("slate2", [("all", ["ross.s2.1", "ross.s2.2"])]),
    "rise": ("slate2", [("tempt", ["rise1.3"]), ("notempt", ["rise2.3"])]),
    "knob": ("slate2", [("help", ["knob1.3"]), ("harm", ["knob2.3"])]),
    "hsee": ("slate2", [("scarf", ["hsee1.1"]), ("coat", ["hsee2.1"])]),
    "schw": ("slate2", [("general", ["schw1.1", "schw1.2"]), ("specific", ["schw2.1", "schw2.2"])]),
    "shaf": ("slate2", [("award", ["shaf1.1"]), ("deny", ["shaf2.1"])]),
    "tver": ("slate2", [("costly", ["tver1.1"]), ("cheap", ["tver2.1"])]),
    "zav": ("slate2", [("cold", ["zav.dv.3"]), ("heat", ["zav.dv.3"])]),
}
OPTION_PAGES = {  # pages with listed options in the adapter's materials: parse_reply sets "code"
    "rott1.1", "rott2.1", "kay1.4", "kay1.5", "kay1.6", "kay2.4", "kay2.5", "kay2.6", "baudv.4", "haus1.1", "haus1.2", "haus2.1",
    "haus2.2", "ross.s1.2", "ross.s2.2", "rise1.3", "rise2.3", "knob1.3", "knob2.3", "hsee1.1", "hsee2.1", "schw1.1", "schw1.2",
    "schw2.1", "schw2.2", "shaf1.1", "shaf2.1", "tver1.1", "tver2.1", "zav.dv.3",
}


def _num(v):
    if v is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(v).replace(",", ""))
    return float(m.group()) if m else None


def _has(row, col):
    return row.get(col, "") not in ("", "NA", None)


def _parsed(page, raw):
    n = _num(raw)
    code = int(n) if page in OPTION_PAGES and n is not None and n == int(n) else None
    return {"raw": str(raw).strip(), "code": code, "number": n}


def _read(path):
    csv.field_size_limit(10 ** 9)
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def _condition(effect, row):
    """The condition this participant saw, or None if they did not take part in the effect."""
    if effect == "rott":
        return "certain" if _has(row, "rott1.1") else "lottery" if _has(row, "rott2.1") else None
    if effect == "kay":
        return "structure" if _has(row, "kay1.4") else "random" if _has(row, "kay2.4") else None
    if effect == "bau":
        return "consumer" if _has(row, "bau1.1") else "individual" if _has(row, "bau2.1") else None
    if effect == "crit":
        return "p97" if _has(row, "crit1.1") or _has(row, "crit1.1_1_TEXT") else "p17" if _has(row, "crit2.1") or _has(row, "crit2.1_1_TEXT") else None
    if effect == "haus12":
        return "side" if _has(row, "haus1.1") else "means" if _has(row, "haus2.1") else None
    if effect in ("ross1", "ross2"):
        return "all" if _has(row, "ross.s1.2" if effect == "ross1" else "ross.s2.2") else None
    if effect == "rise":
        return "tempt" if _has(row, "rise1.3") else "notempt" if _has(row, "rise2.3") else None
    if effect == "knob":
        return "help" if _has(row, "knob1.3") else "harm" if _has(row, "knob2.3") else None
    if effect == "hsee":
        return "scarf" if _has(row, "hsee1.1") else "coat" if _has(row, "hsee2.1") else None
    if effect == "schw":
        return "general" if _has(row, "schw1.1") or _has(row, "schw1.2") else "specific" if _has(row, "schw2.1") or _has(row, "schw2.2") else None
    if effect == "shaf":
        return "award" if _has(row, "shaf1.1") else "deny" if _has(row, "shaf2.1") else None
    if effect == "tver":
        return "costly" if _has(row, "tver1.1") else "cheap" if _has(row, "tver2.1") else None
    if effect == "zav":
        if not _has(row, "zav.dv.3"):
            return None
        return "cold" if _has(row, "zav1.1t_1") else "heat" if _has(row, "zav2.1t_1") else None
    return None


def _answers(effect, cond, row):
    """{page: raw value} for the effect, or None if an exclusion applies or a needed item is missing."""
    if effect == "crit":
        col = "crit1.1" if cond == "p97" else "crit2.1"
        v = row.get(col) if _has(row, col) else row.get(col + "_1_TEXT")
        n = _num(v)
        if n is None or not 0 <= n <= 100:
            return None
        return {col: v}
    if effect in ("ross1", "ross2"):
        k = "ross.s1" if effect == "ross1" else "ross.s2"
        e1, e2, ch = row.get(k + ".1_1_TEXT"), row.get(k + ".1_2_TEXT"), row.get(k + ".2")
        ns = [_num(e1), _num(e2), _num(ch)]
        if any(n is None for n in ns) or not (0 <= ns[0] <= 100 and 0 <= ns[1] <= 100):
            return None
        return {k + ".1": e1, k + ".2": ch}
    pages = dict(SPECS[effect][1])[cond]
    if effect == "haus12":
        t = _num(row.get("haus1.1t_3" if cond == "side" else "haus2.1t_3"))
        if t is not None and t < 4:
            return None
        if not _has(row, pages[0]):
            return None
        return {p: row[p] for p in pages if _has(row, p)}
    if effect == "zav" and row.get("zav.include.strict") != "TRUE":
        return None
    if effect == "schw":
        if not all(_has(row, p) for p in pages):
            return None
    out = {}
    for p in pages:
        if not _has(row, p):
            return None
        out[p] = row[p]
    return out


def load(data_dir):
    """Sessions for every participant and effect; see module docstring."""
    d = Path(data_dir)
    files = {"slate1": d / "ML2_S1.csv", "slate2": d / "ML2_S2.csv"}
    out = []
    for arm, path in files.items():
        for i, row in enumerate(_read(path)):
            pid = row.get("ResponseID") or row.get(".id") or str(i)
            for effect, (earm, _) in SPECS.items():
                if earm != arm:
                    continue
                cond = _condition(effect, row)
                if cond is None:
                    continue
                raw = _answers(effect, cond, row)
                if raw is None:
                    continue
                parsed = {p: _parsed(p, v) for p, v in raw.items()}
                out.append({"study": effect, "condition": cond, "arm": arm, "participant": pid, "parsed": parsed, "valid": True,
                            "source": "human", "effect": effect, "position": None, "answers": parsed})
    return out


# ---- key effects -------------------------------------------------------------------------------------------------------

def _norm_p(z):
    return math.erfc(abs(z) / math.sqrt(2)) if z is not None else None


def _betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
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
        if abs(delta - 1) < 1e-12:
            break
    return h


def _t_p(t, df):
    """Two-sided p of Student t via the regularized incomplete beta."""
    if t is None or df is None or df <= 0:
        return None
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x)) if 0 < x < 1 else (1.0 if x >= 1 else 0.0)
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1 - front * _betacf(b, a, 1 - x) / b


def _d_from_props(r):
    p1, p2, n1, n2 = r["p_first"], r["p_second"], r["n"][0], r["n"][1]
    if min(p1, p2) <= 0 or max(p1, p2) >= 1:
        return None
    return math.log((p1 / (1 - p1)) / (p2 / (1 - p2))) * math.sqrt(3) / math.pi


# id -> (effect, label, reported value in the paper's own units)
_KEY = [
    ("kay", "Kay: structure vs random event, willingness to pursue goal (d)", -0.02),
    ("rott", "Rottenstreich & Hsee: affect-rich choice, 1% chance vs certain (d from odds ratio)", -0.08),
    ("bau", "Bauer: trust in others, individual minus consumer wording (d)", 0.12),
    ("crit", "Critcher & Gilovich: sales estimate, P97 minus P17 (d)", 0.04),
    ("haus12", "Hauser: side-effect vs greater-good permissibility (d from odds ratio)", 1.35),
    ("ross1", "Ross: false consensus, supermarket (d)", 1.18),
    ("ross2", "Ross: false consensus, traffic ticket (d)", 0.95),
    ("tver", "Tversky & Kahneman: framing, cheap vs costly item on sale (d from odds ratio)", 0.40),
    ("rise", "Risen & Gilovich: tempting fate raises likelihood of being called on (d)", 0.18),
    ("hsee", "Hsee: less is better, scarf minus coat generosity (d)", 0.78),
    ("schw", "Schwarz: assimilation/contrast, specific-first minus general-first correlation (Cohen's q)", -0.07),
    ("shaf", "Shafir: P(award extreme) + P(deny extreme) - 1, converted to d from the z test", -0.13),
    ("zav", "Zaval: heat minus cold priming, concern about global warming (d)", -0.03),
    ("knob", "Knobe: harm minus help condition, intentionality (d)", 1.75),
]
_IDS = {"kay": "kay_1", "rott": "rott_1", "bau": "bau_1", "crit": "anchoring_1", "haus12": "haus12_1", "ross1": "ross1_1",
        "ross2": "ross2_1", "tver": "tver_1", "rise": "rise_1", "hsee": "hsee_1", "schw": "schw_1", "shaf": "shaf_1",
        "zav": "zav_1", "knob": "knob_1"}


def key_effects(evaluation):
    out = []
    for eff, label, reported in _KEY:
        r = (evaluation or {}).get(eff) or {}
        est = p = None
        if r.get("status") == "computed":
            if eff in ("rott", "tver", "haus12"):
                est = _d_from_props(r)
                p = _norm_p(r.get("z"))
            elif eff == "schw":
                est, p = r.get("q"), _norm_p(r.get("z"))
            elif eff == "shaf":
                n = sum(r["n"])
                z = r.get("z")
                if z is not None and n:
                    rr = z / math.sqrt(n)
                    est = 2 * rr / math.sqrt(1 - rr * rr)
                    p = _norm_p(z)
            else:
                est = r.get("d")
                p = _t_p(r.get("t"), sum(r["n"]) - 2) if r.get("t") is not None else None
        out.append({"id": _IDS[eff], "label": label, "estimate": est, "p": p, "reported": reported})
    return out
