#!/usr/bin/env python3
"""Many Labs 1 human data -> sessions in the shape run_sessions returns (standard library only).

Codebook entries are cited as [codebook: <variable>]. The file is one row per participant (session_id),
each of whom took all 11 text studies in a random order; the order is not used here.
"""
import csv
import math
from pathlib import Path

SOURCE = ("human-data/hd/Tab.delimited.Cleaned.dataset.WITH.variable.values.dat (Many Labs 1 cleaned data, "
          "CC0, OSF https://osf.io/wx7ck/), variables described in human-data/hd/Codebook.xlsx; reference "
          "effect sizes from human-data/hd/ML_Summary_Statistics.xlsx and Table 2 of Klein et al. (2014)")

EXCLUSIONS = (
    "As stated in the paper/supplement: (1) Flag priming (s09): US samples only (us_or_international == 0) and "
    "participants who gave a time-of-day estimate for all four photos (flagfilter == 1; 82 dropped), leaving 4,897. "
    "(2) Currency priming (s10): only participants who answered at least 6 of the 8 system-justification items "
    "(11 dropped). (3) Gambler's fallacy (s04): the evaluation applies the paper's 3-SD outlier rule itself. "
    "(4) All other studies: no participant excluded by the authors; a participant with no valid answer on a study "
    "has no session for that study. Not stated by the paper but forced by the adapter's shape: anchoring (s03) "
    "keeps only participants who answered all four estimates AND were assigned the same (low or high) anchor on "
    "all four, because the data randomise the anchor per question while a session carries one condition."
)

NOT_MAPPED = [
    {"arm": "s12", "reason": "IAT reaction-time latencies are not a text response and the adapter has no parser for them."},
    {"arm": "s03 (participants with mixed anchors)", "reason": "The anchor was randomised per question (anch1group..anch4group differ "
     "within a participant) but a session holds one condition; those participants are left out (see EXCLUSIONS)."},
]

_LOW = ["Up to a half hour", "Half an hour to an hour", "One to one and a half hours",
        "One and a half to two hours", "Two to two and a half hours", "More than two and a half hours"]
_HIGH = ["Up to two and a half hours", "Two and a half to three hours", "Three to three and a half hours",
         "Three and a half to four hours", "Four to four and a half hours", "More than four and a half hours"]


def _f(x):
    try:
        v = float(str(x).strip())
    except (TypeError, ValueError):
        return None
    return v if not math.isnan(v) else None


def _session(study, cond, pid, parsed):
    # Flat fields as specified; "responses" repeats the one response in the layout evaluate() reads
    # (run_sessions nests responses under a participant), so the package's evaluation runs on these unchanged.
    arm = f"{study}:{cond}"
    return {"study": study, "condition": cond, "arm": arm, "participant": pid, "parsed": parsed,
            "valid": parsed is not None, "source": "human",
            "responses": [{"study": study, "condition": cond, "arm": arm, "parsed": parsed}]}


def _build(r):
    pid = r["session_id"]
    out = []
    # s01 sunk costs. [codebook: sunkcosta] paid ticket, [sunkcostb] free ticket, 1-9; [sunkDV] copy of the answered one.
    # Group coding (checked against which of a/b is filled): sunkgroup 0 = paid, 1 = free.
    v = _f(r["sunkDV"])
    if v is not None and 1 <= v <= 9:
        out.append(_session("s01", "paid" if r["sunkgroup"] == "0" else "free", pid, {"value": v}))
    # s02 framing. [codebook: diseaseframinga] gain frame 1 = "200 saved" (sure), 2 = gamble;
    # [diseaseframingb] loss frame 1 = "400 die" (sure), 2 = gamble. [gainlossDV] copy; gainlossgroup 1 = gain, 0 = loss.
    v = _f(r["gainlossDV"])
    if v in (1.0, 2.0):
        gain = r["gainlossgroup"] == "1"
        sure = v == 1.0
        label = ("A" if sure else "B") if gain else ("C" if sure else "D")
        out.append(_session("s02", "gain" if gain else "loss", pid, {"choice": label, "sure": sure}))
    # s03 anchoring. [codebook: anchoring1..anchoring4] NYC miles, Chicago population, Everest feet, babies/day
    # (cleaned, common units); anch{k}group 0 = low anchor (anchoring{k}a), 1 = high (anchoring{k}b).
    vals = [_f(r[f"anchoring{k}"]) for k in range(1, 5)]
    groups = {r[f"anch{k}group"] for k in range(1, 5)}
    if all(x is not None for x in vals) and len(groups) == 1 and groups <= {"0", "1"}:
        out.append(_session("s03", "low" if groups == {"0"} else "high", pid, {"values": vals}))
    # s04 gambler's fallacy. [codebook: gamblerfallacya] three 6s, [gamblerfallacyb] two 6s and a 3; the file stores
    # them (and [gambfalDV]) already square-rooted, and the evaluation takes the square root itself, so the
    # earlier count is gambfalDV squared. gambfalgroup 1 = three 6s, 0 = two 6s and a 3.
    v = _f(r["gambfalDV"])
    if v is not None and v >= 0:
        out.append(_session("s04", "three6" if r["gambfalgroup"] == "1" else "two6one3", pid, {"values": [v * v]}))
    # s05 category scales. [codebook: scalesa] low-frequency scale, [scalesb] high-frequency scale, categories 1-6
    # (low: 6 = more than 2.5 h; high: 1 = up to 2.5 h). [scales] = 1 if over 2.5 h; scalesgroup 0 = low, 1 = high.
    # The label is the printed category text for that scale, as parse_reply returns it.
    cat = _f(r["scalesa"]) if r["scalesgroup"] == "0" else _f(r["scalesb"])
    if cat is not None and 1 <= cat <= 6 and r["scalesgroup"] in ("0", "1"):
        low = r["scalesgroup"] == "0"
        out.append(_session("s05", "low" if low else "high", pid, {"choice": (_LOW if low else _HIGH)[int(cat) - 1]}))
    # s06 reciprocity. [codebook: reciprocityusa] NK reporters into US asked first, [reciprocityusb] asked second;
    # 1 = Yes, 2 = No. us_first = American-reporters question asked first (the NK-reporters question second).
    # reciprocitygroup 1 = NK-reporters question first (nk_first), 0 = us_first.
    nk_first = r["reciprocitygroup"] == "1"
    v = _f(r["reciprocityusa"] if nk_first else r["reciprocityusb"])
    if v in (1.0, 2.0):
        out.append(_session("s06", "nk_first" if nk_first else "us_first", pid, {"yes": v == 1.0}))
    # s07 allowed/forbidden. [codebook: allowedforbiddena] forbid wording, [allowedforbiddenb] allow wording; 1 = Yes, 2 = No.
    # allowedforbiddenGroup 1 = forbid, 0 = allow.
    forbid = r["allowedforbiddenGroup"] == "1"
    v = _f(r["allowedforbiddena"] if forbid else r["allowedforbiddenb"])
    if v in (1.0, 2.0):
        out.append(_session("s07", "forbid" if forbid else "allow", pid, {"yes": v == 1.0}))
    # s08 quote. [codebook: quotea] Washington, [quoteb] Bin Laden (1 = strongly agree .. 9 = strongly disagree);
    # [quote] is the recoded agreement score (= 10 - raw); the session holds the raw printed 1-9 value
    # (10 - quote) because the evaluation reverses it itself.
    # quoteGroup 1 = liked (Washington), 0 = disliked (Bin Laden).
    v = _f(r["quote"])
    if v is not None and 1 <= v <= 9:
        out.append(_session("s08", "liked" if r["quoteGroup"] == "1" else "disliked", pid, {"value": 10 - v}))
    # s09 flag priming. [codebook: flagdv1..flagdv8] Republican-Democratic scale items, higher = more conservative.
    # flagGroup 1 = flag prime, 0 = control (2 = unassigned, all outside the filter). Keep flagfilter == 1 (US + photo check).
    # The data file already reverse-scores the <<Reverse scored>> items (2-6, 8): unrecoded item means give the paper's
    # cell means (3.14 control, 3.18 flag vs 3.15 and 3.17). The evaluation reverses those items itself, as it must
    # for AI participants who answer on the printed scale, so the session holds the printed 1-7 answer (8 - value).
    # Positions are kept (None for a missing item) so each value stays with its item.
    items = [None if x is None else (8 - x if k in (2, 3, 4, 5, 6, 8) else x)
             for k, x in ((k, _f(r[f"flagdv{k}"])) for k in range(1, 9))]
    if r["flagfilter"] == "1" and r["flagGroup"] in ("0", "1") and any(x is not None for x in items):
        out.append(_session("s09", "flag" if r["flagGroup"] == "1" else "control", pid, {"items": items}))
    # s10 currency priming. [codebook: sysjust1..sysjust8] system justification, already recoded, higher = more.
    # MoneyGroup 1 = money prime (checked: mean 3.58 as in the paper), 0 = control. Need >= 6 of 8 items.
    # Items 3 and 7 are <<Reverse scored>> and already recoded in the data; the evaluation reverses them itself, so
    # the session holds the printed 1-7 answer (8 - value), with positions kept as for s09.
    items = [None if x is None else (8 - x if k in (3, 7) else x)
             for k, x in ((k, _f(r[f"sysjust{k}"])) for k in range(1, 9))]
    if sum(x is not None for x in items) >= 6:
        out.append(_session("s10", "money" if r["MoneyGroup"] == "1" else "control", pid, {"items": items}))
    # s11 imagined contact. [codebook: imaginedexplicit1..imaginedexplicit4] contact intentions, 1-9.
    # ContactGroup 1 = imagined contact (mean 4.83 as in the paper), 0 = control.
    items = [x for x in (_f(r[f"imaginedexplicit{k}"]) for k in range(1, 5)) if x is not None]
    if items:
        out.append(_session("s11", "contact" if r["ContactGroup"] == "1" else "control", pid, {"items": items}))
    return out


def load(data_dir):
    path = next(Path(data_dir).rglob("Tab.delimited.Cleaned.dataset.WITH.variable.values.dat"))
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    sessions = []
    for r in rows:
        sessions.extend(_build(r))
    return sessions


# (id, label, study key, reported weighted d from Table 2 of the paper)
# allowed_forbidden: Table 2's d=1.96 (chi2=3088.7) compares the raw Yes rate between wordings (7% yes to forbid vs 76% yes
# to allow), which is not the evaluation's restrictive-answer contrast (No to allow vs Yes to forbid), so reported is None.
# s11: the evaluation returns not_ready (materials.json has scoring_ready inside response_format, not on the study), so estimate is None.
_KEYS = [
    ("sunk_costs", "Sunk costs: paid vs free ticket", "s01", 0.27),
    ("framing", "Gain vs loss framing: sure option chosen more in the gain frame", "s02", 0.60),
    ("anchoring_nyc", "Anchoring: distance to NYC (high - low)", ("s03", "nyc"), 1.17),
    ("anchoring_chicago", "Anchoring: Chicago population (high - low)", ("s03", "chicago"), 1.79),
    ("anchoring_everest", "Anchoring: Mt. Everest (high - low)", ("s03", "everest"), 2.23),
    ("anchoring_babies", "Anchoring: babies born per day (high - low)", ("s03", "babies"), 2.42),
    ("gamblers_fallacy", "Retrospective gambler's fallacy: three 6s vs two 6s and a 3", "s04", 0.61),
    ("category_scales", "Category scales: over 2.5 h more often on the low-frequency scale", "s05", 0.49),
    ("reciprocity", "Norm of reciprocity: more support when the American question comes first", "s06", 0.30),
    ("allowed_forbidden", "Allowed/forbidden: more restrictive answers with 'allow' wording", "s07", None),
    ("quote_attribution", "Quote attribution: more agreement when attributed to the liked speaker", "s08", 0.32),
    ("flag_priming", "Flag priming: flag minus control", "s09", 0.03),
    ("currency_priming", "Currency priming: money minus control", "s10", -0.02),
    ("imagined_contact", "Imagined contact: contact minus control", "s11", 0.13),
]


def _effect(res):
    """Signed standardised effect (the evaluator's contrast is already in the predicted direction)."""
    if not res or res.get("status") != "ok":
        return None, None
    if res.get("test") == "t":
        return res["d"], res["p"]
    if res.get("test") == "chi2":
        r = res["phi"]   # the paper's chi-square effect sizes are d = 2r / sqrt(1 - r^2) with r = phi (matches Table 2)
        d = 2 * r / math.sqrt(1 - r * r)
        p = res["proportions"]
        sign = 1.0 if p[0] >= p[1] else -1.0   # first condition minus second, as in the evaluator's CONTRAST
        return sign * d, res["p"]
    return None, None


def key_effects(evaluation):
    """Estimate is None (and p None) when the evaluation could not run that test (status not_ready)."""
    studies = (evaluation or {}).get("studies", {})
    out = []
    for kid, label, key, reported in _KEYS:
        res = studies.get(key[0] if isinstance(key, tuple) else key)
        if isinstance(key, tuple) and res:
            res = (res.get("topics") or {}).get(key[1])
        est, p = _effect(res)
        out.append({"id": kid, "label": label, "estimate": est, "p": p, "reported": reported})
    return out
