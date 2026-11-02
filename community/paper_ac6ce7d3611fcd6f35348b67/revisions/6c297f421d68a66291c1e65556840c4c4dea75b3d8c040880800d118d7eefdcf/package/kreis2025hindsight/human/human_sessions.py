"""Human data for Kreis et al. (2025), study s1 (hindsight bias through knowledge updating, sugar content).

Standard library only. load(data_dir) -> sessions in the shape of task/adapter.run_sessions; key_effects(evaluation).
"""
import csv
import json
import re
from pathlib import Path

SOURCE = ("human-data/data/HBKU_Rep_Sugar_after_exclusion_all.csv (OSF materials of Kreis et al., 2025; "
          "'Dataset 2' in description.txt: after participant exclusions, still containing perfectly accurate "
          "judgments (CJ) and perfect reproductions (CR)). Variable meanings: human-data/data/description.txt.")

EXCLUSIONS = ("Participant exclusions stated in the paper were already applied by the authors in the file "
              "(unfinished within 87 min, smartphone users, cheating / just clicking through, technical problems, "
              ">=10 estimates above the item weight). Not applied here: perfect-OJ and perfect-reproduction item "
              "cases (CJ/CR) and the outlier rule, which evaluation.evaluate applies itself to every session set; "
              "the CJ/CR rows are kept in the sessions (Dataset 2).")

NOT_MAPPED = []  # all four arms (control, feedback, domain, irrelevant) are in the data

# Codebook entries (description.txt) -> use:
#   id            "unique subject ID"                       -> session["participant"]
#   cond          "experimental condition/group"            -> arm (Control/Feedback/Domain/Irrelevant)
#   phase         "OJScreen / ROJScreen / NJScreen"         -> parsed["oj"] / ["roj"] / ["nj"]
#   idn           "id of the food item, numbers 1-96"       -> item (via _IDN_TO_ITEM, matched on set, sugar, category, name)
#   setNumber     "which set an item belongs to"            -> 1,2,3 = package sets A,B,C (sugar values agree)
#   estimate      "estimate of sugar content (renamed foodEstimate)" -> estimate / parsed value
#   sugar         "true sugar content of the food"          -> checked against the package item's sugar
#   food_category "category an item belongs to"             -> used to disambiguate items
#   position      "ascending number of the item per person and phase" -> order of the parsed list
# Not used: counterBalance, frequency_contact_num, deviation_abs, deviation_simple (derived or not needed).

_ARM = {"Control": "control", "Feedback": "feedback", "Domain": "domain", "Irrelevant": "irrelevant"}
_COND = {"control": "c_control", "feedback": "c_feedback", "domain": "c_domain", "irrelevant": "c_irrelevant"}
_PHASE = {"OJScreen": "oj", "ROJScreen": "roj", "NJScreen": "nj"}
_CAT = {"grain": "Grains and grain products", "vegetables": "Vegetables", "fruits": "Fruit", "milk": "Milk and dairy products",
        "fishmeat": "Fish, meat, and sausage", "oilsfats": "Oils and fats", "drinks": "Drinks", "extras": "Sweets and snacks"}
# idn -> package item id where set, sugar and category alone do not decide (German name -> English item)
_MANUAL = {1: "i52", 2: "i49", 3: "i51", 6: "i38", 7: "i40", 8: "i04", 9: "i02", 10: "i61", 11: "i64",
           33: "i56", 34: "i53", 36: "i66", 37: "i67", 45: "i08", 47: "i05", 48: "i07",
           66: "i48", 67: "i46", 69: "i70", 70: "i71", 72: "i59", 73: "i57", 74: "i60", 75: "i12", 76: "i09",
           84: "i22", 85: "i23"}
_MAT = Path(__file__).resolve().parent.parent / "materials" / "materials.json"


def _items():
    return {i["id"]: i for i in json.loads(_MAT.read_text(encoding="utf-8"))["studies"]["s1"]["items"]}


def _idn_map(rows, items):
    seen = {}
    for r in rows:
        seen.setdefault(int(r["idn"]), (int(r["sugar"]), r["food_category"], "ABC"[int(r["setNumber"]) - 1]))
    out = {}
    for idn, (sugar, cat, st) in seen.items():
        if idn in _MANUAL:
            it = items[_MANUAL[idn]]
        else:
            c = [i for i in items.values() if i["set"] == st and i["sugar"] == sugar and i["subcategory"] == _CAT[cat]]
            if len(c) != 1:
                raise ValueError("item idn %d is ambiguous: %s" % (idn, [i["id"] for i in c]))
            it = c[0]
        assert it["sugar"] == sugar and it["set"] == st, (idn, it["id"])
        out[idn] = it
    return out


def load(data_dir):
    path = Path(data_dir) / "data" / "HBKU_Rep_Sugar_after_exclusion_all.csv"
    if not path.exists():
        path = Path(data_dir) / "HBKU_Rep_Sugar_after_exclusion_all.csv"
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    items = _items()
    imap = _idn_map(rows, items)
    by = {}
    for r in rows:
        by.setdefault(r["id"], {"cond": r["cond"], "rows": {"oj": [], "roj": [], "nj": []}})["rows"][_PHASE[r["phase"]]].append(r)
    sessions = []
    for pid, p in by.items():
        arm = _ARM[p["cond"]]
        rec, parsed = {}, {}
        for ph, rs in p["rows"].items():
            rs = sorted(rs, key=lambda r: int(r["position"]))
            rec[ph] = []
            for r in rs:
                it = imap[int(r["idn"])]
                est = r["estimate"].strip()
                rec[ph].append(dict(item=it["id"], name=it["name"], weight=it["size"], actual=it["sugar"],
                                    subcategory=it["subcategory"], estimate=float(est.replace(",", ".")) if est not in ("", "NA") else None))
            vals = [t["estimate"] for t in rec[ph]]
            parsed[ph] = (vals + [None] * 32)[:32]
        sessions.append({"study": "s1", "condition": _COND[arm], "arm": arm, "participant": "human-%s" % pid,
                         "parsed": parsed, "valid": all(v is not None for v in parsed["oj"]),
                         "source": "human", "oj": rec["oj"], "roj": rec["roj"], "nj": rec["nj"]})
    order = {a: n for n, a in enumerate(("control", "feedback", "domain", "irrelevant"))}
    sessions.sort(key=lambda s: (order[s["arm"]], int(s["participant"].split("-")[1])))
    return sessions


# Reported b on |Delta| (Results, pp. 17-18); negative = more accurate than OJ. Predicted direction: negative for
# feedback and domain (hindsight: ROJ closer to the actual value; transfer: NJ closer). Estimates below are
# sign-flipped so that positive = improvement in accuracy = as predicted; control/irrelevant are predicted ~0.
_REP_HB = {"feedback": -2.88, "domain": -2.25, "irrelevant": 0.87, "control": 1.10}
_REP_TL = {"feedback": -6.59, "domain": -3.71, "irrelevant": 0.52, "control": -0.92}
_LABEL = {"feedback": "feedback", "domain": "domain-information", "irrelevant": "irrelevant-information", "control": "control"}


def key_effects(evaluation):
    out = []
    g = evaluation.get("groups", {})
    for kind, key, rep, name in (("hindsight", "hindsight_ROJ_minus_OJ", _REP_HB, "hindsight effect (OJ-ROJ reduction in |dev|)"),
                                 ("transfer", "transfer_NJ_minus_OJ", _REP_TL, "transfer learning (OJ-NJ reduction in |dev|)")):
        for arm in ("feedback", "domain", "irrelevant", "control"):
            d = g.get(arm, {}).get(key, {})
            est = -d["mean_diff"] if "mean_diff" in d else float("nan")
            out.append({"id": "%s_%s" % (kind, arm), "label": "%s, %s group" % (name, _LABEL[arm]),
                        "estimate": est, "p": None, "reported": -rep[arm]})
    return out
