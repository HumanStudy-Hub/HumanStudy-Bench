"""Human data for Cheung, Maier & Lieder (2025), Study 2 (between-subjects, six dilemmas x three framings)."""
import csv
import json
import math
import sys
from pathlib import Path

SOURCE = ("human-data/data_p.csv (Study 2 Qualtrics export merged with Prolific export, 501 rows; from the authors' "
          "OSF materials, https://osf.io/ybdr9). human-data/prompts_sacrificial.csv holds LLM prompts only (unused).")
EXCLUSIONS = ("Preregistered attention check: dropped participants whose <Vignette>_Attn answer is not 1 (the correct "
              "scenario description; 27 of 501 dropped, leaving the paper's N = 474).")
NOT_MAPPED = [{"arm": "s1_moral_sequential",
               "reason": "Study 1 human data (N = 285, 13 sequential dilemmas) are not in human-data; data_p.csv is Study 2 only."}]

HERE = Path(__file__).resolve().parent
_ADAPTER = HERE.parent / "task" / "adapter.py"

# Codebook (no separate codebook file; coding inferred from the Qualtrics export, see notes):
#   <Vignette>_Q / _YN_Q / _O_Q  = answer to the original / Yes-No / Action-Omission version (exactly one filled per row).
#     Endowment omission version is column Q674 (the only unlabelled answer column).
#     Codes: 1/2 (Medicine) or 5/6 (all others); the lower code = "yes", the higher = "no". Confirmed against the paper:
#     Endowment original "yes" (= approve, CBR+action) is chosen by ~9-14%, matching the paper's 9%.
#   <Vignette>_Attn = attention check; 1 = correct.
_COLS = {  # vignette -> (original, yes_no, omission)
    "suicide": ("Suicide_Q", "Suicide_YN_Q", "Suicide_O_Q"),
    "medicine": ("Medicine_Q", "Medicine_YN_Q", "Medicine_O_Q"),
    "raf": ("RAF_Q", "RAF_YN_Q", "RAF_O_Q"),
    "ransom": ("Ransom_Q", "Ransom_YN_Q", "Ransom_O_Q"),
    "vet": ("Vet_Q", "Vet_YN_Q", "Vet_O_Q"),
    "endowment": ("Endow_Q", "Endow_YN_Q", "Q674"),
}
_ATTN = {"suicide": "Suicide_Attn", "medicine": "Medicine_Attn", "raf": "RAF_Attn", "ransom": "Ransom_Attn",
         "vet": "Vet_Attn", "endowment": "Endow_Attn"}
_ARMS = ("s2_original", "s2_yes_no", "s2_omission")
_FRAMING = ("original", "yes_no", "omission")
_CONDITION = {"original": "c2_orig", "yes_no": "c2_yn", "omission": "c2_om"}


def _mat():
    return json.loads((HERE.parent / "materials" / "materials.json").read_text(encoding="utf-8"))["studies"]["s2"]["vignettes"]


def load(data_dir):
    vig = _mat()
    path = Path(data_dir) / "data_p.csv"
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    sessions = []
    for i, row in enumerate(rows):
        for v, cols in _COLS.items():
            for arm, framing, col in zip(_ARMS, _FRAMING, cols):
                code = (row.get(col) or "").strip()
                if not code:
                    continue
                if (row.get(_ATTN[v]) or "").strip() != "1":
                    continue  # failed attention check
                if code not in ("1", "2", "5", "6"):
                    continue
                answer = "yes" if code in ("1", "5") else "no"
                item = vig[v][framing]
                cbr = item["answer_key"][answer] == "CBR"
                parsed = {"answer": answer, "cbr": cbr, "yes_is_cbr": item["answer_key"]["yes"] == "CBR",
                          "cbr_is_action": item["cbr_is_action"], "vignette": v, "framing": framing, "reply": answer}
                sessions.append({"study": "s2", "condition": _CONDITION[framing], "arm": arm, "participant": i,
                                 "trials": [parsed], "parsed": parsed, "valid": True, "source": "human"})
    return sessions


def key_effects(evaluation):
    s2 = evaluation.get("study_2") or {}
    out = []
    for id_, label, k, rep in (
            ("yes_no_bias_1", "Study 2 yes-no bias, participants (CBR when yes=CBR minus when no=CBR; paper: 60% vs 56%, "
             "reported as no significant bias; sign flipped so positive = bias toward 'no')", "yes_no_bias", -0.04),
            ("omission_bias_1", "Study 2 omission bias, participants (CBR on omission minus on action)", "omission_bias", None)):
        r = s2.get(k) or {}
        d = r.get("difference")
        if d is None:
            continue
        est = -d if k == "yes_no_bias" else d
        out.append({"id": id_, "label": label, "estimate": float(est), "p": r.get("p"), "reported": rep})
    return out
