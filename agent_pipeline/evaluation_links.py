"""Does each evaluation link hold the statistic it claims?

Called by ``package_checks.check_analyses``. Contract: ``agent_pipeline/CLAUDE.md``,
"Each recorded result is computed or explicitly not ready". A recorded human result
names, in ``extensions.evaluation``, the ``evaluate()`` output that computes it. Checking
only that the named key exists let a correlation and an alpha point at a paired t, a
share point at a mean in cents, and a six-column regression table point at one model.

1. **Statistic kind.** The kinds a result records (r, alpha, t, F, chi2, z, OR, h, d,
   b, mean, SD, share/percent, ...) come from its field id and value keys. The output
   must hold each kind: a key of that kind in the stand-in run's output, or, when the
   output is not ready on the stand-in run (so its keys cannot be seen), a declaration
   in ``evaluation/evaluation.py``::

       OUTPUT_KINDS = {
           "study2.bsr_vs_chopra_participant": "mean sd t df p",
           "descriptives.dg_mean_share": {"kinds": "mean", "unit": "share"},
           "stage1_lie.*": "n share",           # one segment wildcard
           "exp2.table3": {"kinds": "chi2 p or h n", "rows": True},
       }

   A declaration the computed output contradicts (declares r, returns no r) fails too.
2. **Units.** A share or percentage linked to an output in cents, points or seconds
   (or a share to a percentage) fails.
3. **Cells.** A table, or a field recording several values of one kind ("AWL" and
   "PWL" shares), cannot be linked as a whole to one output that holds a single value:
   link each cell, ``"cells": {"rows.PWL.IV": "dg_truth_regression.PWL", ...}``, with a
   whole-field ``not_ready`` for the cells not computed.
4. **n.** When a table's n for one group is the same in most rows but clearly smaller
   in one (the paper drops part of a group from that comparison), the evaluation must
   drop it too: the link says so in ``"exclusion"`` (what the paper drops and where
   ``evaluate()`` drops it).
"""
from __future__ import annotations

import fnmatch
import json
import re
from collections import Counter
from typing import Any, Iterable

KINDS: dict[str, tuple[str, ...]] = {
    "mean": ("mean", "means", "m", "avg", "average", "mu", "mdn", "median"),
    "sd": ("sd", "sds", "std", "stdev", "stddev", "standard_deviation"),
    "t": ("t", "tstat", "t_stat", "t_value", "tvalue", "ttest"),
    "f": ("f", "fstat", "f_value", "anova"),
    "chi2": ("chi2", "chisq", "chi_square", "chisquare", "x2", "χ2", "χ²", "chi"),
    "z": ("z", "zstat", "z_value"),
    "r": ("r", "corr", "correlation", "correlations", "pearson", "pearson_r"),
    "rho": ("rho", "spearman"),
    "alpha": ("alpha", "alphas", "cronbach", "cronbachs", "reliability"),
    "d": ("d", "cohens_d", "cohen_d", "dz"),
    "g": ("g", "hedges_g"),
    "h": ("h", "cohens_h", "cohen_h"),
    "or": ("or", "odds_ratio", "oddsratio", "odds"),
    "b": ("b", "coef", "coeff", "coefficient", "coefficients", "beta", "estimate", "slope"),
    "eta2": ("eta2", "eta_squared", "etasq", "partial_eta2", "etap2", "pes"),
    "u": ("u", "mann_whitney", "mannwhitney", "wilcoxon", "ranksum"),
    "share": ("share", "shares", "proportion", "proportions", "prop", "fraction", "rate", "rates", "percent",
              "percentage", "pct"),
}
# Kinds a link must hold. p, df, n and CIs go with a test and are not judged alone.
MULTIWORD = {"cohens_d", "cohen_d", "hedges_g", "cohens_h", "cohen_h", "odds_ratio", "standard_deviation",
             "chi_square", "mann_whitney", "eta_squared", "partial_eta2", "pearson_r", "t_stat", "t_value",
             "f_value", "z_value"}
TOKEN_KIND = {word: kind for kind, words in KINDS.items() for word in words}
UNITS: dict[str, tuple[str, ...]] = {
    "share": ("share", "shares", "proportion", "proportions", "fraction", "prop"),
    "percent": ("percent", "percentage", "pct"),
    "money": ("cents", "cent", "dollars", "dollar", "usd", "eur", "euro", "euros", "pounds", "gbp", "pence",
              "points", "tokens", "ecu", "coins", "yen", "francs", "chf", "money"),
    "time": ("ms", "msec", "milliseconds", "seconds", "secs", "sec", "minutes", "mins", "hours", "hrs", "days"),
}
TOKEN_UNIT = {word: unit for unit, words in UNITS.items() for word in words}
COUNT_KEYS = re.compile(r"(?i)^(chosen|count|k|x|successes|yes|hits|lies|cooperat\w*|choices?|selected|correct)$")
META_KEYS = re.compile(r"(?i)^(title|caption|columns?|labels?|notes?|source|page|stars|label|what|scenario|option|"
                       r"row|col|study|condition|group|text|printed|p_printed|unit|units)$")


def tokens(name: str) -> list[str]:
    """Words of a key or id: "dg_mean_share_of_endowment" -> dg, mean, share, of, endowment;
    multi-word kind names ("cohens_d", "odds_ratio") stay one token."""
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(name)).replace("²", "2").lower()
    parts = [p for p in re.split(r"[^a-z0-9χ]+", name) if p]
    out: list[str] = []
    skip = False
    for index, part in enumerate(parts):
        if skip:
            skip = False
            continue
        pair = f"{part}_{parts[index + 1]}" if index + 1 < len(parts) else ""
        if pair in MULTIWORD:
            out.append(pair)
            skip = True
        else:
            out.append(part)
    return out


def kinds_of(name: str) -> set[str]:
    return {TOKEN_KIND[t] for t in tokens(name) if t in TOKEN_KIND}


def units_of(name: str) -> set[str]:
    return {TOKEN_UNIT[t] for t in tokens(name) if t in TOKEN_UNIT}


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def leaves(value: Any, path: tuple[str, ...] = (), with_none: bool = False) -> Iterable[tuple[tuple[str, ...], Any]]:
    """Numeric leaves (and numeric strings) with their key path; with_none also keys whose
    value is null (an output that could not compute it on this run)."""
    if isinstance(value, dict):
        # Keys "a", "b", "c" side by side are group labels, not a coefficient b.
        letters = sum(1 for k in value if re.fullmatch(r"[A-Ea-e]", str(k)))
        for key, item in value.items():
            label = letters >= 2 and re.fullmatch(r"[A-Ea-e]", str(key))
            yield from leaves(item, path + (("label",) if label else (str(key),)), with_none)
    elif isinstance(value, list):
        if value and all(_number(v) for v in value) and len(value) <= 3:  # [estimate, se] or a CI: one cell
            yield path, value
            return
        for index, item in enumerate(value):
            yield from leaves(item, path + (str(index),), with_none)
    elif _number(value) or (isinstance(value, str) and re.fullmatch(r"\s*-?\d*\.?\d+\s*", value)):
        yield path, value
    elif value is None and with_none:
        yield path, value


def _leaf_kind(path: tuple[str, ...], default: set[str]) -> set[str]:
    for key in reversed(path):
        if META_KEYS.match(key):
            return set()
        found = kinds_of(key)
        if found:
            return found
    return set(default)


def recorded_kinds(field_id: str, value: Any) -> tuple[set[str], Counter]:
    """The statistic kinds a result field records, and how many values of each."""
    default = kinds_of(field_id)
    counts: Counter = Counter()
    kinds: set[str] = set()
    for path, _ in leaves(value):
        found = _leaf_kind(path, default)
        kinds |= found
        for kind in found:
            counts[kind] += 1
    if not counts and not default and isinstance(value, str):
        # "d = 0.77", "r = .88, p < .001": the statistic named before each number.
        for name in re.findall(r"(?<![\w.])([A-Za-zχηα²2]+)\s*(?:\([^)]{0,20}\))?\s*=\s*-?\d*\.?\d", value):
            default |= kinds_of(name)
    if not counts and default:
        kinds |= default
        for kind in default:
            counts[kind] += 1
    return kinds, counts


def recorded_units(field: dict) -> set[str]:
    units = units_of(str(field.get("id", ""))) | units_of(str(field.get("label", "")))
    value = field.get("value")
    if isinstance(value, dict):
        for path, _ in leaves(value):
            for key in path:
                units |= units_of(key)
        if isinstance(value.get("unit"), str):
            units |= units_of(value["unit"])
    return units


# --------------------------------------------------------------------------
# What an output holds


def lookup(output: Any, path: str) -> tuple[bool, Any]:
    current = output
    for part in [p for p in re.split(r"\.|/", path) if p]:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, current


def not_ready(value: Any) -> bool:
    return isinstance(value, dict) and (bool(value.get("not_ready")) or value.get("status") == "not_ready")


def _hidden(value: Any) -> bool:
    """True when part of the output is not ready, so its keys cannot be seen."""
    if value is None or not_ready(value):
        return True
    if isinstance(value, dict):
        return any(_hidden(v) for v in value.values())
    if isinstance(value, list):
        return any(_hidden(v) for v in value)
    return False


def declared(declarations: Any, path: str) -> dict | None:
    """The OUTPUT_KINDS entry for an output path ({"kinds", "unit", "rows"}), or None."""
    if not isinstance(declarations, dict):
        return None
    for pattern, entry in declarations.items():
        if not isinstance(pattern, str):
            continue
        if pattern == path or ("*" in pattern and len(pattern.split(".")) == len(path.split("."))
                               and fnmatch.fnmatchcase(path, pattern)):
            if isinstance(entry, str):
                entry = {"kinds": entry}
            elif isinstance(entry, list):
                entry = {"kinds": " ".join(map(str, entry))}
            if not isinstance(entry, dict):
                return None
            words = entry.get("kinds", entry.get("stats", ""))
            words = " ".join(map(str, words)) if isinstance(words, list) else str(words)
            kinds = set()
            for word in re.split(r"[\s,;]+", words):
                kinds |= kinds_of(word)
            unit = entry.get("unit")
            return {"kinds": kinds, "unit": units_of(unit) if isinstance(unit, str) else set(),
                    "rows": bool(entry.get("rows") or entry.get("table"))}
    return None


def output_kinds(path: str, value: Any) -> tuple[set[str], Counter, set[str]]:
    """Kinds an output holds, counts, and its units (from the path of a scalar output and
    from its keys)."""
    kinds: set[str] = set()
    counts: Counter = Counter()
    units: set[str] = set(units_of(path.split(".")[-1]))
    if not isinstance(value, (dict, list)) or (isinstance(value, list) and value
                                               and all(_number(v) or v is None for v in value)):
        # A scalar (or a CI pair, or a value not computed on this run): its key names it.
        found = kinds_of(path.split(".")[-1])
        kinds |= found
        counts.update({k: 1 for k in found})
        return kinds, counts, units
    for leaf_path, _ in leaves(value, with_none=True):
        found = _leaf_kind(leaf_path, set())
        kinds |= found
        counts.update({k: 1 for k in found})
        for key in leaf_path:
            units |= units_of(key)
    if "share" not in kinds and _holds_counts(value):
        kinds.add("share")  # "chosen" out of "n": the rate is there
    return kinds, counts, units


def _holds_counts(value: Any) -> bool:
    if isinstance(value, dict):
        keys = {str(k) for k, v in value.items() if _number(v)}
        if any(k.lower() in ("n", "total") for k in keys) and any(COUNT_KEYS.match(k) for k in keys):
            return True
        return any(_holds_counts(v) for v in value.values())
    if isinstance(value, list):
        return any(_holds_counts(v) for v in value)
    return False


def _rows(value: Any) -> Any:
    if isinstance(value, dict) and isinstance(value.get("rows"), (list, dict)):
        return value["rows"]
    return None


def _row_count(value: Any) -> int:
    rows = _rows(value)
    return len(rows) if rows is not None else 0


# --------------------------------------------------------------------------
# The check


def _cells(link: dict) -> dict:
    cells = link.get("cells")
    return cells if isinstance(cells, dict) else {}


def _cell_link(entry: Any) -> dict | None:
    if isinstance(entry, str):
        return {"output": entry}
    return entry if isinstance(entry, dict) else None


def links_of(link: dict) -> list[tuple[str, dict]]:
    """Every (cell path, link) of a field's extensions.evaluation; "" is the whole field."""
    out = []
    if isinstance(link.get("output"), str) or isinstance(link.get("not_ready"), str):
        out.append(("", link))
    for cell, entry in _cells(link).items():
        cell_link = _cell_link(entry)
        if cell_link is not None:
            out.append((str(cell), cell_link))
    return out


def has_link(link: Any) -> bool:
    if not isinstance(link, dict):
        return False
    return any((isinstance(l.get("output"), str) and l["output"].strip())
               or (isinstance(l.get("not_ready"), str) and l["not_ready"].strip()) for _, l in links_of(link))


def outputs_named(link: Any) -> list[str]:
    return [l["output"] for _, l in links_of(link) if isinstance(l.get("output"), str)] \
        if isinstance(link, dict) else []


def _sub(value: Any, cell: str) -> tuple[bool, Any]:
    return lookup(value, cell) if cell else (True, value)


def _is_table(value: Any) -> bool:
    return _row_count(value) >= 2


def check_field(name: str, field: dict, output: Any, declarations: Any) -> list[str]:
    """Problems of one result field's evaluation links."""
    link = (field.get("extensions") or {}).get("evaluation")
    if not isinstance(link, dict):
        return []
    problems: list[str] = []
    value = field.get("value")
    field_id = str(field.get("id", ""))
    cells = _cells(link)
    for cell in cells:
        if not _sub(value, str(cell))[0]:
            problems.append(f"{name}: evaluation cell '{cell}' is not a path into the recorded value "
                            "(use the keys of the value, e.g. rows.AWL.I or rows.3)")
    for cell, entry in links_of(link):
        path = entry.get("output")
        if not isinstance(path, str) or not path.strip():
            continue
        found, sub = _sub(value, cell)
        if not found:
            continue
        where = f"{name}" + (f" cell {cell}" if cell else "")
        exists, out = lookup(output, path)
        if not exists:
            continue  # reported by the existing "do not exist" check
        declaration = declared(declarations, path)
        # A whole table linked to one output that is not a table of the same rows.
        if not cell and _is_table(value) and not (declaration or {}).get("rows"):
            rows = _row_count(value)
            out_rows = _row_count(out) or (len(out) if isinstance(out, list) else 0)
            # An output keyed by row ({"AWL": {...}, "PWL": {...}}) is a table too.
            keyed = len([v for v in out.values() if isinstance(v, dict)]) if isinstance(out, dict) else 0
            if max(out_rows, keyed) < rows:
                problems.append(
                    f"{where} links a table of {rows} rows to one evaluate() output ({path}) that is not a table "
                    "of those rows. Link each computed cell or row: extensions.evaluation = {\"not_ready\": "
                    "\"<why the other cells are not computed>\", \"cells\": {\"rows.<row>.<column>\": "
                    "\"<output key path>\"}}")
                continue
        # Rows are judged one by one when the output is a table of them.
        rec_kinds, rec_counts = recorded_kinds(field_id if not cell else cell.split(".")[-1] + " " + field_id, sub)
        out_kinds, out_counts, out_units = output_kinds(path, out)
        if declaration:
            out_kinds |= declaration["kinds"]
            out_units |= declaration["unit"]
            if not _hidden(out):
                contradicted = sorted(k for k in declaration["kinds"] - output_kinds(path, out)[0])
                if contradicted:
                    problems.append(f"evaluation/evaluation.py OUTPUT_KINDS declares {', '.join(contradicted)} for "
                                    f"{path}, but the computed output holds no such value")
        units = recorded_units(field) if not cell else units_of(cell) | recorded_units(field)
        clash = _unit_clash(units, out_units)
        if clash:
            problems.append(f"{where} records a {clash[0]} but links to {path}, which is in {clash[1]}: "
                            "link an output on the same scale (or compute it, e.g. the mean share of the endowment)")
            continue
        missing = sorted(rec_kinds - out_kinds)
        if missing:
            if _hidden(out) and declaration is None:
                problems.append(
                    f"{where} records {', '.join(missing)} and links to {path}, which is not computed on the stand-in "
                    "run, so the check cannot see what it holds. Declare it in evaluation/evaluation.py: OUTPUT_KINDS "
                    f"= {{\"{path}\": \"<kinds it returns, e.g. t df p mean sd>\"}}; if it does not compute "
                    f"{', '.join(missing)}, link the output that does or give that value its own not_ready.")
            else:
                problems.append(
                    f"{where} records {', '.join(missing)} but {path} holds no {', '.join(missing)} (it holds "
                    f"{', '.join(sorted(out_kinds)) or 'none of the recorded kinds'}). Link the output that computes "
                    "each value, or mark that value not_ready (extensions.evaluation.cells).")
            continue
        if not _hidden(out) and declaration is None and not _is_table(sub):
            over = sorted(k for k in rec_kinds if rec_counts[k] >= 2 and out_counts.get(k, 0) == 1)
            if over:
                problems.append(
                    f"{where} records {rec_counts[over[0]]} values of {over[0]} but links them all to {path}, which "
                    f"holds one. Link each value to its own output (extensions.evaluation.cells = {{\"<key>\": "
                    "\"<output key path>\"}}) or mark the others not_ready.")
    problems += _n_problems(name, field, link)
    return problems


def _unit_clash(recorded: set[str], output: set[str]) -> tuple[str, str] | None:
    if not recorded or not output or recorded & output:
        return None
    names = {"share": "share (0-1)", "percent": "percentage", "money": "money amount (cents, points)",
             "time": "time"}
    return names[sorted(recorded)[0]], names[sorted(output)[0]]


def _n_paths(row: Any, path: tuple[str, ...] = ()) -> dict[tuple[str, ...], float]:
    out: dict[tuple[str, ...], float] = {}
    if isinstance(row, dict):
        for key, item in row.items():
            if str(key) in ("n", "N") and _number(item):
                out[path + (str(key),)] = float(item)
            elif isinstance(item, (dict, list)):
                out.update(_n_paths(item, path + (str(key),)))
    return out


def _n_problems(name: str, field: dict, link: dict) -> list[str]:
    """A group's n that is the same in most rows and clearly smaller in one: the paper
    drops part of the group from that comparison."""
    if not any(isinstance(l.get("output"), str) for _, l in links_of(link)):
        return []
    rows = _rows(field.get("value"))
    if rows is None:
        return []
    items = list(rows.items()) if isinstance(rows, dict) else list(enumerate(rows))
    if len(items) < 4:
        return []
    noted = isinstance(link.get("exclusion"), str) and len(re.findall(r"\w+", link["exclusion"])) >= 4
    cell_notes = {str(c) for c, l in links_of(link) if c and isinstance(l.get("exclusion"), str)
                  and len(re.findall(r"\w+", l["exclusion"])) >= 4}
    by_path: dict[tuple[str, ...], list[tuple[Any, float]]] = {}
    for key, row in items:
        ns = _n_paths(row)
        for path, n in ns.items():
            by_path.setdefault(path, []).append((key, n))
        if len(ns) >= 2:  # the row's groups together: the same participants re-split in every row
            by_path.setdefault(("all groups",), []).append((key, sum(ns.values())))
    found = []
    for path, values in by_path.items():
        mode, times = Counter(n for _, n in values).most_common(1)[0]
        if times < 3 or times < len(values) / 2:
            continue
        for key, n in values:
            if n < mode * 0.9 and mode - n >= 5 and not noted and not any(c.startswith(f"rows.{key}") for c in
                                                                           cell_notes):
                label = rows[key].get("option") or rows[key].get("label") or key if isinstance(rows, list) \
                    and isinstance(rows[key], dict) else key
                found.append(f"row {label} {'.'.join(path)} n = {n:g} (other rows {mode:g})")
    if not found:
        return []
    return [f"{name}: the paper's n is smaller in one row than in the others ({'; '.join(found[:3])}), so the "
            "paper drops part of that group from this comparison (an exclusion it may state only in the text). "
            "Find why, apply the same exclusion in evaluate() for that row, and say so in extensions.evaluation "
            "\"exclusion\": \"<what the paper drops, where it says so, where evaluate() drops it>\"."]


def check(fields: list[tuple[str, dict]], output: Any, declarations: Any) -> list[str]:
    problems: list[str] = []
    for name, field in fields:
        try:
            problems += check_field(name, field, output, declarations)
        except Exception as exc:  # noqa: BLE001 - a malformed record is reported, not a crash
            problems.append(f"{name}: evaluation link could not be checked ({type(exc).__name__}: {exc})")
    return list(dict.fromkeys(problems))


def summary(problems: list[str]) -> list[str]:
    if not problems:
        return []
    return ["analyses: evaluation links do not hold what the recorded results claim: " + " | ".join(problems[:12])
            + (f" | and {len(problems) - 12} more" if len(problems) > 12 else "")
            + " (contract 'Each recorded result is computed or explicitly not ready'; declare what each output "
              "holds in evaluation/evaluation.py OUTPUT_KINDS)."]


if __name__ == "__main__":
    print(json.dumps({"kinds": sorted(KINDS), "units": sorted(UNITS)}))
