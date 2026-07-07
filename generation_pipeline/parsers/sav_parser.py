from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Variables that are survey plumbing / identifiers / quality flags, not measures.
_META_NAME_RE = re.compile(
    r"^(id|.*_id|.*id\d*|prolific.*|mturk.*|worker.*|ip(addr)?|"
    r"duration.*|finished|progress|status|recorded.*|response.*|"
    r"start.*|end.*|date|consent|exclude.*|attn.*|attention.*|"
    r"timer.*|.*_timer|q_.*|distribution.*|channel|latitude|longitude|"
    r"unnamed.*|index|row)$",
    re.IGNORECASE,
)

# Demographic variables: categorical but not experimental conditions.
_DEMO_NAME_RE = re.compile(
    r"^(gender|sex|male|female|age|race.*|ethnic.*|income|education|edu|"
    r"degree|party|politic.*|conserv.*|liberal|religio.*|ses|"
    r"hispanic|nationality|country|state|zip|marital|employ.*)$",
    re.IGNORECASE,
)


@dataclass
class SavVariable:
    name: str
    label: str
    measure: str
    role: str
    type: str
    choices: List[str] = field(default_factory=list)
    scale: Optional[Dict[str, Any]] = None

    def is_response_item(self) -> bool:
        return self.role == "item"


@dataclass
class ParsedDataset:
    dataset_name: str
    n_rows: int
    variables: List[SavVariable] = field(default_factory=list)
    source_kind: str = "sav_data"
    provides_stimulus: bool = False

    def response_items(self) -> List[SavVariable]:
        return [v for v in self.variables if v.role == "item"]

    def condition_variables(self) -> List[SavVariable]:
        return [v for v in self.variables if v.role == "condition"]

    def demographics(self) -> List[SavVariable]:
        return [v for v in self.variables if v.role == "demographic"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "source_kind": self.source_kind,
            "provides_stimulus": self.provides_stimulus,
            "n_rows": self.n_rows,
            "variables": [asdict(v) for v in self.variables],
        }

# Value-label interpretation

def _ordered_value_labels(value_labels: Optional[Dict[Any, str]]) -> List[tuple]:
    """[(numeric_key, label), ...] sorted by numeric key when possible."""
    if not value_labels:
        return []
    pairs = []
    for k, v in value_labels.items():
        try:
            key = (0, float(k))
        except (TypeError, ValueError):
            key = (1, str(k))
        pairs.append((key, k, str(v).strip()))
    pairs.sort(key=lambda t: t[0])
    return [(k, lab) for _, k, lab in pairs]


def _scale_key_text(value: Any) -> str:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    return str(int(n)) if n == int(n) else str(n)


def _clean_scale_anchor(key: Any, label: Any) -> str:
    key_text = _scale_key_text(key)
    text = re.sub(r"\s+", " ", str(label or "")).strip()
    if not text:
        return key_text

    leading = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*(.*)$", text)
    if leading:
        leading_key = _scale_key_text(leading.group(1))
        rest = leading.group(2).strip()
        if leading_key == key_text:
            rest = re.sub(
                r"^(?:[?=:.\-]|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|\u2212)+\s*",
                "",
                rest,
            ).strip()
            return rest or key_text

    return text


def _looks_like_rating_scale(value_labels: Optional[Dict[Any, str]], measure: str) -> bool:
    """A rating scale has >=3 numeric, monotonically keyed anchors."""
    ordered = _ordered_value_labels(value_labels)
    if len(ordered) < 3:
        return False
    nums = []
    for k, _ in ordered:
        try:
            nums.append(float(k))
        except (TypeError, ValueError):
            return False
    return measure in ("scale", "ordinal") or nums == sorted(nums)


def _scale_from_value_labels(value_labels: Dict[Any, str]) -> Optional[Dict[str, Any]]:
    nums: List[float] = []
    anchors: Dict[str, str] = {}
    for k, lab in value_labels.items():
        try:
            n = float(k)
        except (TypeError, ValueError):
            continue
        nums.append(n)
        if lab and str(lab).strip():
            ik = int(n) if n == int(n) else n
            anchors[str(ik)] = _clean_scale_anchor(ik, lab)
    if not nums:
        return None
    lo, hi = min(nums), max(nums)
    return {
        "min": int(lo) if lo == int(lo) else lo,
        "max": int(hi) if hi == int(hi) else hi,
        "anchors": anchors,
    }


# Role + type classification

def _classify(name: str, label: str, measure: str,
              value_labels: Optional[Dict[Any, str]]) -> tuple[str, str]:
    """Return (role, normalized_type).

    role: id_meta | demographic | condition | item | derived
    """
    has_labels = bool(value_labels)
    is_scale = _looks_like_rating_scale(value_labels, measure)

    if _META_NAME_RE.match(name):
        return "id_meta", "meta"

    if re.search(r"_x_|_X_|interaction", name) or (not label and not has_labels):
        return "derived", "numeric"

    if _DEMO_NAME_RE.match(name):
        return "demographic", "multiple_choice" if has_labels else "numeric"

    if has_labels and not is_scale and measure == "nominal":
        return "condition", "multiple_choice"

    if is_scale:
        return "item", "scale"

    if has_labels:
        return "item", "multiple_choice"

    # No value labels: a measured numeric or free response.
    if measure == "scale":
        return "item", "numeric"
    return "item", "text"


def _build_variable(name: str, label: Optional[str], measure: str,
                    value_labels: Optional[Dict[Any, str]]) -> SavVariable:
    label = (label or "").strip()
    role, ntype = _classify(name, label, measure, value_labels)

    choices: List[str] = []
    scale: Optional[Dict[str, Any]] = None
    if value_labels:
        if ntype == "scale":
            scale = _scale_from_value_labels(value_labels)
        else:
            choices = [lab for _, lab in _ordered_value_labels(value_labels) if lab]

    return SavVariable(
        name=name,
        label=label,
        measure=measure or "unknown",
        role=role,
        type=ntype,
        choices=choices,
        scale=scale,
    )

# Core parse

def parse_sav_meta(meta: Any, dataset_name: str = "") -> ParsedDataset:
    """Build a ParsedDataset from a pyreadstat metadata object."""
    names = list(meta.column_names)
    labels = dict(getattr(meta, "column_names_to_labels", {}) or {})
    measures = dict(getattr(meta, "variable_measure", {}) or {})
    value_labels = dict(getattr(meta, "variable_value_labels", {}) or {})

    variables = [
        _build_variable(n, labels.get(n), measures.get(n, "unknown"), value_labels.get(n))
        for n in names
    ]
    return ParsedDataset(
        dataset_name=dataset_name or getattr(meta, "file_label", "") or "",
        n_rows=int(getattr(meta, "number_rows", 0) or 0),
        variables=variables,
    )


def parse_sav_file(path: str | Path) -> ParsedDataset:
    """Load a .sav file's metadata (no data rows) and parse it.

    Requires `pyreadstat`. Raises a clear error if it is not installed.
    """
    try:
        import pyreadstat  # noqa: WPS433 (lazy import; optional dependency)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading .sav files requires pyreadstat. Install it with "
            "`pip install pyreadstat`."
        ) from exc

    p = Path(path)
    _, meta = pyreadstat.read_sav(str(p), metadataonly=True, output_format="dict")
    return parse_sav_meta(meta, dataset_name=p.stem)

# CLI

def _main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Parse an SPSS .sav data file's codebook.")
    ap.add_argument("sav", help="Path to the .sav file")
    ap.add_argument("--json", action="store_true", help="Emit full parsed JSON")
    ap.add_argument("--all", action="store_true", help="Include id/meta variables")
    args = ap.parse_args(argv)

    ds = parse_sav_file(args.sav)

    if args.json:
        out = ds.to_dict()
        if not args.all:
            out["variables"] = [v for v in out["variables"] if v["role"] != "id_meta"]
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    print(f"Dataset: {ds.dataset_name}  ({len(ds.variables)} vars, {ds.n_rows} rows)")
    print(f"provides_stimulus={ds.provides_stimulus} (codebook only; stimulus must come from PDF)")

    groups = [
        ("Conditions", ds.condition_variables()),
        ("Items (DVs / measures)", ds.response_items()),
        ("Demographics", ds.demographics()),
    ]
    if args.all:
        groups.append(("ID / meta", [v for v in ds.variables if v.role == "id_meta"]))

    for title, vs in groups:
        if not vs:
            continue
        print(f"\n## {title}  ({len(vs)})")
        for v in vs:
            preview = v.label.replace("\n", " ")
            if len(preview) > 78:
                preview = preview[:78] + "…"
            extra = ""
            if v.scale:
                extra = f"  scale={v.scale['min']}-{v.scale['max']}"
            elif v.choices:
                extra = f"  options={v.choices[:4]}"
            print(f"  - {v.name:18s} ({v.type}) {preview}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
