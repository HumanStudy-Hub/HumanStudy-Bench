"""Print a small, pasteable digest of an evaluation run.

The full per-row output of a suite is hundreds of kilobytes, most of it scores
for individual items. What a reader needs to decide the next step is a few dozen
numbers: the calibration distance against its own scale, the same distance on
the subsets that actually carry the effect, and the letter-bias diagnostic that
says whether the run is trustworthy at all.

This prints exactly that, small enough to paste into a message.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _fmt(value: Optional[float], width: int = 8, places: int = 4) -> str:
    if value is None:
        return "-".rjust(width)
    return "{:{}.{}f}".format(float(value), width, places)


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest_suite(results_dir: Path, reference_dir: Optional[Path]) -> List[str]:
    lines: List[str] = []
    files = sorted(results_dir.glob("*.json"))
    files = [path for path in files if path.name not in {"suite_manifest.json"}]

    reference: Dict[str, Any] = {}
    if reference_dir is not None:
        overview = reference_dir / "reference_overview.json"
        if overview.exists():
            reference = _load(overview)

    manifest = results_dir / "suite_manifest.json"
    if manifest.exists():
        payload = _load(manifest)
        lines.append(
            "model={}  adapter={}".format(
                payload.get("model_label"),
                payload.get("adapter") or "none",
            )
        )
        lines.append("base_model={}".format(payload.get("base_model")))
        lines.append("")

    header = "{:<11}{:>8}{:>8}{:>8}{:>8}{:>9}{:>8}{:>8}".format(
        "dataset", "MAE", "floor", "bayes", "chance", "letterB", "paired", "acc"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for path in files:
        if path.name == "knowledge_probe.json":
            continue
        payload = _load(path)
        summary = payload.get("summary")
        if not summary:
            continue
        label = path.stem
        calibration = summary["calibration"]
        bias = summary.get("response_code_bias", {}).get("raw", {})
        mirror = summary.get("response_code_bias", {}).get("mirror", {})
        scale = reference.get(label, {}).get("scale", {})
        models = reference.get(label, {}).get("models", {})
        lines.append(
            "{:<11}{}{}{}{}{}{:>8}{}".format(
                label,
                _fmt(calibration.get("mae")),
                _fmt((scale.get("noise_floor_mae") or {}).get("mean")),
                _fmt((models.get("bayesian_hard") or {}).get("mae")),
                _fmt((models.get("uniform_half") or {}).get("mae")),
                _fmt(bias.get("median_log_odds_x"), width=9, places=2),
                mirror.get("paired_items", "-"),
                _fmt(summary["normative"].get("accuracy")),
            )
        )
    lines.append("")
    lines.append("MAE=calibration distance to humans (primary)  floor=perfect-model")
    lines.append("bayes=fully rational  chance=constant 0.5  letterB=raw letter bias")
    lines.append("(log odds of DECISION=X before symmetrization; 0 means unbiased)")
    return lines


def digest_conditions(results_dir: Path) -> List[str]:
    """Per-condition breakdown for the dataset that carries the target effect."""

    lines: List[str] = []
    for name in ("C_test", "D_test"):
        path = results_dir / "{}.json".format(name)
        if not path.exists():
            continue
        summary = _load(path)["summary"]
        calibration = summary["calibration"]
        lines.append("")
        lines.append("{} by condition:".format(name))
        for key, group in sorted(calibration.get("by_authority_condition", {}).items()):
            lines.append(
                "  {:<36}{}  n={}".format(key, _fmt(group.get("mae")), group.get("rows"))
            )
        indifference = calibration.get("indifference_subset", {})
        if indifference.get("rows"):
            lines.append(
                "  {:<36}{}  n={}".format(
                    "indifference subset",
                    _fmt(indifference.get("mae")),
                    indifference.get("rows"),
                )
            )
        authority = summary.get("authority", {})
        if authority.get("rows"):
            lines.append(
                "  authority follow: model={} human={}".format(
                    _fmt(authority.get("hard_alignment_rate"), width=6, places=3),
                    _fmt(
                        authority.get("human_mean_alignment_probability"),
                        width=6,
                        places=3,
                    ),
                )
            )
        overshoot = summary.get("overshoot", {})
        if overshoot.get("rows"):
            lines.append(
                "  overshoot: dpo_unreachable={} model_more_extreme={}".format(
                    _fmt(overshoot.get("dpo_unreachable_rate"), width=6, places=3),
                    _fmt(overshoot.get("model_more_extreme_rate"), width=6, places=3),
                )
            )
    return lines


def digest_knowledge_probe(path: Path) -> List[str]:
    if not path.exists():
        return []
    payload = _load(path)
    summary = payload["summary"]
    lines = ["", "knowledge probe:"]
    lines.append(
        "  overall forced-choice accuracy = {}".format(
            _fmt(summary.get("overall_forced_choice_accuracy"), width=5, places=3)
        )
    )
    for effect, values in sorted(summary.get("by_effect", {}).items()):
        lines.append(
            "  {}: fc_acc={} P(true)={} open_recall={}".format(
                effect,
                _fmt(values.get("forced_choice_accuracy"), width=5, places=2),
                _fmt(values.get("mean_true_statement_probability"), width=6, places=3),
                _fmt(values.get("mean_open_recall"), width=5, places=2),
            )
        )
    for probe in payload.get("probes", []):
        if probe.get("kind") == "forced_choice":
            lines.append(
                "    {:<24}P(true)={}".format(
                    probe["probe_id"],
                    _fmt(probe.get("true_statement_probability"), width=6, places=3),
                )
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    lines = digest_suite(args.results_dir, args.reference_dir)
    lines.extend(digest_conditions(args.results_dir))
    lines.extend(digest_knowledge_probe(args.results_dir / "knowledge_probe.json"))
    rendered = "\n".join(lines)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
