"""Move evaluation results into the repo so they survive the runtime that made them.

The Colab runtime has been lost twice. Datasets are deterministic and rebuild in
a minute, adapters are reproducible from them, but evaluation results are the
actual measurements and there is nothing to regenerate them from once the VM is
gone. They are also small: a suite is a few hundred kilobytes, so version
control is the right place for them.

`--from` copies one run's JSON files into `records/<label>/` and refreshes the
index. `--summary` prints one table across every run collected so far, which is
the cross-run view the report needs and no single digest can give.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .evaluate_choices import summarize_scored_rows


RECORDS_DIR = Path(__file__).resolve().parent / "records"


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_of(payload: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Recompute from stored per-item scores so metric fixes apply retroactively."""

    rows = payload.get("rows")
    if rows:
        return summarize_scored_rows(rows)
    return payload.get("summary")


def describe(directory: Path, label: str) -> Dict[str, Any]:
    """Summarize a record directory in place, without copying anything."""

    files = [path.name for path in sorted(directory.glob("*.json"))
             if path.name != "index.json"]
    if not files:
        raise ValueError("no JSON results found in {}".format(directory))

    destination = directory
    entry: Dict[str, Any] = {"label": label, "files": files, "datasets": {}}
    for name in files:
        if name in {"suite_manifest.json", "knowledge_probe.json"}:
            continue
        payload = _load(destination / name)
        summary = _summary_of(payload)
        if not summary:
            continue
        calibration = summary["calibration"]
        entry["datasets"][Path(name).stem] = {
            "mae": calibration.get("mae"),
            "cross_entropy": calibration.get("cross_entropy"),
            "rows": calibration.get("rows"),
            "accuracy": summary["normative"].get("accuracy"),
            "letter_bias_logodds": summary.get("response_code_bias", {})
            .get("raw", {})
            .get("median_log_odds_x"),
            "paired_items": summary.get("response_code_bias", {})
            .get("mirror", {})
            .get("paired_items"),
        }
    manifest = destination / "suite_manifest.json"
    if manifest.exists():
        payload = _load(manifest)
        entry["base_model"] = payload.get("base_model")
        entry["adapter"] = payload.get("adapter")
    probe = destination / "knowledge_probe.json"
    if probe.exists():
        entry["knowledge_probe"] = _load(probe)["summary"].get(
            "overall_forced_choice_accuracy"
        )
    return entry


def collect(source: Path, label: str, records_dir: Path) -> Dict[str, Any]:
    """Copy one run's results into the records tree, then summarize it."""

    destination = records_dir / label
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(source.glob("*.json")):
        target = destination / path.name
        if path.resolve() != target.resolve():
            shutil.copy2(path, target)
        copied += 1
    if not copied:
        raise ValueError("no JSON results found in {}".format(source))
    return describe(destination, label)


def refresh_index(records_dir: Path) -> Dict[str, Any]:
    index: Dict[str, Any] = {"runs": {}}
    for directory in sorted(records_dir.iterdir()):
        if not directory.is_dir():
            continue
        try:
            index["runs"][directory.name] = describe(directory, directory.name)
        except ValueError:
            continue
    (records_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return index


def _fmt(value: Optional[float], width: int = 8, places: int = 4) -> str:
    if value is None:
        return "-".rjust(width)
    return "{:{}.{}f}".format(float(value), width, places)


def summary_table(records_dir: Path) -> List[str]:
    index_path = records_dir / "index.json"
    index = _load(index_path) if index_path.exists() else refresh_index(records_dir)
    datasets: List[str] = []
    for run in index["runs"].values():
        for name in run.get("datasets", {}):
            if name not in datasets:
                datasets.append(name)
    datasets.sort()

    lines = ["{:<18}{}".format("run", "".join("{:>12}".format(d[:11]) for d in datasets))]
    lines.append("-" * len(lines[0]))
    for label, run in sorted(index["runs"].items()):
        cells = "".join(
            _fmt(run.get("datasets", {}).get(name, {}).get("mae"), width=12)
            for name in datasets
        )
        lines.append("{:<18}{}".format(label[:17], cells))
    lines.append("")
    lines.append("MAE against the human distribution; lower is better.")
    return lines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="source", type=Path, help="Results dir to copy.")
    parser.add_argument("--label", help="Name to file the run under.")
    parser.add_argument(
        "--records-dir",
        type=Path,
        default=RECORDS_DIR,
        help="Where records live; defaults to effect_algebra/records.",
    )
    parser.add_argument("--summary", action="store_true", help="Print a cross-run table.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.records_dir.mkdir(parents=True, exist_ok=True)

    if args.source:
        if not args.label:
            raise SystemExit("--label is required with --from")
        entry = collect(args.source, args.label, args.records_dir)
        refresh_index(args.records_dir)
        print(
            json.dumps(
                {"collected": entry["label"], "files": len(entry["files"])},
                indent=2,
            )
        )

    if args.summary or not args.source:
        print("\n".join(summary_table(args.records_dir)))


if __name__ == "__main__":
    main()
