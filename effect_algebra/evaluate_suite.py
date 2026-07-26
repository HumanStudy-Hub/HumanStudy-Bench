"""Load one base/adapter model once and evaluate multiple choice datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

from .datasets import load_jsonl
from .evaluate_choices import evaluate_rows, summarize_scored_rows
from .modeling import (
    DEFAULT_BASE_MODEL,
    load_4bit_base,
    load_adapter_for_evaluation,
    load_tokenizer,
)


def _dataset_argument(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be LABEL=/path/to/file.jsonl")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--dataset label and path cannot be empty")
    return label.strip(), Path(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=_dataset_argument,
        required=True,
        help="Repeatable LABEL=/path/to/file.jsonl.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-rows", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    tokenizer = load_tokenizer(args.base_model)
    model = load_4bit_base(args.base_model, for_training=False)
    model = load_adapter_for_evaluation(model, args.adapter)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}

    for label, dataset_path in args.dataset:
        rows = load_jsonl(dataset_path)
        if args.max_rows is not None:
            rows = rows[: args.max_rows]
        print(
            "evaluating {} on {} ({} rows)".format(
                args.model_label,
                label,
                len(rows),
            ),
            flush=True,
        )
        scored_rows = evaluate_rows(model, tokenizer, rows)
        result = {
            "schema_version": 1,
            "model_label": args.model_label,
            "base_model": args.base_model,
            "adapter": str(args.adapter.resolve()) if args.adapter else None,
            "dataset_label": label,
            "dataset": str(dataset_path.resolve()),
            "dataset_sha256": _sha256(dataset_path),
            "summary": summarize_scored_rows(scored_rows),
            "rows": scored_rows,
        }
        output_path = args.output_dir / "{}.json".format(label)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        outputs[label] = str(output_path)

    suite_manifest = {
        "model_label": args.model_label,
        "base_model": args.base_model,
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "outputs": outputs,
    }
    (args.output_dir / "suite_manifest.json").write_text(
        json.dumps(suite_manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(suite_manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
