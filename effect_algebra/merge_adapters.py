"""Compose effect-A and effect-B LoRA deltas with PEFT adapter arithmetic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .modeling import (
    DEFAULT_BASE_MODEL,
    load_4bit_base,
    load_tokenizer,
    resolve_adapter_dir,
    validate_adapter_compatibility,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-a", type=Path, required=True)
    parser.add_argument("--adapter-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--weight-a", type=float, default=1.0)
    parser.add_argument("--weight-b", type=float, default=1.0)
    parser.add_argument(
        "--combination-type",
        choices=("cat", "linear", "ties", "ties_svd"),
        default="cat",
    )
    parser.add_argument("--density", type=float, default=0.5)
    return parser


def main() -> None:
    args = _parser().parse_args()
    from peft import PeftModel

    adapter_a = resolve_adapter_dir(args.adapter_a)
    adapter_b = resolve_adapter_dir(args.adapter_b)
    signature = validate_adapter_compatibility([adapter_a, adapter_b])
    configured_base = signature["base_model_name_or_path"]
    if configured_base and configured_base != args.base_model:
        raise ValueError(
            "--base-model {} does not match adapter base {}".format(
                args.base_model,
                configured_base,
            )
        )

    model = load_4bit_base(args.base_model, for_training=False)
    model = PeftModel.from_pretrained(
        model,
        str(adapter_a),
        adapter_name="effect_a",
        is_trainable=False,
    )
    model.load_adapter(str(adapter_b), adapter_name="effect_b", is_trainable=False)
    kwargs = {}
    if args.combination_type in {"ties", "ties_svd"}:
        kwargs["density"] = args.density
    model.add_weighted_adapter(
        adapters=["effect_a", "effect_b"],
        weights=[args.weight_a, args.weight_b],
        adapter_name="a_plus_b",
        combination_type=args.combination_type,
        **kwargs
    )
    model.set_adapter("a_plus_b")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), selected_adapters=["a_plus_b"])
    load_tokenizer(args.base_model).save_pretrained(str(output_dir))
    resolved_output = resolve_adapter_dir(output_dir)
    manifest = {
        "method": "PEFT_add_weighted_adapter",
        "combination_type": args.combination_type,
        "exact_delta_sum": args.combination_type == "cat",
        "base_model": args.base_model,
        "adapter_a": str(adapter_a.resolve()),
        "adapter_b": str(adapter_b.resolve()),
        "weight_a": args.weight_a,
        "weight_b": args.weight_b,
        "input_signature": signature,
        "resolved_adapter_dir": str(resolved_output),
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
