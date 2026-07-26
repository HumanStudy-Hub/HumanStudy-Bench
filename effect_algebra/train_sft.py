"""Train a chosen-answer QLoRA-SFT baseline on the same A/B records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from .datasets import load_jsonl
from .modeling import (
    DEFAULT_BASE_MODEL,
    enforce_prompt_limit,
    load_4bit_base,
    load_tokenizer,
    make_lora_config,
    precision_flags,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in load_jsonl(path):
        if row.get("effect") not in {"A", "B"}:
            raise ValueError("{} contains held-out effect {}".format(path, row.get("effect")))
        records.append({"prompt": row["prompt"], "completion": row["chosen"]})
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2056)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _parser().parse_args()
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    train_rows = load_jsonl(args.train_file)
    eval_rows = load_jsonl(args.eval_file)
    train_records = _records(args.train_file)
    eval_records = _records(args.eval_file)
    tokenizer = load_tokenizer(args.base_model)
    prompt_limit = args.max_length - 8
    token_lengths = {
        "train": enforce_prompt_limit(
            tokenizer,
            train_rows,
            prompt_limit,
            dataset_name=str(args.train_file),
        ),
        "eval": enforce_prompt_limit(
            tokenizer,
            eval_rows,
            prompt_limit,
            dataset_name=str(args.eval_file),
        ),
    }
    model = load_4bit_base(args.base_model, for_training=True)
    peft_config = make_lora_config(
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )
    precision = precision_flags()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = SFTConfig(
        output_dir=str(output_dir),
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        completion_only_loss=True,
        optim="paged_adamw_8bit",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        bf16=precision["bf16"],
        fp16=precision["fp16"],
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(train_records),
        eval_dataset=Dataset.from_list(eval_records),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    manifest = {
        "method": "QLoRA-SFT",
        "run_name": args.run_name,
        "base_model": args.base_model,
        "train_file": str(args.train_file.resolve()),
        "train_sha256": _sha256(args.train_file),
        "train_rows": len(train_records),
        "eval_file": str(args.eval_file.resolve()),
        "eval_sha256": _sha256(args.eval_file),
        "eval_rows": len(eval_records),
        "token_lengths": token_lengths,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
        },
        "seed": args.seed,
        "log_history": trainer.state.log_history,
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
