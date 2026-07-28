"""Pairwise DPO baseline against the soft-label objective in `train_soft`.

Labels here already encode the human proportion: within a bucket the chosen
side is flipped for the share of rows that deviated in the published data, so
no prompt is duplicated to express a ratio.

This is a *baseline*, not the main method. At the DPO optimum,

    logit(p_model) = logit(p_base) + logit(p_human) / beta,

and the shift always carries the sign of the human majority. When the base
model is already more extreme than the humans in that direction, no positive
beta reaches the target; it only overshoots further. Sweep beta, but read the
Gate 0 overshoot diagnostic first: it says how much of the evaluation set is
structurally out of reach before any compute is spent.
"""

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


def _training_records(path: Path) -> List[Dict[str, Any]]:
    """Read preference pairs, refusing anything an evaluation set produced.

    The guard is the per-row `trainable` flag rather than the effect name.
    Gate 1 trains on C scenarios on purpose, so banning effect C outright would
    either block the ceiling measurement or invite someone to disable the check;
    the flag distinguishes a cross-validation training fold from the held-out
    evaluation set, which is the distinction that actually matters. Evaluation
    rows also carry no chosen/rejected pair at all, so this is the second of two
    independent locks.
    """

    rows = load_jsonl(path)
    records: List[Dict[str, Any]] = []
    for row in rows:
        if not row.get("trainable"):
            raise ValueError(
                "{} [{}] is not trainable; evaluation rows cannot train adapters".format(
                    path,
                    row.get("id"),
                )
            )
        if row.get("target_code") not in {"X", "Y"}:
            raise ValueError("{} [{}] has no preference label".format(path, row.get("id")))
        if "chosen" not in row or "rejected" not in row:
            raise ValueError(
                "{} [{}] carries no preference pair".format(path, row.get("id"))
            )
        records.append(
            {
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            }
        )
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss-type", default="sigmoid", choices=("sigmoid", "ipo"))
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-completion-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--no-precompute-ref-log-probs",
        action="store_false",
        dest="precompute_ref_log_probs",
        help="Compute reference log probabilities during every training step.",
    )
    parser.set_defaults(precompute_ref_log_probs=True)
    parser.add_argument("--resume-from-checkpoint")
    return parser


def main() -> None:
    args = _parser().parse_args()
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    train_rows = load_jsonl(args.train_file)
    train_records = _training_records(args.train_file)
    tokenizer = load_tokenizer(args.base_model)
    token_lengths = {
        "train": enforce_prompt_limit(
            tokenizer,
            train_rows,
            args.max_prompt_length,
            dataset_name=str(args.train_file),
        ),
    }
    # Optional: the cross-validation folds have no trainable dev split, and
    # their real measurement is `evaluate_suite` on the held-out fold anyway.
    eval_records = None
    if args.eval_file is not None:
        eval_records = _training_records(args.eval_file)
        token_lengths["eval"] = enforce_prompt_limit(
            tokenizer,
            load_jsonl(args.eval_file),
            args.max_prompt_length,
            dataset_name=str(args.eval_file),
        )
    model = load_4bit_base(
        args.base_model,
        for_training=True,
    )
    peft_config = make_lora_config(
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )
    precision = precision_flags()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DPOConfig(
        output_dir=str(output_dir),
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        loss_type=args.loss_type,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_length=args.max_prompt_length + args.max_completion_length,
        truncation_mode="keep_end",
        precompute_ref_log_probs=args.precompute_ref_log_probs,
        optim="paged_adamw_8bit",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy="epoch" if args.eval_file is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        bf16=precision["bf16"],
        fp16=precision["fp16"],
        remove_unused_columns=True,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=Dataset.from_list(train_records),
        eval_dataset=(
            Dataset.from_list(eval_records) if eval_records is not None else None
        ),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    manifest = {
        "method": "QLoRA-DPO",
        "run_name": args.run_name,
        "base_model": args.base_model,
        "train_file": str(args.train_file.resolve()),
        "train_sha256": _sha256(args.train_file),
        "train_rows": len(train_records),
        "eval_file": str(args.eval_file.resolve()) if args.eval_file else None,
        "eval_sha256": _sha256(args.eval_file) if args.eval_file else None,
        "eval_rows": len(eval_records) if eval_records is not None else 0,
        "token_lengths": token_lengths,
        "dpo": {
            "loss_type": args.loss_type,
            "beta": args.beta,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "precompute_ref_log_probs": args.precompute_ref_log_probs,
        },
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
