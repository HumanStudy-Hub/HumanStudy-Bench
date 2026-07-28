"""Train a QLoRA adapter to match human response distributions.

The objective is a two-class cross-entropy on the answer token:

    loss = - sum_c  p_human(c) * log softmax([logit_X, logit_Y])[c]

Its optimum is exactly the human distribution, which is what makes it the right
tool for a calibration target. Pairwise DPO cannot make the same claim: at its
optimum the model log-odds equal the base log-odds plus logit(p_human)/beta, a
shift whose sign always follows the human majority. A base model that is
already more extreme than the humans in that direction therefore cannot be
corrected by any positive beta, only pushed further out. See
`human_priors.dpo_reachable` and the Gate 0 overshoot diagnostic.

Practical consequences on a single Colab GPU: no reference model to hold or
precompute, roughly half the memory of DPO and about twice the throughput, and
no beta to sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .datasets import load_jsonl
from .evaluate_choices import answer_token_plan
from .modeling import (
    DEFAULT_BASE_MODEL,
    load_4bit_base,
    load_tokenizer,
    make_lora_config,
    precision_flags,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_examples(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    max_prompt_length: int,
    dataset_name: str,
    require_trainable: bool,
) -> List[Dict[str, Any]]:
    """Turn rows into (context, two answer tokens, target distribution)."""

    examples: List[Dict[str, Any]] = []
    longest = 0
    for row in rows:
        if require_trainable and not row.get("trainable"):
            raise ValueError(
                "{} [{}]: row is not trainable; evaluation sets must never "
                "reach a training entry point".format(dataset_name, row.get("id"))
            )
        distribution = row.get("human_probability_by_code")
        if not isinstance(distribution, dict):
            raise ValueError(
                "{} [{}]: row carries no human distribution to fit".format(
                    dataset_name,
                    row.get("id"),
                )
            )
        context, token_ids = answer_token_plan(tokenizer, row["prompt"], row["options"])
        longest = max(longest, len(context))
        if len(context) > max_prompt_length:
            raise ValueError(
                "{} has a {}-token context, exceeding --max-prompt-length {}. "
                "Raise the limit; truncating a stateful B episode is forbidden.".format(
                    dataset_name,
                    len(context),
                    max_prompt_length,
                )
            )
        examples.append(
            {
                "id": row["id"],
                "input_ids": context,
                "token_x": token_ids["X"],
                "token_y": token_ids["Y"],
                "target_x": float(distribution["X"]),
                "target_y": float(distribution["Y"]),
            }
        )
    if not examples:
        raise ValueError("{} produced no trainable examples".format(dataset_name))
    print(
        "{}: {} examples, longest context {} tokens".format(
            dataset_name,
            len(examples),
            longest,
        ),
        flush=True,
    )
    return examples


class SoftLabelCollator:
    """Right-pad contexts and remember where each sequence's last token sits."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        import torch

        lengths = [len(feature["input_ids"]) for feature in features]
        width = max(lengths)
        input_ids = torch.full(
            (len(features), width),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((len(features), width), dtype=torch.long)
        for index, feature in enumerate(features):
            length = lengths[index]
            input_ids[index, :length] = torch.tensor(
                feature["input_ids"], dtype=torch.long
            )
            attention_mask[index, :length] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "last_index": torch.tensor(
                [length - 1 for length in lengths], dtype=torch.long
            ),
            "answer_tokens": torch.tensor(
                [[feature["token_x"], feature["token_y"]] for feature in features],
                dtype=torch.long,
            ),
            "target_distribution": torch.tensor(
                [[feature["target_x"], feature["target_y"]] for feature in features],
                dtype=torch.float32,
            ),
        }


def soft_label_loss(model: Any, batch: Mapping[str, Any]) -> Any:
    """Two-class cross-entropy between model and human answer distributions."""

    import torch

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    rows = torch.arange(batch["input_ids"].shape[0], device=outputs.logits.device)
    final = outputs.logits[rows, batch["last_index"].to(outputs.logits.device)]
    # Restricting the softmax to the two candidate tokens matches how the model
    # is scored at evaluation time: only the relative preference is fitted, and
    # no gradient is spent moving probability mass off unrelated vocabulary.
    pair = final.gather(1, batch["answer_tokens"].to(final.device)).float()
    log_probabilities = torch.log_softmax(pair, dim=-1)
    targets = batch["target_distribution"].to(log_probabilities.device)
    return -(targets * log_probabilities).sum(dim=-1).mean()


def evaluate_calibration(
    model: Any,
    examples: Sequence[Mapping[str, Any]],
    collator: SoftLabelCollator,
    *,
    batch_size: int,
) -> Dict[str, float]:
    """Mean absolute error against the human distribution on held-out rows."""

    import torch

    model.eval()
    absolute: List[float] = []
    losses: List[float] = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = collator(examples[start : start + batch_size])
            device = next(model.parameters()).device
            batch = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            rows = torch.arange(batch["input_ids"].shape[0], device=outputs.logits.device)
            final = outputs.logits[rows, batch["last_index"]]
            pair = final.gather(1, batch["answer_tokens"]).float()
            log_probabilities = torch.log_softmax(pair, dim=-1)
            targets = batch["target_distribution"]
            losses.append(
                float((-(targets * log_probabilities).sum(dim=-1)).mean().item())
            )
            model_x = log_probabilities[:, 0].exp()
            absolute.extend(
                (model_x - targets[:, 0]).abs().detach().cpu().tolist()
            )
    model.train()
    return {
        "mae": sum(absolute) / len(absolute),
        "cross_entropy": sum(losses) / len(losses),
        "rows": float(len(absolute)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--resume-from-checkpoint")
    return parser


def main() -> None:
    args = _parser().parse_args()
    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainerCallback, TrainingArguments

    tokenizer = load_tokenizer(args.base_model)
    train_rows = load_jsonl(args.train_file)
    eval_rows = load_jsonl(args.eval_file)
    train_examples = build_examples(
        tokenizer,
        train_rows,
        max_prompt_length=args.max_prompt_length,
        dataset_name=str(args.train_file),
        require_trainable=True,
    )
    # The eval file only has to expose a human distribution; it must not be
    # trainable, and nothing here writes gradients from it.
    eval_examples = build_examples(
        tokenizer,
        eval_rows,
        max_prompt_length=args.max_prompt_length,
        dataset_name=str(args.eval_file),
        require_trainable=False,
    )

    model = load_4bit_base(args.base_model, for_training=True)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    model = get_peft_model(
        model,
        make_lora_config(rank=args.rank, alpha=args.alpha, dropout=args.dropout),
    )
    model.print_trainable_parameters()

    collator = SoftLabelCollator(tokenizer.pad_token_id)
    precision = precision_flags()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class SoftLabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            loss = soft_label_loss(model, inputs)
            return (loss, None) if return_outputs else loss

    calibration_history: List[Dict[str, Any]] = []

    class CalibrationCallback(TrainerCallback):
        def on_epoch_end(self, arguments, state, control, **kwargs):
            metrics = evaluate_calibration(
                model,
                eval_examples,
                collator,
                batch_size=args.eval_batch_size,
            )
            metrics["epoch"] = state.epoch
            calibration_history.append(metrics)
            print(json.dumps({"calibration_eval": metrics}), flush=True)

    training_arguments = TrainingArguments(
        output_dir=str(output_dir),
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        bf16=precision["bf16"],
        fp16=precision["fp16"],
        remove_unused_columns=False,
        label_names=[],
    )
    trainer = SoftLabelTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_examples,
        data_collator=collator,
        callbacks=[CalibrationCallback()],
    )

    baseline = evaluate_calibration(
        model,
        eval_examples,
        collator,
        batch_size=args.eval_batch_size,
    )
    print(json.dumps({"calibration_eval_before_training": baseline}), flush=True)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    manifest = {
        "method": "QLoRA soft-label distribution matching",
        "objective": "two-class cross-entropy on the answer token",
        "run_name": args.run_name,
        "base_model": args.base_model,
        "train_file": str(args.train_file.resolve()),
        "train_sha256": _sha256(args.train_file),
        "train_rows": len(train_examples),
        "eval_file": str(args.eval_file.resolve()),
        "eval_sha256": _sha256(args.eval_file),
        "eval_rows": len(eval_examples),
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "max_prompt_length": args.max_prompt_length,
        },
        "lora": {"rank": args.rank, "alpha": args.alpha, "dropout": args.dropout},
        "seed": args.seed,
        "calibration_before_training": baseline,
        "calibration_history": calibration_history,
        "log_history": trainer.state.log_history,
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
