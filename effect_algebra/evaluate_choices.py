"""Evaluate base or adapter models with forced-choice conditional log probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .datasets import load_jsonl, preference_probability
from .modeling import (
    DEFAULT_BASE_MODEL,
    load_4bit_base,
    load_adapter_for_evaluation,
    load_tokenizer,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _device(model: Any) -> Any:
    return next(model.parameters()).device


def score_completion(
    model: Any,
    tokenizer: Any,
    prompt: Sequence[Mapping[str, str]],
    completion: str,
) -> float:
    """Return summed assistant-token log probability for one completion."""

    import torch

    prompt_ids = tokenizer.apply_chat_template(
        list(prompt),
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        list(prompt) + [{"role": "assistant", "content": completion}],
        tokenize=True,
        add_generation_prompt=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "chat template does not preserve the prompt prefix; cannot score safely"
        )
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("completion produced no scoreable tokens")

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=_device(model))
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
    start = len(prompt_ids)
    positions = torch.arange(start - 1, len(full_ids) - 1, device=input_ids.device)
    targets = input_ids[0, start:]
    selected = log_probs[0, positions, targets]
    return float(selected.sum().item())


def _mean_or_none(values: Iterable[float]) -> Optional[float]:
    materialized = list(values)
    return mean(materialized) if materialized else None


def _group_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    labeled = [row for row in rows if row["target_code"] in {"X", "Y"}]
    return {
        "rows": len(rows),
        "labeled_rows": len(labeled),
        "accuracy": _mean_or_none(
            1.0 if row["predicted_code"] == row["target_code"] else 0.0
            for row in labeled
        ),
        "mean_target_probability": _mean_or_none(
            float(row["probability_by_code"][row["target_code"]])
            for row in labeled
        ),
        "mean_preference_margin": _mean_or_none(
            float(row["log_probability_by_code"][row["target_code"]])
            - float(row["log_probability_by_code"][
                "Y" if row["target_code"] == "X" else "X"
            ])
            for row in labeled
        ),
        "decision_x_rate": _mean_or_none(
            1.0 if row["predicted_code"] == "X" else 0.0 for row in rows
        ),
    }


def summarize_scored_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_effect: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_category: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_authority: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_effect[str(row["effect"])].append(row)
        category = row.get("category")
        if category:
            by_category[str(category)].append(row)
        condition = row.get("authority_condition")
        if condition:
            by_authority[str(condition)].append(row)

    human_rows = [
        row for row in rows if isinstance(row.get("human_probability_by_code"), dict)
    ]
    human_weight = sum(float(row.get("human_n") or 0) for row in human_rows)
    human_mae = None
    human_cross_entropy = None
    if human_rows and human_weight:
        human_mae = sum(
            float(row["human_n"])
            * abs(
                float(row["probability_by_code"]["X"])
                - float(row["human_probability_by_code"]["X"])
            )
            for row in human_rows
        ) / human_weight
        human_cross_entropy = sum(
            float(row["human_n"])
            * (
                -float(row["human_probability_by_code"]["X"])
                * math.log(max(float(row["probability_by_code"]["X"]), 1e-12))
                -float(row["human_probability_by_code"]["Y"])
                * math.log(max(float(row["probability_by_code"]["Y"]), 1e-12))
            )
            for row in human_rows
        ) / human_weight

    authority_rows = [
        row for row in rows if row.get("medical_director_code") in {"X", "Y"}
    ]
    agreement_rows = [
        row for row in rows if row.get("agreeing_advisor_code") in {"X", "Y"}
    ]
    return {
        "overall": _group_metrics(rows),
        "by_effect": {
            key: _group_metrics(value) for key, value in sorted(by_effect.items())
        },
        "by_category": {
            key: _group_metrics(value) for key, value in sorted(by_category.items())
        },
        "by_authority_condition": {
            key: _group_metrics(value) for key, value in sorted(by_authority.items())
        },
        "human_distribution": {
            "rows": len(human_rows),
            "weighted_probability_mae": human_mae,
            "weighted_cross_entropy": human_cross_entropy,
        },
        "authority": {
            "rows": len(authority_rows),
            "hard_alignment_rate": _mean_or_none(
                1.0
                if row["predicted_code"] == row["medical_director_code"]
                else 0.0
                for row in authority_rows
            ),
            "mean_alignment_probability": _mean_or_none(
                float(row["probability_by_code"][row["medical_director_code"]])
                for row in authority_rows
            ),
        },
        "advisor_agreement": {
            "rows": len(agreement_rows),
            "hard_agreeing_choice_rate": _mean_or_none(
                1.0
                if row["predicted_code"] == row["agreeing_advisor_code"]
                else 0.0
                for row in agreement_rows
            ),
            "mean_agreeing_choice_probability": _mean_or_none(
                float(row["probability_by_code"][row["agreeing_advisor_code"]])
                for row in agreement_rows
            ),
        },
    }


def evaluate_rows(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        logp_x = score_completion(model, tokenizer, row["prompt"], row["options"]["X"])
        logp_y = score_completion(model, tokenizer, row["prompt"], row["options"]["Y"])
        probability_x = preference_probability(logp_x, logp_y)
        predicted_code = "X" if probability_x >= 0.5 else "Y"
        metadata = row.get("metadata", {})
        agreeing_advisor_code = None
        code_to_advisor = metadata.get("code_to_advisor", {})
        advisor_name_to_type = metadata.get("advisor_name_to_type", {})
        for code, advisor_name in code_to_advisor.items():
            if advisor_name_to_type.get(advisor_name) == "agreeing":
                agreeing_advisor_code = code
                break
        scored.append(
            {
                "id": row["id"],
                "effect": row["effect"],
                "split": row["split"],
                "target_code": row.get("target_code"),
                "predicted_code": predicted_code,
                "correct": (
                    predicted_code == row["target_code"]
                    if row.get("target_code") in {"X", "Y"}
                    else None
                ),
                "log_probability_by_code": {"X": logp_x, "Y": logp_y},
                "probability_by_code": {
                    "X": probability_x,
                    "Y": 1.0 - probability_x,
                },
                "category": metadata.get("category"),
                "authority_condition": metadata.get("authority_condition"),
                "medical_director_code": metadata.get("medical_director_code"),
                "agreeing_advisor_code": agreeing_advisor_code,
                "human_probability_by_code": metadata.get("human_probability_by_code"),
                "human_n": metadata.get("human_n"),
            }
        )
        if index % 25 == 0 or index == len(rows):
            print("scored {}/{}".format(index, len(rows)), flush=True)
    return scored


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-rows", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    rows = load_jsonl(args.dataset)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    tokenizer = load_tokenizer(args.base_model)
    model = load_4bit_base(args.base_model, for_training=False)
    model = load_adapter_for_evaluation(model, args.adapter)
    scored_rows = evaluate_rows(model, tokenizer, rows)
    result = {
        "schema_version": 1,
        "model_label": args.model_label,
        "base_model": args.base_model,
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset),
        "summary": summarize_scored_rows(scored_rows),
        "rows": scored_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
