"""Score models against human response distributions with forced-choice logits.

The primary metric is distance to the human distribution (MAE), not accuracy
against a normative answer. Accuracy is still reported, because "closer to
humans" and "more Bayesian" can move in opposite directions and collapsing them
into one number hides that.

Scoring reads the two answer-token logits from a single forward pass. The two
completions differ in exactly one token, so one pass yields both, and the
normalized two-way probability is exactly the quantity being compared against
the human proportion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .datasets import load_jsonl, preference_probability
from .human_priors import binomial_noise_floor, dpo_reachable, logit, trivial_baselines
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


def answer_token_plan(
    tokenizer: Any,
    prompt: Sequence[Mapping[str, str]],
    options: Mapping[str, str],
) -> Tuple[List[int], Dict[str, int]]:
    """Locate the single token position where the two completions diverge.

    Returns the shared context to run a forward pass on, plus the token id each
    response code contributes at the next position. Raising here rather than
    guessing keeps a tokenizer change from silently corrupting every score.
    """

    encoded: Dict[str, List[int]] = {}
    for code in ("X", "Y"):
        encoded[code] = tokenizer.apply_chat_template(
            list(prompt) + [{"role": "assistant", "content": options[code]}],
            tokenize=True,
            add_generation_prompt=False,
        )
    ids_x, ids_y = encoded["X"], encoded["Y"]
    if len(ids_x) != len(ids_y):
        raise ValueError(
            "the two completions tokenize to different lengths ({} vs {}); "
            "single-pass scoring needs them to differ in one position".format(
                len(ids_x),
                len(ids_y),
            )
        )
    divergent = [index for index in range(len(ids_x)) if ids_x[index] != ids_y[index]]
    if len(divergent) != 1:
        raise ValueError(
            "expected exactly one divergent token between completions, found {}".format(
                len(divergent)
            )
        )
    position = divergent[0]
    prompt_ids = tokenizer.apply_chat_template(
        list(prompt),
        tokenize=True,
        add_generation_prompt=True,
    )
    if ids_x[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template does not preserve the prompt prefix")
    if position < len(prompt_ids):
        raise ValueError("completions diverge inside the prompt, not the answer")
    return ids_x[:position], {"X": ids_x[position], "Y": ids_y[position]}


def score_answer_tokens(
    model: Any,
    tokenizer: Any,
    prompt: Sequence[Mapping[str, str]],
    options: Mapping[str, str],
) -> Dict[str, float]:
    """Log probabilities of the two answer tokens from one forward pass."""

    import torch

    context, token_ids = answer_token_plan(tokenizer, prompt, options)
    input_ids = torch.tensor([context], dtype=torch.long, device=_device(model))
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits[0, -1].float(), dim=-1)
    return {code: float(log_probs[token_id].item()) for code, token_id in token_ids.items()}


def score_completion(
    model: Any,
    tokenizer: Any,
    prompt: Sequence[Mapping[str, str]],
    completion: str,
) -> float:
    """Summed assistant-token log probability for one completion.

    Kept as a cross-check for `score_answer_tokens`: it costs one forward pass
    per option instead of one per row, and additionally scores the turn-ending
    tokens, which are not part of the choice.
    """

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


def response_code_bias(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """How much of the model's answer is decided by the letter, not the content.

    Response codes are assigned independently of the label, so a content-driven
    model should sit at P(X) = 0.5 on average. Any systematic departure is a
    preference for the letter itself, and it corrupts both the calibration
    metric and any forced-choice probe. Reported in log odds because that is the
    scale on which it is additive with the content signal.
    """

    if not rows:
        return {"rows": 0}
    log_odds = sorted(
        logit(float(row["probability_by_code"]["X"])) for row in rows
    )
    middle = len(log_odds) // 2
    median = (
        log_odds[middle]
        if len(log_odds) % 2
        else 0.5 * (log_odds[middle - 1] + log_odds[middle])
    )
    return {
        "rows": len(rows),
        "mean_probability_x": mean(
            float(row["probability_by_code"]["X"]) for row in rows
        ),
        "median_log_odds_x": median,
        "argmax_x_rate": mean(
            1.0 if row["predicted_code"] == "X" else 0.0 for row in rows
        ),
    }


def _reference_log_odds(row: Mapping[str, Any], reference_code: str) -> float:
    """Log odds favouring the reference option, taken from raw log probabilities.

    The scorer keeps the two answer-token log probabilities, whose difference is
    the log odds exactly. Recovering it from the normalized probability instead
    loses precision once the model saturates: at a log-odds of 17 the probability
    rounds to 1.0 and the magnitude is gone. Reference predictors carry no real
    log probabilities, so those fall back to the probability.
    """

    other = "Y" if reference_code == "X" else "X"
    log_probabilities = row.get("log_probability_by_code") or {}
    first = log_probabilities.get(reference_code)
    second = log_probabilities.get(other)
    # Two log probabilities of exactly 0.0 would mean both options are certain,
    # which only happens for the placeholder a reference predictor writes.
    if first is not None and second is not None and (first != 0.0 or second != 0.0):
        return float(first) - float(second)
    return logit(float(row["probability_by_code"][reference_code]))


def merge_mirror_pairs(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Average each item over both letter assignments to cancel letter bias.

    Every evaluation item is scored twice, once with each mapping of response
    codes onto the two options. The letter preference is an additive term in log
    odds: writing b for it and c for the content signal, one frame scores b + c
    and the mirror scores c - b, so averaging *log odds* returns exactly c.

    Averaging probabilities instead only works while b is small. Qwen2.5-14B
    carries b = +8.2 on the B probes, where the sigmoid is saturated: both
    frames read as near-certainty, their probabilities average to 0.5 whatever
    the content, and the signal is destroyed rather than recovered. That showed
    up as a probe MAE of 0.338, which is just |0.5 - 0.83| against the human
    rate.

    Items with no mirror pass through unchanged, so a partially mirrored set
    still evaluates, and the report says how many items were actually paired.
    """

    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    unpaired: List[Dict[str, Any]] = []
    for row in rows:
        pair_id = row.get("pair_id")
        if pair_id:
            groups[str(pair_id)].append(row)
        else:
            unpaired.append(dict(row))

    merged: List[Dict[str, Any]] = list(unpaired)
    paired_groups = 0
    for _pair_id, group in sorted(groups.items()):
        if len(group) == 1:
            merged.append(dict(group[0]))
            continue
        frame = dict(group[0])
        reference_code = frame.get("reference_code")
        if reference_code not in {"X", "Y"}:
            merged.extend(dict(row) for row in group)
            continue
        # Express every member in the first row's code frame before averaging.
        mean_log_odds = mean(
            _reference_log_odds(row, str(row["reference_code"])) for row in group
        )
        probability_reference = 1.0 / (1.0 + math.exp(-max(min(mean_log_odds, 700.0), -700.0)))
        other = "Y" if reference_code == "X" else "X"
        frame["probability_by_code"] = {
            reference_code: probability_reference,
            other: 1.0 - probability_reference,
        }
        frame["log_probability_by_code"] = {
            reference_code: math.log(max(probability_reference, 1e-12)),
            other: math.log(max(1.0 - probability_reference, 1e-12)),
        }
        frame["mean_log_odds_reference"] = mean_log_odds
        frame["predicted_code"] = (
            reference_code if probability_reference >= 0.5 else other
        )
        if frame.get("target_code") in {"X", "Y"}:
            frame["correct"] = (
                None
                if probability_reference == 0.5
                else frame["predicted_code"] == frame["target_code"]
            )
        frame["mirror_size"] = len(group)
        merged.append(frame)
        paired_groups += 1

    return merged, {
        "input_rows": len(rows),
        "merged_rows": len(merged),
        "paired_items": paired_groups,
        "unpaired_items": len(merged) - paired_groups,
    }


def _mean_or_none(values: Iterable[float]) -> Optional[float]:
    materialized = list(values)
    return mean(materialized) if materialized else None


def _calibration_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Distance between the model distribution and the human distribution."""

    scored = [
        row for row in rows if isinstance(row.get("human_probability_by_code"), dict)
    ]
    if not scored:
        return {"rows": 0, "mae": None, "rmse": None, "cross_entropy": None}

    absolute = []
    squared = []
    cross_entropy = []
    for row in scored:
        model_x = float(row["probability_by_code"]["X"])
        human_x = float(row["human_probability_by_code"]["X"])
        human_y = float(row["human_probability_by_code"]["Y"])
        absolute.append(abs(model_x - human_x))
        squared.append((model_x - human_x) ** 2)
        cross_entropy.append(
            -human_x * math.log(max(model_x, 1e-12))
            - human_y * math.log(max(1.0 - model_x, 1e-12))
        )
    return {
        "rows": len(scored),
        "mae": mean(absolute),
        "rmse": math.sqrt(mean(squared)),
        "cross_entropy": mean(cross_entropy),
    }


def _calibration_scale(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The floor and the trivial ceilings any MAE has to be read against."""

    scored = [
        row for row in rows if isinstance(row.get("human_probability_by_code"), dict)
    ]
    if not scored:
        return {}
    proportions = [
        float(row["human_probability_by_code"]["X"]) for row in scored
    ]
    counts = [int(row.get("human_n") or 0) for row in scored]
    scale: Dict[str, Any] = {"trivial_baselines": trivial_baselines(proportions)}
    if all(count > 0 for count in counts):
        scale["noise_floor_mae"] = binomial_noise_floor(proportions, counts)
    return scale


def _overshoot_diagnostic(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """How often proportional DPO structurally cannot reach the human value.

    At the DPO optimum the model log-odds move by logit(p_human)/beta, whose
    sign follows the human majority. When the model already sits further from
    0.5 than the humans do in that same direction, no positive beta closes the
    gap; it only widens it. Counting those rows before training decides whether
    the pairwise objective is usable at all.
    """

    scored = [
        row for row in rows if isinstance(row.get("human_probability_by_code"), dict)
    ]
    if not scored:
        return {"rows": 0}
    unreachable = []
    signed_gap = []
    for row in scored:
        model_x = float(row["probability_by_code"]["X"])
        human_x = float(row["human_probability_by_code"]["X"])
        unreachable.append(0.0 if dpo_reachable(human_x, model_x) else 1.0)
        signed_gap.append(abs(logit(model_x)) - abs(logit(human_x)))
    return {
        "rows": len(scored),
        "dpo_unreachable_rate": mean(unreachable),
        "mean_excess_confidence_logits": mean(signed_gap),
        "model_more_extreme_rate": mean(
            1.0 if value > 0 else 0.0 for value in signed_gap
        ),
    }


def _tie_aware_correctness(row: Mapping[str, Any]) -> float:
    """Score a forced choice, splitting an exact tie instead of awarding it.

    Taking the argmax of two equal probabilities silently resolves the tie in a
    fixed direction. Where the target happens to sit in that direction, a model
    carrying no information at all scores a perfect 1.0, which is how a constant
    0.5 predictor came out looking flawless on the B probes.
    """

    probability_target = float(row["probability_by_code"][str(row["target_code"])])
    if probability_target > 0.5:
        return 1.0
    if probability_target < 0.5:
        return 0.0
    return 0.5


def _normative_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    labeled = [row for row in rows if row.get("target_code") in {"X", "Y"}]
    return {
        "rows": len(rows),
        "labeled_rows": len(labeled),
        "accuracy": _mean_or_none(_tie_aware_correctness(row) for row in labeled),
        "tied_rows": sum(
            1
            for row in labeled
            if float(row["probability_by_code"][str(row["target_code"])]) == 0.5
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


def _group_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metrics = _normative_metrics(rows)
    metrics.update(_calibration_metrics(rows))
    return metrics


def _grouped(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value:
            buckets[str(value)].append(row)
    return {name: _group_metrics(group) for name, group in sorted(buckets.items())}


def summarize_scored_rows(
    scored_rows: Sequence[Mapping[str, Any]],
    *,
    symmetrize: bool = True,
) -> Dict[str, Any]:
    # Letter bias is measured on the raw rows and removed before anything else,
    # because it inflates every calibration number downstream.
    raw_bias = response_code_bias(scored_rows)
    if symmetrize:
        rows, mirror_report = merge_mirror_pairs(scored_rows)
    else:
        rows, mirror_report = list(scored_rows), {"paired_items": 0}

    authority_rows = [
        row for row in rows if row.get("medical_director_code") in {"X", "Y"}
    ]
    agreement_rows = [
        row for row in rows if row.get("agreeing_advisor_code") in {"X", "Y"}
    ]
    indifference_rows = [row for row in rows if row.get("indifference_scenario")]

    summary: Dict[str, Any] = {
        "response_code_bias": {
            # Measured on the raw rows only. After merging, the "X" slot holds
            # the item's reference *option*, not the letter X, so the same
            # statistic there would measure content preference, not bias.
            "raw": raw_bias,
            "mirror": mirror_report,
            "symmetrized": symmetrize,
        },
        "calibration": {
            **_calibration_metrics(rows),
            "scale": _calibration_scale(rows),
            "by_bucket": {
                name: _calibration_metrics(group)
                for name, group in sorted(
                    _collect(rows, "bucket").items()
                )
            },
            "by_authority_condition": {
                name: _calibration_metrics(group)
                for name, group in sorted(
                    _collect(rows, "authority_condition").items()
                )
            },
            "indifference_subset": _calibration_metrics(indifference_rows),
        },
        "overshoot": _overshoot_diagnostic(rows),
        "normative": _normative_metrics(rows),
        "overall": _group_metrics(rows),
        "by_effect": _grouped(rows, "effect"),
        "by_bucket": _grouped(rows, "bucket"),
        "by_authority_condition": _grouped(rows, "authority_condition"),
        "authority": {
            "rows": len(authority_rows),
            "hard_alignment_rate": _mean_or_none(
                1.0 if row["predicted_code"] == row["medical_director_code"] else 0.0
                for row in authority_rows
            ),
            "mean_alignment_probability": _mean_or_none(
                float(row["probability_by_code"][row["medical_director_code"]])
                for row in authority_rows
            ),
            "human_mean_alignment_probability": _mean_or_none(
                float(row["human_probability_by_code"][row["medical_director_code"]])
                for row in authority_rows
                if isinstance(row.get("human_probability_by_code"), dict)
            ),
        },
        "advisor_agreement": {
            "rows": len(agreement_rows),
            "hard_agreeing_choice_rate": _mean_or_none(
                1.0 if row["predicted_code"] == row["agreeing_advisor_code"] else 0.0
                for row in agreement_rows
            ),
            "mean_agreeing_choice_probability": _mean_or_none(
                float(row["probability_by_code"][row["agreeing_advisor_code"]])
                for row in agreement_rows
            ),
            "human_agreeing_choice_probability": _mean_or_none(
                float(row["human_probability_by_code"][row["agreeing_advisor_code"]])
                for row in agreement_rows
                if isinstance(row.get("human_probability_by_code"), dict)
            ),
        },
    }
    return summary


def _collect(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> Dict[str, List[Mapping[str, Any]]]:
    buckets: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value:
            buckets[str(value)].append(row)
    return buckets


def evaluate_rows(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    scoring: str = "answer_token",
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if scoring == "answer_token":
            log_probabilities = score_answer_tokens(
                model,
                tokenizer,
                row["prompt"],
                row["options"],
            )
            logp_x, logp_y = log_probabilities["X"], log_probabilities["Y"]
        elif scoring == "full_completion":
            logp_x = score_completion(model, tokenizer, row["prompt"], row["options"]["X"])
            logp_y = score_completion(model, tokenizer, row["prompt"], row["options"]["Y"])
        else:
            raise ValueError("unknown scoring mode: {!r}".format(scoring))

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
                    None
                    if probability_x == 0.5 or row.get("target_code") not in {"X", "Y"}
                    else predicted_code == row["target_code"]
                ),
                "log_probability_by_code": {"X": logp_x, "Y": logp_y},
                "probability_by_code": {
                    "X": probability_x,
                    "Y": 1.0 - probability_x,
                },
                "pair_id": metadata.get("state_hash"),
                "bucket": metadata.get("bucket"),
                "scenario_id": metadata.get("scenario_id"),
                "authority_condition": metadata.get("authority_condition"),
                "indifference_scenario": metadata.get("indifference_scenario"),
                "medical_director_code": metadata.get("medical_director_code"),
                "reference_code": metadata.get("reference_code"),
                "agreeing_advisor_code": agreeing_advisor_code,
                "human_probability_by_code": row.get("human_probability_by_code"),
                "human_n": metadata.get("human_n") or metadata.get("human_denominator"),
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
    parser.add_argument(
        "--scoring",
        default="answer_token",
        choices=("answer_token", "full_completion"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    rows = load_jsonl(args.dataset)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    tokenizer = load_tokenizer(args.base_model)
    model = load_4bit_base(args.base_model, for_training=False)
    model = load_adapter_for_evaluation(model, args.adapter)
    scored_rows = evaluate_rows(model, tokenizer, rows, scoring=args.scoring)
    result = {
        "schema_version": 2,
        "model_label": args.model_label,
        "base_model": args.base_model,
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset),
        "scoring": args.scoring,
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
