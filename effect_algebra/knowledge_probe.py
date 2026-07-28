"""Gate 0 knowledge probe: does the model already know what these papers found?

The calibration framing rests on a claim that has to be tested, not assumed:
that behavioural divergence from human data is *not* explained by the model
never having seen the source material. If the model can state a paper's finding
and still fails to reproduce the behaviour in context, the gap is about
behaviour rather than knowledge. If it cannot state the finding, that claim is
unavailable and the framing has to change.

Two probes per finding, because they fail differently:

* **open** - a free-text question, scored by whether the reported quantity
  appears in the answer. Cheap and strict; a miss is not proof of ignorance.
* **forced choice** - the true reported value against a plausible decoy, scored
  from the answer-token logits. Immune to phrasing and refusal, and it yields a
  probability rather than a binary hit.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .modeling import DEFAULT_BASE_MODEL, load_4bit_base, load_adapter_for_evaluation, load_tokenizer


PROBE_SYSTEM_PROMPT = (
    "Answer questions about published experimental findings in psychology and "
    "economics. If you are unsure, give your best estimate rather than refusing."
)


class Probe(dict):
    pass


PROBES: Sequence[Mapping[str, Any]] = (
    {
        "probe_id": "A_citation",
        "effect": "A",
        "kind": "open",
        "question": (
            "What did Anderson and Holt (1997), 'Information Cascades in the "
            "Laboratory', American Economic Review, report? Name the phenomenon "
            "and the main quantitative result."
        ),
        "accept_patterns": [r"cascade", r"\b0?\.7\d|\b7[0-9]\s?%|41\s*(of|/)\s*56"],
    },
    {
        "probe_id": "A_cascade_rate",
        "effect": "A",
        "kind": "forced_choice",
        "question": (
            "In Anderson & Holt (1997), cascade behaviour was observed in what "
            "share of the periods where a cascade was possible?"
        ),
        "true_value": "about 73 percent (41 of 56 periods)",
        "decoy_value": "about 31 percent (17 of 56 periods)",
    },
    {
        "probe_id": "A_tie_rule",
        "effect": "A",
        "kind": "forced_choice",
        "question": (
            "In Anderson & Holt (1997), when the Bayesian posterior was exactly "
            "1/2 and the private draw conflicted with the previous decision, "
            "what did subjects mostly do?"
        ),
        "true_value": "followed their own private signal in 57 of 68 cases",
        "decoy_value": "followed the previous public decision in 57 of 68 cases",
    },
    {
        "probe_id": "B_citation",
        "effect": "B",
        "kind": "open",
        "question": (
            "What did Jaquiery and Yeung (2024), 'Preferences for Advisor "
            "Agreement and Accuracy', PLOS ONE, find about how people choose "
            "between an accurate advisor and an advisor who agrees with them?"
        ),
        "accept_patterns": [r"accur", r"agree", r"feedback"],
    },
    {
        "probe_id": "B_3c_pick_rate",
        "effect": "B",
        "kind": "forced_choice",
        "question": (
            "In Jaquiery & Yeung (2024) Experiment 3C, the Dates task with "
            "feedback, how often did participants pick the agreeing advisor "
            "over the accurate one?"
        ),
        "true_value": "rarely, about 17 percent of choices",
        "decoy_value": "usually, about 71 percent of choices",
    },
    {
        "probe_id": "B_feedback_contrast",
        "effect": "B",
        "kind": "forced_choice",
        "question": (
            "In Jaquiery & Yeung (2024), how did the availability of outcome "
            "feedback change the preference between an accurate and an agreeing "
            "advisor in the Dates task?"
        ),
        "true_value": (
            "without feedback the choice was near chance, with feedback "
            "participants strongly preferred the accurate advisor"
        ),
        "decoy_value": (
            "without feedback participants strongly preferred the accurate "
            "advisor, with feedback the choice was near chance"
        ),
    },
    {
        "probe_id": "C_citation",
        "effect": "C",
        "kind": "open",
        "question": (
            "What did Schoebel, Rieskamp, and Huber (2016), 'Social Influences "
            "in Sequential Decision Making', PLOS ONE, find about the influence "
            "of a medical director on diagnostic choices?"
        ),
        "accept_patterns": [r"authorit|director", r"influenc|weight|follow"],
    },
    {
        "probe_id": "C_director_alignment",
        "effect": "C",
        "kind": "forced_choice",
        "question": (
            "In Schoebel et al. (2016) Study 2, how often did participants' "
            "diagnoses align with the medical director's stated diagnosis?"
        ),
        "true_value": "about 75 percent of the time",
        "decoy_value": "about 45 percent of the time",
    },
    {
        "probe_id": "C_equal_accuracy",
        "effect": "C",
        "kind": "forced_choice",
        "question": (
            "In Schoebel et al. (2016), how did the stated diagnostic accuracy "
            "of the medical director compare with that of the assistant "
            "physician?"
        ),
        "true_value": "identical: both were correct in two of three cases",
        "decoy_value": "the director was more accurate: 0.80 against 0.67",
    },
)


def _generate(model: Any, tokenizer: Any, question: str, max_new_tokens: int) -> str:
    import torch

    messages = [
        {"role": "system", "content": PROBE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(next(model.parameters()).device)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(generated[0, input_ids.shape[-1] :], skip_special_tokens=True)


def _forced_choice_probability(
    model: Any,
    tokenizer: Any,
    question: str,
    true_value: str,
    decoy_value: str,
    *,
    true_code: str,
) -> float:
    """Probability assigned to the true statement, from the answer-token logits.

    Scored under both letter assignments and averaged. Without this the probe
    measures the model's preference for a letter rather than its knowledge: on
    the base model the letter effect is large enough to make every item whose
    truth sits on the disfavoured letter come out wrong.
    """

    from .evaluate_choices import score_answer_tokens
    from .datasets import preference_probability

    probabilities = []
    for assigned_true_code in ("X", "Y"):
        statements = {
            assigned_true_code: true_value,
            ("Y" if assigned_true_code == "X" else "X"): decoy_value,
        }
        prompt = [
            {"role": "system", "content": PROBE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "{}\n\n"
                    "DECISION=X means: {}\n"
                    "DECISION=Y means: {}\n\n"
                    "Output exactly DECISION=X or DECISION=Y."
                ).format(question, statements["X"], statements["Y"]),
            },
        ]
        log_probabilities = score_answer_tokens(
            model,
            tokenizer,
            prompt,
            {"X": "DECISION=X", "Y": "DECISION=Y"},
        )
        probability_x = preference_probability(
            log_probabilities["X"], log_probabilities["Y"]
        )
        probabilities.append(
            probability_x if assigned_true_code == "X" else 1.0 - probability_x
        )
    return sum(probabilities) / len(probabilities)


def run_probes(
    model: Any,
    tokenizer: Any,
    *,
    max_new_tokens: int = 220,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for index, probe in enumerate(PROBES):
        record: Dict[str, Any] = {
            "probe_id": probe["probe_id"],
            "effect": probe["effect"],
            "kind": probe["kind"],
            "question": probe["question"],
        }
        if probe["kind"] == "open":
            answer = _generate(model, tokenizer, probe["question"], max_new_tokens)
            matched = [
                pattern
                for pattern in probe["accept_patterns"]
                if re.search(pattern, answer, flags=re.IGNORECASE)
            ]
            record["answer"] = answer.strip()
            record["matched_patterns"] = matched
            record["recall_score"] = len(matched) / len(probe["accept_patterns"])
        else:
            # Both letter assignments are scored and averaged, so no letter can
            # masquerade as knowledge.
            record["scored_both_letter_assignments"] = True
            record["true_value"] = probe["true_value"]
            record["decoy_value"] = probe["decoy_value"]
            record["true_statement_probability"] = _forced_choice_probability(
                model,
                tokenizer,
                probe["question"],
                probe["true_value"],
                probe["decoy_value"],
                true_code="X",
            )
        results.append(record)
        print(json.dumps(record, ensure_ascii=False)[:400], flush=True)
    return results


def summarize(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_effect: Dict[str, Dict[str, Any]] = {}
    for effect in sorted({str(record["effect"]) for record in results}):
        group = [record for record in results if record["effect"] == effect]
        forced = [
            float(record["true_statement_probability"])
            for record in group
            if record["kind"] == "forced_choice"
        ]
        open_scores = [
            float(record["recall_score"])
            for record in group
            if record["kind"] == "open"
        ]
        by_effect[effect] = {
            "forced_choice_probes": len(forced),
            "mean_true_statement_probability": (
                sum(forced) / len(forced) if forced else None
            ),
            "forced_choice_accuracy": (
                sum(1.0 for value in forced if value > 0.5) / len(forced)
                if forced
                else None
            ),
            "open_probes": len(open_scores),
            "mean_open_recall": (
                sum(open_scores) / len(open_scores) if open_scores else None
            ),
        }
    forced_all = [
        float(record["true_statement_probability"])
        for record in results
        if record["kind"] == "forced_choice"
    ]
    return {
        "by_effect": by_effect,
        "overall_forced_choice_accuracy": (
            sum(1.0 for value in forced_all if value > 0.5) / len(forced_all)
            if forced_all
            else None
        ),
        "interpretation": (
            "High forced-choice accuracy with a large behavioural MAE supports "
            "the claim that the gap is behavioural rather than a knowledge gap. "
            "Low accuracy invalidates that claim and the framing must change."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    return parser


def main() -> None:
    args = _parser().parse_args()
    tokenizer = load_tokenizer(args.base_model)
    model = load_4bit_base(args.base_model, for_training=False)
    model = load_adapter_for_evaluation(model, args.adapter)
    results = run_probes(model, tokenizer, max_new_tokens=args.max_new_tokens)
    payload = {
        "schema_version": 1,
        "model_label": args.model_label,
        "base_model": args.base_model,
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "summary": summarize(results),
        "probes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
