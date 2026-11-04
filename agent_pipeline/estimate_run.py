#!/usr/bin/env python3
"""Estimate the size and cost of replaying a package at the paper's sample size.

    python agent_pipeline/estimate_run.py package/<paper-slug> --n 692

``--n`` is the number of sessions the paper ran (participants for individual
designs, groups for group designs), summed over conditions. The adapter runs once
per condition with an offline stand-in model (no network, no paid calls); the
calls and prompt lengths it records are scaled to ``--n``. Prices are US dollars
per million tokens; the defaults are an example, so name the model and price you
assumed when you quote the result.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

try:
    from agent_pipeline.package_checks import run_probe
except ModuleNotFoundError:
    from package_checks import run_probe

OUTPUT_TOKENS_PER_CALL = 150  # a short answer; free-text screens run longer
CHARS_PER_TOKEN = 4


def estimate(probe: dict, n: int, price_in: float, price_out: float) -> dict:
    per_session_calls, prompt_chars = [], []
    for run in probe.get("runs") or []:
        calls = run.get("calls") or []
        if not calls:
            continue
        sessions = max(1, int(run.get("sessions") or 1))
        per_session_calls.append(len(calls) / sessions)
        prompt_chars.append(sum(len(call["prompt"]) for call in calls) / sessions)
    if not per_session_calls:
        raise SystemExit("The stand-in run produced no model calls; fix the adapter first.")
    calls = statistics.mean(per_session_calls) * n
    input_tokens = statistics.mean(prompt_chars) * n / CHARS_PER_TOKEN
    output_tokens = calls * OUTPUT_TOKENS_PER_CALL
    cost = input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out
    return {
        "sessions": n,
        "model_calls": round(calls),
        "calls_per_session": round(statistics.mean(per_session_calls), 1),
        "input_tokens": round(input_tokens),
        "output_tokens": round(output_tokens),
        "cost_usd": round(cost, 2),
        "price_per_million": {"input": price_in, "output": price_out},
        "truncated_conditions": [run.get("arm") for run in probe.get("runs") or [] if run.get("truncated")],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package", type=Path, help="the paper folder (package/<paper-slug>)")
    parser.add_argument("--n", type=int, required=True, help="sessions at the paper's sample size")
    parser.add_argument("--price-in", type=float, default=1.0, help="USD per million input tokens")
    parser.add_argument("--price-out", type=float, default=5.0, help="USD per million output tokens")
    args = parser.parse_args()
    root = args.package.resolve()
    if not (root / "task/adapter.py").is_file():
        candidates = [path for path in root.iterdir() if (path / "task/adapter.py").is_file()] if root.is_dir() else []
        if len(candidates) != 1:
            raise SystemExit("Give the paper folder that holds task/adapter.py.")
        root = candidates[0]
    probe = run_probe(root)
    if not probe:
        raise SystemExit("The stand-in run failed; run task/adapter.py --smoke-test to see why.")
    result = estimate(probe, args.n, args.price_in, args.price_out)
    print(json.dumps(result, indent=2))
    print(f"\nAt {result['sessions']:,} sessions: about {result['model_calls']:,} model calls "
          f"({result['calls_per_session']} per session), {result['input_tokens'] / 1e6:.1f} million input tokens, "
          f"about ${result['cost_usd']:,.2f} at ${args.price_in:g} / ${args.price_out:g} per million "
          "input / output tokens."
          + (f" Conditions cut short by the stand-in's call cap: {', '.join(map(str, result['truncated_conditions']))}"
             " (the real count is higher)." if result["truncated_conditions"] else ""))


if __name__ == "__main__":
    main()
