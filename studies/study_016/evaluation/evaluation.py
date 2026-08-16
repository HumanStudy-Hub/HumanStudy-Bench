#!/usr/bin/env python3
"""
Evaluation pipeline for the CoMAP human-study package.

Implements the statistical procedure reported in the paper for Study 2
(main study, N=30 within-subjects crossover):
  1. Shapiro-Wilk normality test on each paired difference (CoMAP - Baseline).
  2. If normal: paired-samples t-test. If non-normal: Wilcoxon signed-rank test.
  3. ANCOVA on the outcome with Intelligent-TPACK total score as covariate.
  4. Cohen's d effect size (|d|>=0.2 small, >=0.5 medium, >=0.8 large).

Behavioral log-derived metrics (node-to-edge ratio, average inter-node
distance, chat turns, message length, negative-keyword count) are also
scored where the required inputs are present. Two of these metrics cannot
be computed faithfully without a researcher decision recorded in
audit/missing_information.json (canvas_coordinate_assignment,
negative_keyword_lexicon); for those, this module returns a structured
`not_ready` result naming the missing requirement instead of guessing.

This module performs no network access. All statistics use scipy/numpy,
which must be available in the execution environment; if they are not,
each check reports `not_ready` with reason "dependency_unavailable" rather
than crashing.
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    from scipy import stats as scipy_stats
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
AUDIT_JSON = PACKAGE_ROOT / "audit" / "missing_information.json"

EFFECT_SIZE_THRESHOLDS = {"small": 0.2, "medium": 0.5, "large": 0.8}


def _not_ready(check_name, reason, missing_field=None):
    result = {"check": check_name, "status": "not_ready", "reason": reason}
    if missing_field:
        result["missing_field"] = missing_field
        result["audit_ref"] = f"audit/missing_information.json entry '{missing_field}'"
    return result


def cohens_d_paired(diffs):
    diffs = np.asarray(diffs, dtype=float)
    sd = diffs.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diffs.mean() / sd)


def classify_effect_size(d):
    magnitude = abs(d)
    if magnitude >= EFFECT_SIZE_THRESHOLDS["large"]:
        return "large"
    if magnitude >= EFFECT_SIZE_THRESHOLDS["medium"]:
        return "medium"
    if magnitude >= EFFECT_SIZE_THRESHOLDS["small"]:
        return "small"
    return "negligible"


def paired_comparison(check_name, condition_a_scores, condition_b_scores, alpha=0.05):
    """
    condition_a_scores, condition_b_scores: equal-length lists of per-participant
    scores for the same outcome under CoMAP vs. Baseline (paired by participant).
    Implements Shapiro-Wilk -> paired t-test / Wilcoxon, per the paper's reported method.
    """
    if not _DEPS_AVAILABLE:
        return _not_ready(check_name, "dependency_unavailable: numpy/scipy not importable in this environment")

    a = np.asarray(condition_a_scores, dtype=float)
    b = np.asarray(condition_b_scores, dtype=float)
    if len(a) != len(b):
        return _not_ready(check_name, "input_shape_mismatch: condition_a_scores and condition_b_scores must be equal length and paired by participant")
    n = len(a)
    if n < 3:
        return _not_ready(check_name, f"insufficient_n: Shapiro-Wilk requires n>=3, got n={n}")

    diffs = a - b
    shapiro_stat, shapiro_p = scipy_stats.shapiro(diffs)
    is_normal = shapiro_p > alpha

    if is_normal:
        test_stat, test_p = scipy_stats.ttest_rel(a, b)
        test_name = "paired_t_test"
    else:
        try:
            test_stat, test_p = scipy_stats.wilcoxon(a, b)
        except ValueError as e:
            return _not_ready(check_name, f"wilcoxon_failed: {e}")
        test_name = "wilcoxon_signed_rank"

    d = cohens_d_paired(diffs)

    return {
        "check": check_name,
        "status": "ok",
        "n": n,
        "normality_test": {"test": "shapiro_wilk", "statistic": float(shapiro_stat), "p_value": float(shapiro_p), "normal_at_alpha_0.05": bool(is_normal)},
        "difference_test": {"test": test_name, "statistic": float(test_stat), "p_value": float(test_p)},
        "significant_at_alpha_0.05": bool(test_p < alpha),
        "effect_size": {"cohens_d": d, "magnitude": classify_effect_size(d)},
        "means": {"condition_a_mean": float(a.mean()), "condition_b_mean": float(b.mean())},
    }


def ancova_with_tpack_covariate(check_name, outcome_scores, condition_labels, tpack_scores):
    """
    outcome_scores: list of per-observation outcome scores (both conditions pooled)
    condition_labels: parallel list of "comap"/"baseline" labels
    tpack_scores: parallel list of each participant's Intelligent-TPACK total score
    Per the paper, TPACK is entered as a covariate to control for prior technology-integration competence.
    """
    if not _DEPS_AVAILABLE:
        return _not_ready(check_name, "dependency_unavailable: numpy/scipy not importable in this environment")

    y = np.asarray(outcome_scores, dtype=float)
    labels = np.asarray(condition_labels)
    covariate = np.asarray(tpack_scores, dtype=float)
    if not (len(y) == len(labels) == len(covariate)):
        return _not_ready(check_name, "input_shape_mismatch: outcome_scores, condition_labels, and tpack_scores must be equal length")
    if len(set(labels.tolist())) != 2:
        return _not_ready(check_name, "condition_labels must contain exactly two distinct condition values")

    group_indicator = np.where(labels == labels[0], 0.0, 1.0)
    design = np.column_stack([np.ones_like(y), group_indicator, covariate])
    try:
        beta, residuals, rank, singular_values = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError as e:
        return _not_ready(check_name, f"ancova_regression_failed: {e}")

    fitted = design @ beta
    resid = y - fitted
    n, p = design.shape
    if n - p <= 0:
        return _not_ready(check_name, f"insufficient_n: need more observations than model parameters (n={n}, params={p})")

    mse = float((resid ** 2).sum() / (n - p))
    xtx_inv = np.linalg.pinv(design.T @ design)
    se_group_coef = float(np.sqrt(mse * xtx_inv[1, 1]))
    group_coef = float(beta[1])
    t_stat = group_coef / se_group_coef if se_group_coef > 0 else float("nan")
    df = n - p
    p_value = float(2 * (1 - scipy_stats.t.cdf(abs(t_stat), df))) if se_group_coef > 0 else float("nan")

    return {
        "check": check_name,
        "status": "ok",
        "n": n,
        "model": "outcome ~ condition + intelligent_tpack_total",
        "condition_coefficient": group_coef,
        "condition_coefficient_se": se_group_coef,
        "t_statistic": t_stat,
        "df": df,
        "p_value": p_value,
        "significant_at_alpha_0.05": bool(p_value < 0.05) if p_value == p_value else None,
    }


# ---------------------------------------------------------------------------
# Behavioral log-derived metrics
# ---------------------------------------------------------------------------

def metric_node_to_edge_ratio(graph_log):
    """graph_log: {'nodes': [...], 'edges': [...]} from a CoMAP-condition session."""
    num_nodes = len(graph_log.get("nodes", []))
    num_edges = len(graph_log.get("edges", []))
    if num_edges == 0:
        return {"check": "node_to_edge_ratio", "status": "ok", "num_nodes": num_nodes, "num_edges": num_edges, "ratio": None, "note": "undefined (zero edges)"}
    return {"check": "node_to_edge_ratio", "status": "ok", "num_nodes": num_nodes, "num_edges": num_edges, "ratio": num_nodes / num_edges}


def metric_avg_inter_node_distance(graph_log):
    """
    Requires each node's canvas (x, y) coordinates in creation order. The paper
    reports this metric but never states how coordinates are assigned in a
    replicated (non-GUI) run -- see audit 'canvas_coordinate_assignment'.
    """
    nodes = graph_log.get("nodes", [])
    if not nodes or "x" not in nodes[0] or "y" not in nodes[0]:
        return _not_ready("avg_inter_node_distance",
                           "no canvas (x, y) coordinates present in the supplied graph_log; "
                           "this replication has no faithful way to assign canvas position for "
                           "programmatically-created nodes",
                           missing_field="canvas_coordinate_assignment")
    if not _DEPS_AVAILABLE:
        return _not_ready("avg_inter_node_distance", "dependency_unavailable: numpy not importable in this environment")

    coords = np.array([[n["x"], n["y"]] for n in nodes], dtype=float)
    if len(coords) < 2:
        return {"check": "avg_inter_node_distance", "status": "ok", "avg_distance": None, "note": "fewer than 2 nodes"}
    deltas = coords[1:] - coords[:-1]
    dists = np.sqrt((deltas ** 2).sum(axis=1))
    return {"check": "avg_inter_node_distance", "status": "ok", "avg_distance": float(dists.mean()), "num_consecutive_pairs": len(dists)}


def metric_chat_turns_and_length(chat_log):
    """chat_log: list of {'role': 'user'|'assistant', 'content': str}"""
    user_messages = [m["content"] for m in chat_log if m.get("role") == "user"]
    total_turns = len(chat_log)
    avg_len = (sum(len(m) for m in user_messages) / len(user_messages)) if user_messages else None
    return {"check": "chat_turns_and_length", "status": "ok", "total_chat_turns": total_turns, "num_user_messages": len(user_messages), "avg_user_message_length_chars": avg_len}


# Section 5.3 gives these three words as illustrative examples ('e.g.') of the
# 'predefined lexicon' it uses for this metric -- it explicitly does NOT claim
# this is the full lexicon. Kept here verbatim for reference only; three words
# are not sufficient to reproduce the paper's reported counts (see audit
# 'negative_keyword_lexicon'), so the default lexicon=None path below still
# reports not_ready rather than silently scoring against these three alone.
PAPER_REPORTED_EXAMPLE_KEYWORDS_VERBATIM = ["stuck", "confused", "can't"]


def metric_negative_keyword_count(chat_log, lexicon=None):
    """
    The paper reports a negative-keyword count based on 'a predefined lexicon'
    and gives three words as examples ('e.g., "stuck," "confused," "can't"')
    but never prints the full lexicon -- see audit 'negative_keyword_lexicon'.
    Callers may pass their own researcher-supplied lexicon to unblock this
    check; without one it is not_ready (three examples are not the lexicon).
    """
    if lexicon is None:
        return _not_ready("negative_keyword_count",
                           "the paper names three example keywords ('stuck', 'confused', "
                           "'can\\'t' -- see PAPER_REPORTED_EXAMPLE_KEYWORDS_VERBATIM) but "
                           "states these are illustrative of a larger 'predefined lexicon' "
                           "it never prints in full; three words are insufficient to "
                           "reproduce the paper's reported counts",
                           missing_field="negative_keyword_lexicon")
    user_text = " ".join(m["content"].lower() for m in chat_log if m.get("role") == "user")
    counts = {kw: user_text.count(kw.lower()) for kw in lexicon}
    return {"check": "negative_keyword_count", "status": "ok", "lexicon_source": "researcher_supplied_override", "counts": counts, "total": sum(counts.values())}


# ---------------------------------------------------------------------------
# CLI / smoke test
# ---------------------------------------------------------------------------

CHECK_REGISTRY = {
    "perceived_understanding": "paired_comparison",
    "perceived_expression": "paired_comparison",
    "human_ai_interaction_experience": "paired_comparison",
    "ancova_tpack_covariate": "ancova",
    "node_to_edge_ratio": "behavioral",
    "avg_inter_node_distance": "behavioral",
    "chat_turns_and_length": "behavioral",
    "negative_keyword_count": "behavioral",
}


def run_from_data_file(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for outcome in ("perceived_understanding", "perceived_expression", "human_ai_interaction_experience"):
        if outcome in data.get("paired_outcomes", {}):
            entry = data["paired_outcomes"][outcome]
            results.append(paired_comparison(outcome, entry["comap"], entry["baseline"]))
        else:
            results.append(_not_ready(outcome, f"input data file has no 'paired_outcomes.{outcome}' entry"))

    if "ancova_input" in data:
        anc = data["ancova_input"]
        results.append(ancova_with_tpack_covariate("ancova_tpack_covariate", anc["outcome_scores"], anc["condition_labels"], anc["tpack_scores"]))
    else:
        results.append(_not_ready("ancova_tpack_covariate", "input data file has no 'ancova_input' entry"))

    for graph_log in data.get("graph_logs", []):
        results.append(metric_node_to_edge_ratio(graph_log))
        results.append(metric_avg_inter_node_distance(graph_log))

    for chat_log in data.get("chat_logs", []):
        results.append(metric_chat_turns_and_length(chat_log))
        results.append(metric_negative_keyword_count(chat_log, lexicon=data.get("negative_keyword_lexicon_override")))

    return results


def _run_smoke_test():
    print("[1/3] Running paired_comparison on synthetic (non-paper) placeholder data...")
    result = paired_comparison("smoke_test_outcome", [5, 6, 6, 7, 5, 6, 7], [4, 5, 5, 6, 4, 5, 6])
    print(f"      status={result['status']}" + (f" test={result['difference_test']['test']}" if result["status"] == "ok" else ""))
    assert result["status"] in ("ok", "not_ready")

    print("[2/3] Running ancova_with_tpack_covariate on synthetic (non-paper) placeholder data...")
    anc_result = ancova_with_tpack_covariate(
        "smoke_test_ancova",
        outcome_scores=[5, 6, 4, 7, 5, 6, 4, 7, 6, 5],
        condition_labels=["comap"] * 5 + ["baseline"] * 5,
        tpack_scores=[3.2, 3.5, 2.8, 4.0, 3.1, 3.3, 2.9, 3.9, 3.4, 3.0],
    )
    print(f"      status={anc_result['status']}")
    assert anc_result["status"] in ("ok", "not_ready")

    print("[3/3] Verifying behavioral metrics correctly report not_ready when required inputs are absent...")
    no_coords_result = metric_avg_inter_node_distance({"nodes": [{"id": "1"}, {"id": "2"}]})
    assert no_coords_result["status"] == "not_ready"
    assert no_coords_result["missing_field"] == "canvas_coordinate_assignment"

    no_lexicon_result = metric_negative_keyword_count([{"role": "user", "content": "this is frustrating"}])
    assert no_lexicon_result["status"] == "not_ready"
    assert no_lexicon_result["missing_field"] == "negative_keyword_lexicon"

    ratio_result = metric_node_to_edge_ratio({"nodes": [{"id": "1"}, {"id": "2"}, {"id": "3"}], "edges": [{"from": "1", "to": "2"}]})
    assert ratio_result["status"] == "ok" and ratio_result["ratio"] == 3.0

    print("      ok -- both gap-dependent metrics correctly refuse to guess; ungated metric computed correctly.")
    print("\nSMOKE TEST PASSED. No network access was used. All numeric inputs above are synthetic placeholders, not paper data.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="CoMAP evaluation pipeline")
    parser.add_argument("--smoke-test", action="store_true", help="run an offline self-test with synthetic data")
    parser.add_argument("--data-file", type=str, help="path to a JSON file with paired_outcomes/ancova_input/graph_logs/chat_logs produced by task/adapter.py runs")
    args = parser.parse_args()

    if args.smoke_test:
        sys.exit(_run_smoke_test())

    if not args.data_file:
        print("ERROR: --data-file is required for a real run (or use --smoke-test).", file=sys.stderr)
        sys.exit(2)

    results = run_from_data_file(args.data_file)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
