"""
Explain each finding for a study: hypothesis, PAS/ECS, key tests, and brief interpretation.
"""

from pathlib import Path
from typing import Any, Dict, List


def explain_finding(
    finding_id: str,
    finding_result: Dict[str, Any],
    test_results: List[Dict[str, Any]],
    metadata_finding: Dict[str, Any],
) -> str:
    """
    Produce a short explanation for one finding.

    Args:
        finding_id: e.g. "F1"
        finding_result: one entry from evaluation_results["finding_results"] (finding_score, n_tests, etc.)
        test_results: list of test result dicts for this finding (pas, pi_human, pi_agent, ecs_test, etc.)
        metadata_finding: one entry from metadata["findings"] (main_hypothesis, tests)

    Returns:
        Plain-language explanation.
    """
    hypothesis = (metadata_finding or {}).get("main_hypothesis", "")
    score = finding_result.get("finding_score")
    n_tests = finding_result.get("n_tests", 0)

    lines = []
    if hypothesis:
        lines.append(f"**Hypothesis:** {hypothesis[:300]}{'...' if len(hypothesis) > 300 else ''}")
    if score is not None:
        lines.append(f"**PAS (finding):** {score:.3f}")
    lines.append(f"**Tests:** {n_tests}")

    if test_results:
        # Summarize test-level alignment
        pas_vals = [t.get("pas") or t.get("score") for t in test_results if t.get("pas") is not None or t.get("score") is not None]
        ecs_vals = [t.get("ecs_test") or t.get("replication_consistency") for t in test_results if t.get("ecs_test") is not None or t.get("replication_consistency") is not None]
        if pas_vals:
            avg_pas = sum(pas_vals) / len(pas_vals)
            lines.append(f"**Avg test PAS (raw):** {avg_pas:.3f}")
        if ecs_vals:
            avg_ecs = sum(ecs_vals) / len(ecs_vals)
            lines.append(f"**Avg test ECS:** {avg_ecs:.3f}")
        # One-line interpretation
        if score is not None:
            if score >= 0.7:
                interp = "Agent replicates this finding well."
            elif score >= 0.5:
                interp = "Agent partially aligns with the human finding."
            else:
                interp = "Agent does not replicate this finding (low alignment or contradiction)."
            lines.append(f"**Interpretation:** {interp}")

    return "\n".join(lines)


def explain_study(
    study_id: str,
    evaluation_results: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build per-finding explanations for one study.

    Args:
        study_id: e.g. "study_001"
        evaluation_results: full dict from evaluation_results.json (finding_results, test_results)
        metadata: full dict from data/studies/{study_id}/metadata.json

    Returns:
        Dict with study_id and list of findings, each with finding_id, hypothesis, pas_score, explanation, key_tests.
    """
    finding_results = evaluation_results.get("finding_results") or []
    test_results = evaluation_results.get("test_results") or []
    meta_findings = {f["finding_id"]: f for f in (metadata.get("findings") or []) if f.get("finding_id")}

    findings_out = []
    for fr in finding_results:
        fid = fr.get("finding_id")
        if not fid:
            continue
        tests_for_finding = [t for t in test_results if t.get("finding_id") == fid]
        meta_f = meta_findings.get(fid, {})
        explanation = explain_finding(fid, fr, tests_for_finding, meta_f)
        pas_score = fr.get("finding_score")
        key_tests = []
        for t in tests_for_finding[:5]:
            key_tests.append({
                "test_name": t.get("test_name", ""),
                "pas_raw": t.get("pas") or t.get("score"),
                "ecs_test": t.get("ecs_test") or t.get("replication_consistency"),
            })
        findings_out.append({
            "finding_id": fid,
            "hypothesis": meta_f.get("main_hypothesis", ""),
            "pas_score": pas_score,
            "explanation": explanation,
            "key_tests": key_tests,
        })

    return {
        "study_id": study_id,
        "findings": findings_out,
    }


def run_finding_explanations(
    study_id: str,
    results_dir: Path,
    study_data_dir: Path,
    config_folder: str = None,
) -> Dict[str, Any]:
    """
    Load evaluation + metadata for a study (optionally a specific config folder) and return explanation payload.

    results_dir: e.g. results/benchmark
    study_data_dir: e.g. data/studies
    config_folder: optional, e.g. mistralai_mistral_nemo_v3_human_plus_demo. If None, uses first available folder under results_dir/study_{id}/.
    """
    study_path = results_dir / study_id
    if not study_path.exists():
        return {"study_id": study_id, "findings": [], "error": f"Results path not found: {study_path}"}

    if config_folder:
        config_path = study_path / config_folder
    else:
        configs = [d for d in study_path.iterdir() if d.is_dir()]
        if not configs:
            return {"study_id": study_id, "findings": [], "error": f"No config folders in {study_path}"}
        config_path = configs[0]

    eval_file = config_path / "evaluation_results.json"
    if not eval_file.exists():
        return {"study_id": study_id, "findings": [], "error": f"Missing {eval_file}"}

    meta_file = study_data_dir / study_id / "metadata.json"
    if not meta_file.exists():
        return {"study_id": study_id, "findings": [], "error": f"Missing {meta_file}"}

    import json
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return explain_study(study_id, eval_data, metadata)
