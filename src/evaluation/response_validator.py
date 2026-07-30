"""
Response Validation Utility

Checks if actual agent responses match expected counts and formats.
Run this after evaluation to diagnose parsing/extraction issues.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

def validate_responses(
    study_id: str,
    results_path: Path,
    specification_path: Path = None
) -> Dict[str, Any]:
    """
    Validate agent responses against expected counts and formats.

    Args:
        study_id: Study ID (e.g., 'study_004')
        results_path: Path to full_benchmark.json
        specification_path: Optional path to specification.json

    Returns:
        Validation report with expected vs actual counts
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Load specification for expected counts
    if specification_path is None:
        specification_path = Path(f"data/studies/{study_id}/specification.json")

    spec = {}
    if specification_path.exists():
        with open(specification_path, 'r') as f:
            spec = json.load(f)

    validation_report = {
        "study_id": study_id,
        "sub_studies": {},
        "summary": {}
    }

    # Count responses by sub_study_id
    actual_counts = defaultdict(int)
    response_details = defaultdict(list)

    for participant in results.get("individual_data", []):
        for resp in participant.get("responses", []):
            trial_info = resp.get("trial_info", {})
            sub_id = trial_info.get("sub_study_id")
            if sub_id:
                actual_counts[sub_id] += 1
                response_details[sub_id].append({
                    "response_text": resp.get("response_text", ""),
                    "items": trial_info.get("items", []),  # Items are in trial_info
                    "condition": trial_info.get("condition", {})
                })

    # Get expected counts from specification
    expected_counts = {}
    by_sub_study = spec.get("participants", {}).get("by_sub_study", {})
    for sub_id, sub_spec in by_sub_study.items():
        expected_counts[sub_id] = sub_spec.get("n", 0)

    # Validate each sub-study
    for sub_id in set(list(actual_counts.keys()) + list(expected_counts.keys())):
        expected = expected_counts.get(sub_id, 0)
        actual = actual_counts.get(sub_id, 0)

        # Check parsing completeness
        parsed_counts = check_parsing_completeness(
            response_details.get(sub_id, []),
            sub_id
        )

        validation_report["sub_studies"][sub_id] = {
            "expected_count": expected,
            "actual_count": actual,
            "missing": max(0, expected - actual),
            "extra": max(0, actual - expected),
            "match": expected == actual,
            "parsing": parsed_counts
        }

    # Summary
    total_expected = sum(expected_counts.values())
    total_actual = sum(actual_counts.values())
    validation_report["summary"] = {
        "total_expected": total_expected,
        "total_actual": total_actual,
        "total_missing": total_expected - total_actual,
        "match_rate": total_actual / total_expected if total_expected > 0 else 0
    }

    return validation_report


def check_parsing_completeness(
    responses: List[Dict[str, Any]],
    sub_study_id: str
) -> Dict[str, Any]:
    """
    Check if responses can be parsed correctly.

    Returns counts of:
    - Fully parsed (all expected fields present)
    - Partially parsed (some fields missing)
    - Unparseable (no fields extracted)
    """
    pattern = re.compile(r"(Q\d+(?:\.\d+)?)\s*=\s*([^,\n\s]+)")

    fully_parsed = 0
    partially_parsed = 0
    unparseable = 0
    parsing_issues = []

    for resp in responses:
        response_text = resp.get("response_text", "")
        items = resp.get("items", [])

        # Extract expected Q indices from items
        expected_q_indices = set()
        for i, item in enumerate(items):
            q_idx = item.get("q_idx")
            q_indices = item.get("q_indices", [])
            if q_idx:
                expected_q_indices.add(q_idx)
            if q_indices:
                expected_q_indices.update(q_indices)
            # If no q_idx is set, infer from item position (Q1, Q2, etc.)
            # This handles studies like study_003 where q_idx isn't stored in metadata
            if not q_idx and not q_indices:
                inferred_q = f"Q{i+1}"
                expected_q_indices.add(inferred_q)

        # Parse response
        parsed = {}
        for k, v in pattern.findall(response_text):
            clean_v = v.strip().rstrip('.,;)')
            if clean_v.endswith('%'):
                clean_v = clean_v[:-1]
            parsed[k.strip()] = clean_v

        # Check completeness
        if not expected_q_indices:
            # Can't validate if no expected indices
            continue

        found = set(parsed.keys()) & expected_q_indices
        missing = expected_q_indices - found

        if len(found) == len(expected_q_indices):
            fully_parsed += 1
        elif len(found) > 0:
            partially_parsed += 1
            if len(parsing_issues) < 10:  # Keep first 10 issues
                parsing_issues.append({
                    "missing_fields": list(missing),
                    "found_fields": list(found),
                    "response_preview": response_text[:100]
                })
        else:
            unparseable += 1
            if len(parsing_issues) < 10:
                parsing_issues.append({
                    "missing_fields": list(expected_q_indices),
                    "found_fields": [],
                    "response_preview": response_text[:100]
                })

    return {
        "fully_parsed": fully_parsed,
        "partially_parsed": partially_parsed,
        "unparseable": unparseable,
        "total": len(responses),
        "parsing_issues": parsing_issues
    }


def print_validation_report(report: Dict[str, Any]):
    """Print a human-readable validation report."""
    print("=" * 80)
    print(f"Response Validation Report: {report['study_id']}")
    print("=" * 80)

    print(f"\nSummary:")
    print(f"  Expected total: {report['summary']['total_expected']}")
    print(f"  Actual total: {report['summary']['total_actual']}")
    print(f"  Missing: {report['summary']['total_missing']}")
    print(f"  Match rate: {report['summary']['match_rate']:.1%}")

    print(f"\nSub-Study Details:")
    for sub_id, details in report['sub_studies'].items():
        status = "✓" if details['match'] else "✗"
        print(f"\n  {status} {sub_id}:")
        print(f"    Expected: {details['expected_count']}")
        print(f"    Actual: {details['actual_count']}")
        if details['missing'] > 0:
            print(f"    ⚠ Missing: {details['missing']} responses")
        if details['extra'] > 0:
            print(f"    ⚠ Extra: {details['extra']} responses")

        parsing = details['parsing']
        if parsing['total'] > 0:
            print(f"    Parsing: {parsing['fully_parsed']}/{parsing['total']} fully parsed")
            if parsing['partially_parsed'] > 0:
                print(f"      ⚠ {parsing['partially_parsed']} partially parsed")
            if parsing['unparseable'] > 0:
                print(f"      ⚠ {parsing['unparseable']} unparseable")
                if parsing['parsing_issues']:
                    print(f"      Sample issues:")
                    for issue in parsing['parsing_issues'][:3]:
                        print(f"        - Missing: {issue['missing_fields']}")
                        print(f"          Preview: {issue['response_preview']}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate agent responses")
    parser.add_argument("study_id", help="Study ID (e.g., study_004)")
    parser.add_argument("--results", help="Path to full_benchmark.json")
    parser.add_argument("--output", help="Output JSON file path")

    args = parser.parse_args()

    if args.results:
        results_path = Path(args.results)
    else:
        # Try to find in results directory
        results_path = Path(f"results/benchmark/{args.study_id}/mistralai_mistral_nemo_v3-human-plus-demo/full_benchmark.json")

    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        print(f"Please specify --results path or ensure results are in expected location")
        exit(1)

    report = validate_responses(args.study_id, results_path)
    print_validation_report(report)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output}")
