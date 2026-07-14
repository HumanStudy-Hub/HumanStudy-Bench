"""Format generation-pipeline results as Markdown review files."""

import json
from typing import Any, Dict


class OutputFormatter:

    @staticmethod
    def format_stage1_review(filter_result: Dict[str, Any]) -> str:
        md = f"""# Stage 1: Study Inventory and Simulation Eligibility Review

## Paper Information
- **Title**: {filter_result.get('paper_title', 'N/A')}
- **Authors**: {', '.join(filter_result.get('paper_authors', []))}
- **Abstract**: {filter_result.get('paper_abstract', 'N/A')}

## Experiments Overview

"""
        quality = filter_result.get("stage1_quality") if isinstance(filter_result.get("stage1_quality"), dict) else {}
        if quality:
            md += f"""## Quality Audit
- **Experiments total**: {quality.get('experiments_total', 0)}
- **Eligible / uncertain**: {quality.get('eligible_or_uncertain', 0)}
- **Missing materials count**: {quality.get('missing_materials_count', 0)}
- **Study contract ready / needs review**: {quality.get('study_contract_ready', 'N/A')} / {quality.get('study_contract_needs_review', 'N/A')}
- **Study contract blocking issues**: {quality.get('study_contract_blocking_issues', 'N/A')}
- **Verifier status / overall**: {quality.get('verifier_status', 'N/A')} / {quality.get('verifier_overall', 'N/A')}
- **Verifier suggestions**: {quality.get('verifier_suggestions', 0)}
- **Needs human review**: {quality.get('needs_human_review', False)}

"""
            for issue in quality.get("issues", []):
                if isinstance(issue, dict):
                    md += f"- `{issue.get('severity', 'warning')}` {issue.get('path', '$')}: {issue.get('message', '')}\n"
            if quality.get("issues"):
                md += "\n"

        contract = filter_result.get("stage1_study_contract") if isinstance(filter_result.get("stage1_study_contract"), dict) else {}
        if contract:
            md += f"""## Stage 1 Study Contract
- **Ready studies**: {contract.get('ready', 0)}/{contract.get('total_studies', 0)}
- **Eligible / uncertain**: {contract.get('eligible_or_uncertain', 0)}
- **Needs review**: {contract.get('needs_review', 0)}
- **Missing by field**: `{json.dumps(contract.get('missing_by_field', {}), ensure_ascii=False)}`

"""

        refinement = filter_result.get("stage1_refinement") if isinstance(filter_result.get("stage1_refinement"), dict) else {}
        history = filter_result.get("stage1_refinement_history")
        history = history if isinstance(history, list) else []
        if refinement or history:
            md += f"""## Auto-Refinement
- **Attempts used**: {refinement.get('attempts', len(history))}
- **Max attempts**: {refinement.get('max_attempts', 'N/A')}
- **Final verifier overall**: {refinement.get('final_verifier_overall', 'N/A')}

"""
            for item in history:
                if not isinstance(item, dict):
                    continue
                regen = item.get("regeneration_instructions") if isinstance(item.get("regeneration_instructions"), dict) else {}
                md += (
                    f"- Attempt {item.get('attempt')}: trigger={item.get('trigger')} "
                    f"verifier={item.get('verifier_status')}/{item.get('verifier_overall')} "
                    f"feedback_items={sum(len(v) for v in regen.values() if isinstance(v, list))}\n"
                )
            md += "\n"

        verifier = filter_result.get("stage1_verification") if isinstance(filter_result.get("stage1_verification"), dict) else {}
        if verifier:
            md += f"""## Verifier Report
- **Status**: {verifier.get('status', 'N/A')}
- **Overall**: {verifier.get('overall', 'N/A')}
- **Confidence**: {verifier.get('confidence', 'N/A')}
- **Notes**: {verifier.get('notes', '')}

"""
            for check in verifier.get("inventory_checks", []):
                if isinstance(check, dict):
                    md += (
                        f"- Inventory `{check.get('study', 'N/A')}`: "
                        f"{check.get('verdict', 'needs_review')} - {check.get('issue', '')} "
                        f"{check.get('evidence', '')}\n"
                    )
            for check in verifier.get("eligibility_checks", []):
                if isinstance(check, dict):
                    md += (
                        f"- Eligibility `{check.get('study', 'N/A')}`: "
                        f"{check.get('verdict', 'needs_review')} "
                        f"expected={check.get('expected_label', 'N/A')} - {check.get('issue', '')} "
                        f"{check.get('evidence', '')}\n"
                    )
            if verifier.get("inventory_checks") or verifier.get("eligibility_checks"):
                md += "\n"

        for i, exp in enumerate(filter_result.get("experiments", []), 1):
            source_hints = exp.get("candidate_source_hints") if isinstance(exp.get("candidate_source_hints"), list) else []
            md += f"""### Experiment {i}: {exp.get('experiment_name', exp.get('experiment_id', 'Unknown'))}
- **Study ID**: {exp.get('study_id', 'N/A')}
- **Design type**: {exp.get('design_type', 'N/A')}
- **Conditions / factors**: {_format_stage1_conditions(exp.get('conditions_or_factors'))}
- **Input**: {exp.get('input', 'N/A')}
- **Participant task**: {exp.get('participant_task', 'N/A')}
- **Participants**: {exp.get('participants', 'N/A')}
- **Output**: {exp.get('output', 'N/A')}
- **Candidate source hints**: `{json.dumps(source_hints, ensure_ascii=False)}`
- **Replicable**: {exp.get('replicable', 'UNCERTAIN')}
- **Self-contained materials**: {exp.get('has_self_contained_materials', 'N/A')}
- **Missing materials**: {exp.get('missing_materials', '')}
- **Exclusion reasons**: {', '.join(exp.get('exclusion_reasons', [])) or 'None'}

#### Checklist:
- [ ] Human participant task is representable in HumanStudy-Bench
- [ ] Has quantitative statistic
- [ ] Sample / design recoverable
- [ ] Stimuli / scales accounted for (in-paper, OSF, or cited)

#### Comments:
[填写]

---

"""
        md += f"""## Overall Assessment
- **Any simulation candidate**: {'YES' if filter_result.get('overall_replicable', False) else 'NO'}
- **Confidence**: {filter_result.get('confidence', 0.0)}
- **Notes**: {filter_result.get('notes', 'N/A')}

## Review Status
- **Reviewed By**: [填写]
- **Review Status**: [pending/approved/needs_refinement]
- **Action**: [continue_to_stage2/refine_stage1/exclude]
"""
        return md

    @staticmethod
    def format_stage2_review(extraction: Dict[str, Any]) -> str:
        if not isinstance(extraction, dict):
            raise ValueError(f"extraction is not a dict: {type(extraction)}")

        md = f"""# Stage 2: Study and Finding Extraction Review

## Paper
- **Title**: {extraction.get('paper_title', 'N/A')}
- **Authors**: {', '.join((extraction.get('paper_metadata') or {}).get('authors', []))}
- **Year**: {(extraction.get('paper_metadata') or {}).get('year', 'N/A')}
- **Journal**: {(extraction.get('paper_metadata') or {}).get('journal', 'N/A')}
- **DOI**: {(extraction.get('paper_metadata') or {}).get('doi', 'N/A')}

"""
        quality = extraction.get("stage2_quality") if isinstance(extraction.get("stage2_quality"), dict) else {}
        if quality:
            coverage = quality.get("stage1_stage2_coverage") if isinstance(quality.get("stage1_stage2_coverage"), dict) else {}
            md += f"""## Quality Audit
- **Studies total**: {quality.get('studies_total', 0)}
- **Effects total**: {quality.get('effects_total', 0)}
- **Consolidated findings total**: {quality.get('findings_total', 0)}
- **Primary finding candidates**: {quality.get('primary_finding_candidates', 0)}
- **Effects with reported stats**: {quality.get('effects_with_reported_stats', 0)}
- **Legacy per-effect material slots observed**: {quality.get('legacy_effect_material_slot_hints', 0)}
- **Finding contract ready / needs review**: {quality.get('finding_contract_ready', 'N/A')} / {quality.get('finding_contract_needs_review', 'N/A')}
- **Finding contract blocking issues**: {quality.get('finding_contract_blocking_issues', 'N/A')}
- **Verifier status / overall**: {quality.get('verifier_status', 'N/A')} / {quality.get('verifier_overall', 'N/A')}
- **Verifier suggestions**: {quality.get('verifier_suggestions', 0)}
- **Stage1 eligible / uncertain**: {coverage.get('stage1_eligible_or_uncertain', 'N/A')}
- **Missing from Stage2**: {', '.join(coverage.get('missing_from_stage2', [])) or 'None'}
- **Extra in Stage2**: {', '.join(coverage.get('extra_in_stage2', [])) or 'None'}
- **Needs human review**: {quality.get('needs_human_review', False)}

"""
            for issue in quality.get("issues", []):
                if isinstance(issue, dict):
                    md += f"- `{issue.get('severity', 'warning')}` {issue.get('path', '$')}: {issue.get('message', '')}\n"
            if quality.get("issues"):
                md += "\n"
            legacy_details = [
                item
                for item in quality.get("legacy_effect_material_slot_details", []) or []
                if isinstance(item, dict)
            ]
            if legacy_details:
                md += "### Legacy Effect Material Slot Locations\n"
                for item in legacy_details[:12]:
                    md += (
                        f"- `{item.get('path', '$')}` "
                        f"{item.get('study', '')} effect={item.get('effect_index')} "
                        f"statuses=`{json.dumps(item.get('statuses', {}), ensure_ascii=False)}`\n"
                    )
                if len(legacy_details) > 12:
                    md += f"- ... {len(legacy_details) - 12} more in `stage2_quality.legacy_effect_material_slot_details`\n"
                md += "\n"

        contract = extraction.get("stage2_finding_contract") if isinstance(extraction.get("stage2_finding_contract"), dict) else {}
        if contract:
            md += f"""## Stage 2 Finding Contract
- **Ready studies**: {contract.get('ready', 0)}/{contract.get('total_studies', 0)}
- **Needs review**: {contract.get('needs_review', 0)}
- **Total findings**: {contract.get('total_findings', 0)}
- **Primary simulation targets**: {contract.get('primary_simulation_targets', 0)}
- **Missing by field**: `{json.dumps(contract.get('missing_by_field', {}), ensure_ascii=False)}`

"""

        refinement = extraction.get("stage2_refinement") if isinstance(extraction.get("stage2_refinement"), dict) else {}
        history = extraction.get("stage2_refinement_history")
        history = history if isinstance(history, list) else []
        if refinement or history:
            md += f"""## Auto-Refinement
- **Attempts used**: {refinement.get('attempts', len(history))}
- **Max attempts**: {refinement.get('max_attempts', 'N/A')}
- **Final verifier overall**: {refinement.get('final_verifier_overall', 'N/A')}

"""
            for item in history:
                if not isinstance(item, dict):
                    continue
                regen = item.get("regeneration_instructions") if isinstance(item.get("regeneration_instructions"), dict) else {}
                md += (
                    f"- Attempt {item.get('attempt')}: trigger={item.get('trigger')} "
                    f"verifier={item.get('verifier_status')}/{item.get('verifier_overall')} "
                    f"feedback_items={sum(len(v) for v in regen.values() if isinstance(v, list))}\n"
                )
            md += "\n"

        verification = extraction.get("stage2_verification") if isinstance(extraction.get("stage2_verification"), dict) else {}
        if verification:
            md += f"""## Verifier Report
- **Status**: {verification.get('status', 'N/A')}
- **Overall**: {verification.get('overall', 'N/A')}
- **Confidence**: {verification.get('confidence', 'N/A')}
- **Notes**: {verification.get('notes', '')}

"""
            for item in verification.get("study_coverage", []) or []:
                if isinstance(item, dict):
                    md += (
                        f"- Study `{item.get('study')}`: {item.get('verdict')} "
                        f"issue={item.get('issue', '')} evidence={item.get('evidence', '')}\n"
                    )
            for item in verification.get("finding_checks", []) or []:
                if isinstance(item, dict):
                    md += (
                        f"- Finding `{item.get('finding_id')}`: {item.get('verdict')} "
                        f"issue={item.get('issue', '')} evidence={item.get('evidence', '')}\n"
                    )
            md += "\n"

        for study in extraction.get("eligible_studies", []):
            sample = study.get("sample")
            sample_text = json.dumps(sample, ensure_ascii=False) if sample is not None else "N/A"
            md += f"""## {study.get('study', 'Unknown')}
- **Eligibility rationale**: {study.get('eligibility_rationale', '')}
- **Study sample**: `{sample_text}`

"""
            findings = [finding for finding in study.get("findings", []) or [] if isinstance(finding, dict)]
            if findings:
                md += "### Consolidated Findings\n"
                for finding in findings:
                    target = finding.get("simulation_target") if isinstance(finding.get("simulation_target"), dict) else {}
                    md += (
                        f"- **{finding.get('finding_id')}** "
                        f"role={finding.get('role')} target={target.get('candidate')} "
                        f"effect={finding.get('representative_effect_index')} "
                        f"IV={finding.get('IV')} -> DV={finding.get('DV')} "
                        f"stats={finding.get('reported_statistics')}\n"
                    )
                md += "\n"

            for j, eff in enumerate(study.get("effects", []), 1):
                stats = eff.get("stats", {}) or {}
                md += f"""### Effect {j}
- **Platform**: {eff.get('platform', 'N/A')}
- **IV → DV**: {eff.get('IV', '?')} → {eff.get('DV', '?')}
- **Effect type / direction**: {eff.get('effecttype', '?')} / {eff.get('direction', '?')}
- **N**: {eff.get('size', '?')}
- **Analysis N / scope**: {eff.get('analysis_n', None)} / {eff.get('analysis_scope', None)}
- **Stats**: B={stats.get('B')} t={stats.get('t')} F={stats.get('f')} eta²={stats.get('eta_square')} p={stats.get('p_value')} ({stats.get('sig')})
- **CI**: {stats.get('ci')}
- **Location**: {eff.get('table_or_page_location', 'N/A')}
- **Notes**: {eff.get('materials_notes', '')}

#### Checklist:
- [ ] Stats numbers match paper
- [ ] N matches paper
- [ ] Finding role / simulation-target mapping is correct
- [ ] Material search should use Stage3 study-level evidence, not this effect row

#### Comments:
[填写]

"""
            md += "---\n\n"

        md += """## Review Status
- **Reviewed By**: [填写]
- **Review Status**: [pending/approved/needs_refinement]
- **Action**: [accept/refine_stage2/back_to_stage1]
"""
        return md


def _format_stage1_conditions(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return "; ".join(parts) if parts else "N/A"
    text = str(value).strip()
    return text or "N/A"
