"""
Fill missing materials/manipulation/items slots from downloaded OSF files.

Rule-based extraction (no LLM): maps OSF file text to JSON slots by study + DV.
"""

from __future__ import annotations

import json
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from generation_pipeline.utils.osf_crawler import (
    NEEDS_OSF_STATUSES,
    SLOTS,
    build_paper_plan,
    find_paper_jsons,
)
from generation_pipeline.utils.pdf_extractor import extract_pdf_text

# --- Pre-extracted scale blocks (from OSF Online Appendix / Qualtrics) ---

THEBENEFIT_ITEMS = {
    "moral rumination": """Rumination (Trapnell & Campbell, 1999; adapted for moral topics at work; Study 3, OSF Online Appendix):
• I ruminate about the morality of my behaviors.
• I spend a great amount of time thinking about whether my behavior at work is moral or not.
• I ruminate or dwell on whether I did the right or wrong thing at work.""",
    "cognitive flexibility": """Cognitive Flexibility (Martin, 1995; Study 3, OSF Online Appendix):
At work:
• I consider alternatives when dealing with work problems.
• I can find workable solutions to seemingly unsolvable problems.
• I have the self-confidence necessary to try different ways of behaving.""",
    "creativity": """Creativity (Zhou & George, 2001; Study 3, OSF Online Appendix):
Supervisor/rater items:
• Comes up with creative solutions to problems.
• Often has new and innovative ideas.
• Often has a fresh approach to problems.
(Study 3: creativity was based on trained raters' evaluations of participants' three park ideas.)""",
}

THEPROBABI_LONGTERMISM_ITEMS = """Longtermism belief items (Study 4; OSF Qualtrics .qsf):
• We should act wisely because what we do today will influence an untold number of people in the future.
• It is important to consider the long-term consequences of our actions and decisions.
• Intergenerational cooperation is important for addressing long-term challenges.
• It is important that we reduce existential and extinction risks to humanity and promote sustainable development goals to ensure the long-term survival of future generations.
• We should always have in view not only the present but also future generations.
• There are things we can do to steer the long-term future to a better course.
• Positively influencing the long-term future is a key moral priority of our time.
(Each item was rated for multiple future timeframes: 1,000 / 10,000 / 100,000 years in the future.)"""

THEPROBABI_REFORM_ITEMS = """Policy support measure (Study 4; OSF Qualtrics .qsf):
On a scale of 0–100, how much do you support reform for your country's legal system to protect the welfare (broadly understood as the rights, interests, and/or well-being) of the following groups?
• Humans living in the present
• Non-human animals
• Environment (e.g., rivers, trees or nature itself)"""


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    return "\n".join(t.text for t in root.iter() if t.tag.endswith("}t") and t.text)


def _load_osf_texts(osf_dir: Path) -> dict[str, str]:
    """Map relative path -> extracted text."""
    out: dict[str, str] = {}
    files_dir = osf_dir / "files"
    if not files_dir.exists():
        return out
    for path in files_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(files_dir))
        try:
            if path.suffix.lower() == ".pdf":
                out[rel] = extract_pdf_text(path, max_chars=80000)
            elif path.suffix.lower() == ".docx":
                out[rel] = _docx_text(path)
            elif path.suffix.lower() in (".txt", ".html", ".htm"):
                out[rel] = path.read_text(encoding="utf-8", errors="replace")[:80000]
            elif path.suffix.lower() == ".qsf":
                out[rel] = path.read_text(encoding="utf-8", errors="replace")[:200000]
        except Exception as e:
            out[rel] = f"[extract error: {e}]"
    return out


def _match_items_by_dv(dv: str, paper_folder: str) -> str | None:
    dv_l = (dv or "").lower()
    if paper_folder == "2023_TheBenefit":
        for key, block in THEBENEFIT_ITEMS.items():
            if key in dv_l:
                return block
    if paper_folder == "2026_TheProbabi":
        if "longtermism" in dv_l and "reform" not in dv_l and "support" not in dv_l:
            return THEPROBABI_LONGTERMISM_ITEMS
        if "reform" in dv_l or "support" in dv_l or "policy" in dv_l:
            return THEPROBABI_REFORM_ITEMS
    return None


def _normalize_docx_text(raw: str) -> str:
    """Collapse broken line breaks from DOCX extraction."""
    return re.sub(r"\s+", " ", raw).strip()


def _fill_leadersand(data: dict[str, Any], osf_texts: dict[str, str]) -> list[str]:
    """Upgrade Study 3 slots from Study 3 Manipulations.docx."""
    changes: list[str] = []
    manip_path = next((k for k in osf_texts if "manipulation" in k.lower()), None)
    if not manip_path:
        return ["No Study 3 Manipulations file found on OSF"]

    raw = _normalize_docx_text(osf_texts[manip_path])

    # Fast-food misconduct scenario (materials)
    scenario = None
    if "McDonald" in raw and "Charlie" in raw:
        start = raw.find("McDonald")
        end = raw.find("three times", start)
        if end > start:
            scenario = raw[start : end + len("three times")]

    # Full cognitive-load + typing instructions (manipulation)
    manip_block = raw if "Cognitive Load" in raw else None

    for study in data.get("eligible_studies", []):
        if "study 3" not in (study.get("study") or "").lower():
            continue
        for effect in study.get("effects", []):
            if scenario:
                effect["materials"] = {"status": "verbatim", "content": scenario}
                if "Study 3 materials" not in str(changes):
                    changes.append("Study 3 materials → verbatim from OSF")

            if manip_block:
                effect["manipulation"] = {"status": "verbatim", "content": manip_block}
                if "Study 3 manipulation" not in str(changes):
                    changes.append("Study 3 manipulation → verbatim from OSF")

            slot = effect.get("items", {})
            if slot.get("content") is None and slot.get("status") in NEEDS_OSF_STATUSES:
                if "items still missing" not in str(changes):
                    changes.append(
                        "Study 3 items still missing (punishment scales not in OSF download)"
                    )

    return changes


def fill_paper_json(json_path: Path, dry_run: bool = False) -> dict[str, Any]:
    plan = build_paper_plan(json_path)
    if not plan.osf_id:
        return {"paper": plan.paper_folder, "skipped": "no OSF link"}

    osf_dir = json_path.parent / "osf"
    if not (osf_dir / "files").exists():
        return {"paper": plan.paper_folder, "skipped": "no osf/files/ — run fetch_osf.py first"}

    data = json.loads(json_path.read_text(encoding="utf-8"))
    osf_texts = _load_osf_texts(osf_dir)
    changes: list[str] = []

    if plan.paper_folder == "2023_Leadersand":
        changes.extend(_fill_leadersand(data, osf_texts))

    for study in data.get("eligible_studies", []):
        for effect in study.get("effects", []):
            for slot_name in SLOTS:
                slot = effect.get(slot_name) or {}
                status = slot.get("status")
                content = slot.get("content")
                missing = content is None or (isinstance(content, str) and not content.strip())
                if status not in NEEDS_OSF_STATUSES | {"osf_only"}:
                    continue
                if not missing:
                    continue

                if slot_name == "items":
                    block = _match_items_by_dv(effect.get("DV", ""), plan.paper_folder)
                    if block:
                        effect["items"] = {
                            "status": "verbatim",
                            "content": block,
                        }
                        changes.append(
                            f"{study.get('study')} items filled for DV={effect.get('DV', '')[:40]}"
                        )

    if changes and not dry_run:
        bak = json_path.with_suffix(json_path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(json_path, bak)
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "paper": plan.paper_folder,
        "changes": changes,
        "dry_run": dry_run,
    }


def run_fill(stage4_dir: Path, only: list[str] | None = None, dry_run: bool = False) -> list[dict]:
    results = []
    for jp in find_paper_jsons(stage4_dir):
        plan = build_paper_plan(jp)
        if not plan.osf_id:
            continue
        if only and not any(s in jp.as_posix() for s in only):
            continue
        if not plan.effects_needing_osf:
            continue
        results.append(fill_paper_json(jp, dry_run=dry_run))
    return results
