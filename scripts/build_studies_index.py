#!/usr/bin/env python3
"""
Build studies_index.json and optionally package studies into zips.

Reads legacy index.json or the agent-generated study.json from each study.
Writes co_website/data/studies_index.json and optionally study zips.

Usage (from repo root):
  python scripts/build_studies_index.py              # index only
  python scripts/build_studies_index.py --zips       # index + zips
  python scripts/build_studies_index.py --study X    # single study zip
  python scripts/build_studies_index.py --index-only # index only (default)
"""

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIES_DIR = REPO_ROOT / "studies"
INDEX_OUTPUT = REPO_ROOT / "co_website" / "data" / "studies_index.json"
DEFAULT_ZIP_DIR = REPO_ROOT / "dist" / "study_zips"


def study_id_sort_key(s: str) -> tuple:
    m = re.match(r"study_(\d+)", s, re.I)
    if m:
        return (int(m.group(1)),)
    return (999999, s)


def _contributors(values) -> list[dict]:
    contributors = []
    for value in values or []:
        if isinstance(value, str) and value.strip():
            contributors.append({"name": value.strip()})
        elif isinstance(value, dict) and value.get("name"):
            item = {"name": value["name"]}
            if value.get("github"):
                github = str(value["github"]).strip()
                item["github"] = github if github.startswith("http") else f"https://github.com/{github.lstrip('@/')}"
            if value.get("institution"):
                item["institution"] = value["institution"]
            contributors.append(item)
    return contributors


def _authors(values) -> list[str]:
    authors = []
    for value in values or []:
        if isinstance(value, str) and value.strip():
            authors.append(value.strip())
        elif isinstance(value, dict):
            name = value.get("name") or value.get("full_name")
            if name:
                authors.append(str(name).strip())
    return authors


def read_study_entry(study_dir: Path) -> dict | None:
    legacy = study_dir / "index.json"
    candidates = [legacy] if legacy.exists() else sorted(study_dir.rglob("study.json"))
    if not candidates:
        print(f"Skip {study_dir.name}: no index.json or study.json", file=sys.stderr)
        return None
    path = candidates[0]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Skip {study_dir.name}: invalid {path.name}: {exc}", file=sys.stderr)
        return None

    paper = data.get("paper") if isinstance(data.get("paper"), dict) else {}
    title = data.get("title") or paper.get("title") or study_dir.name
    authors = _authors(data.get("authors") or paper.get("authors"))
    year = data.get("year") or paper.get("year") or paper.get("publication_year")
    description = data.get("description") or data.get("summary") or paper.get("abstract") or ""
    return {
        "study_id": study_dir.name,
        "title": str(title),
        "authors": authors,
        "year": year,
        "description": str(description).strip(),
        "contributors": _contributors(data.get("contributors")),
    }


def build_index() -> list[dict]:
    """Build catalog entries from every numbered study directory."""
    study_dirs = sorted(
        [d for d in STUDIES_DIR.iterdir() if d.is_dir() and d.name.startswith("study_")],
        key=lambda d: study_id_sort_key(d.name),
    )
    studies = []
    for study_dir in study_dirs:
        entry = read_study_entry(study_dir)
        if entry:
            studies.append(entry)
    return studies


def write_index(studies: list[dict]) -> None:
    INDEX_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_OUTPUT, "w", encoding="utf-8") as f:
        json.dump({"studies": studies}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(studies)} studies to {INDEX_OUTPUT}")


def build_zip(study_id: str, output_dir: Path) -> Path:
    """Package study directory as-is (no README/INTERFACE generation)."""
    study_root = STUDIES_DIR / study_id
    if not study_root.is_dir():
        raise FileNotFoundError(f"Study dir not found: {study_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"{study_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in study_root.rglob("*"):
            if f.is_file():
                arcname = Path(study_id) / f.relative_to(study_root)
                zf.write(f, arcname)
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Build studies index and optionally package zips")
    parser.add_argument("--index-only", action="store_true", help="Only build index (default)")
    parser.add_argument("--zips", action="store_true", help="Also build zip for each study")
    parser.add_argument("--study", type=str, help="Package only this study ID")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ZIP_DIR, help="Directory for zips")
    args = parser.parse_args()

    if not STUDIES_DIR.exists():
        print(f"Error: {STUDIES_DIR} not found", file=sys.stderr)
        return 1

    studies = build_index()
    write_index(studies)

    if args.zips or args.study:
        if args.study:
            study_dirs = [STUDIES_DIR / args.study]
            if not study_dirs[0].is_dir():
                print(f"Error: {study_dirs[0]} not found", file=sys.stderr)
                return 1
        else:
            study_dirs = [STUDIES_DIR / s["study_id"] for s in studies]
        for study_dir in study_dirs:
            study_id = study_dir.name
            try:
                zip_path = build_zip(study_id, args.output_dir)
                print(f"Created {zip_path}")
            except Exception as e:
                print(f"Skip {study_id}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
