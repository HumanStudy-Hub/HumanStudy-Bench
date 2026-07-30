"""
Patch Runner — orchestrates Stage 3 across the existing 30-paper corpus.

Pairs each JSON in <json-dir> with a PDF in <pdf-dir> by filename-stem prefix,
runs SlotFiller, and writes back atomically (.tmp → rename), with optional
.bak backup of the original.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generation_pipeline.patchers.slot_filler import SlotFiller
from generation_pipeline.verification.schema_validator import summarize_report, validate_paper


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def _stem_key(name: str, n_chars: int = 60) -> str:
    """Normalize a filename for matching (matches extractor.html's keyOf)."""
    s = Path(name).stem
    s = s.replace(",", "").replace(" ", "_").lower()
    return s[:n_chars]


def pair_json_pdf(json_dir: Path, pdf_dir: Path, *, only: Optional[List[str]] = None) -> List[Tuple[Path, Path]]:
    """Return [(json_path, pdf_path), ...] paired by filename-stem prefix."""
    json_dir = Path(json_dir)
    jsons = list(json_dir.glob("*.json"))
    jsons.extend(path for path in json_dir.glob("*/*.json") if path not in jsons)
    if only:
        jsons = [j for j in jsons if any(o in j.name for o in only)]
    pdfs = list(Path(pdf_dir).glob("*.pdf"))
    pdf_index = {_stem_key(p.name): p for p in pdfs}

    out: List[Tuple[Path, Path]] = []
    for j in jsons:
        jk = _stem_key(j.name)
        # Exact match first, then prefix match
        match = pdf_index.get(jk)
        if match is None:
            for pk, p in pdf_index.items():
                if pk.startswith(jk[:40]) or jk.startswith(pk[:40]):
                    match = p
                    break
        if match is not None:
            out.append((j, match))
        else:
            print(f"  ⚠ no PDF match for {j.name}")
    return out


def _source_dir_from_candidate(path: Path) -> Path | None:
    """Resolve a source path to a directory containing combined_sources.txt.

    Accept both the actual source directory (``.../sources``) and a paper
    directory containing ``sources/combined_sources.txt``.
    """
    path = Path(path)
    if path.is_dir() and (path / "combined_sources.txt").exists():
        return path
    nested = path / "sources"
    if nested.is_dir() and (nested / "combined_sources.txt").exists():
        return nested
    return None


def discover_source_dirs(
    json_path: Path,
    pdf_path: Path,
    *,
    explicit_source_dirs: Optional[List[Path]] = None,
    use_fetched_sources: bool = True,
) -> List[Path]:
    """Find OSF/source directories for a JSON/PDF pair."""
    out: List[Path] = []
    seen: set[str] = set()
    candidates: list[Path] = []
    if explicit_source_dirs:
        candidates.extend(Path(item) for item in explicit_source_dirs)
    if use_fetched_sources:
        candidates.extend(
            [
                Path(json_path).parent / "sources",
                Path(pdf_path).parent / "sources",
            ]
        )

    for candidate in candidates:
        resolved = _source_dir_from_candidate(candidate)
        if resolved is None:
            continue
        key = str(resolved.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def write_json_atomic(path: Path, data: dict, *, backup: bool) -> None:
    backup_path = path.with_suffix(path.suffix + ".bak")
    if backup and path.exists() and not backup_path.exists():
        shutil.copy2(path, backup_path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def format_patch_update(update: Any) -> str:
    """Render one incremental Stage 3 update for CLI logs."""
    if not isinstance(update, dict):
        return str(update)

    path = update.get("path") or ".".join(
        str(part) for part in (update.get("study"), update.get("slot")) if part
    )
    parts = [path or "unknown"]
    status = update.get("status")
    score = update.get("score")
    source = update.get("source")
    iv = update.get("iv")
    dv = update.get("dv")
    filled_nulls = update.get("filled_nulls")
    updated_fields = update.get("updated_fields")
    content_chars = update.get("content_chars")

    if status is not None:
        parts.append(f"status={status}")
    if score is not None:
        try:
            parts.append(f"score={float(score):.0f}")
        except (TypeError, ValueError):
            parts.append(f"score={score}")
    if iv:
        parts.append(f"iv={_short_log_value(iv)}")
    if dv:
        parts.append(f"dv={_short_log_value(dv)}")
    if source:
        parts.append(f"source={source}")
    if filled_nulls is not None:
        parts.append(f"filled_nulls={filled_nulls}")
    if updated_fields is not None:
        parts.append(f"updated_fields={updated_fields}")
    if content_chars is not None:
        parts.append(f"content_chars={content_chars}")
    return " ".join(parts)


def _short_log_value(value: Any, limit: int = 48) -> str:
    text = " ".join(str(value).split())
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return json.dumps(text, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_patch(
    json_dir: Path,
    pdf_dir: Path,
    *,
    filler: SlotFiller,
    only: Optional[List[str]] = None,
    overwrite_filled: bool = False,
    backup: bool = True,
    dry_run: bool = False,
    use_fetched_sources: bool = True,
    source_dirs: Optional[List[Path]] = None,
) -> Dict[str, str]:
    """
    Patch every JSON in `json_dir` whose stem matches a PDF in `pdf_dir`.

    Args:
        only: optional filename substrings — only patch matching JSONs
        overwrite_filled: re-run already-filled slots
        backup: write .bak before overwriting
        dry_run: pair + run LLM but don't write back
        use_fetched_sources: if True (default), automatically look for a
            ``sources/`` directory adjacent to each JSON and paired PDF, then
            pass any ``combined_sources.txt`` found there to the slot filler.
            This lets Stage 3 use OSF/supplementary materials fetched during
            Stage 1 source discovery or stored under ``stage4/<paper>/sources``.
        source_dirs: optional explicit OSF/source directories. Each item may be
            either a directory containing ``combined_sources.txt`` or a paper
            directory containing ``sources/combined_sources.txt``.

    Returns map of {json_filename: status_string}.
    """
    pairs = pair_json_pdf(json_dir, pdf_dir, only=only)

    print(f"Pairing: {len(pairs)} JSON↔PDF pairs found.")
    if not pairs:
        return {}

    results: Dict[str, str] = {}
    for i, (jpath, ppath) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] {jpath.name}")
        try:
            paper = json.loads(jpath.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ✗ could not load JSON: {e}")
            results[jpath.name] = f"load-error: {e}"
            continue

        pair_source_dirs = discover_source_dirs(
            jpath,
            ppath,
            explicit_source_dirs=source_dirs,
            use_fetched_sources=use_fetched_sources,
        )
        for source_dir in pair_source_dirs:
            print(f"  ↳ using fetched sources: {source_dir / 'combined_sources.txt'}")

        def write_incremental(current: dict, update: Any) -> None:
            if dry_run:
                return
            detail = format_patch_update(update)
            repaired, report = validate_paper(current, repair=True, path=jpath)
            if not report.valid:
                raise ValueError(f"Incremental schema validation failed after {detail}: {summarize_report(report)}")
            write_json_atomic(jpath, repaired, backup=backup)
            print(f"    ↳ wrote incremental patch: {detail}")

        try:
            patched = filler.patch_paper(
                paper,
                ppath,
                overwrite_filled=overwrite_filled,
                source_dirs=pair_source_dirs or None,
                on_update=write_incremental,
            )
        except Exception as e:
            print(f"  ✗ patch failed: {e}")
            results[jpath.name] = f"patch-error: {e}"
            continue

        patched, schema_report = validate_paper(patched, repair=True, path=jpath)
        if not schema_report.valid:
            print(f"  ✗ schema validation failed: {summarize_report(schema_report)}")
            results[jpath.name] = f"schema-error: {summarize_report(schema_report)}"
            continue
        if schema_report.changed or schema_report.issues:
            print(f"  schema validation: {summarize_report(schema_report)}")

        if dry_run:
            print(f"  ✓ (dry-run) would write {jpath.name}")
            results[jpath.name] = "dry-run-ok"
            continue

        write_json_atomic(jpath, patched, backup=backup)
        print(f"  ✓ wrote {jpath.name}")
        results[jpath.name] = "ok"

    return results
