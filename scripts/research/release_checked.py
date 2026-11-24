"""Maintainer release of runnable, unreviewed ("checked") protocols for existing library papers.

    python -m scripts.research.release_checked --private <studies-dir> --map <map.json> [--only paper_<id> ...]

For each library paper, `map.json` names exactly one private hsenv study (`{paper_id: {"matches": [slug]}}`). The
private protocol goes through the pinned engine's distribution projection (hsenv.release.project, strict): any
participant-facing text whose source is not hosted becomes a placeholder with a text hash and a locator, which the
desktop app fills in on the researcher's computer. Arm-level `release_hold` reasons are licence holds on that text,
so they are dropped once the projection has held it. The projection is checked with `hsenv.checks --public`; the
revision keeps the paper's current program, sources and artifacts, drops the legacy `package/`, and adds
study.json (quality "checked", no fidelity claim), protocol.json and review/checks-latest.json. Nothing private is
copied: no review notes, source files or source text of the private study.

The new revision is validated by the same rules as CI (revisions.validate_assets) before anything is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from hsenv import distribution, release, schema_check

from scripts.research.revisions import load_revision, revision_id, safe_path, validate_assets

ROOT = Path(__file__).resolve().parents[2]
HOSTED_UNLOCATED = re.compile(r"^\d+ text\(s\) cited to a hosted source are not in the supplied hosted file")


class ProjectionBlocked(Exception):
    """The private protocol cannot be projected for release without changes to its data."""
COMMUNITY = ROOT / "community"
HSENV_RANGE = ">=0.8.1,<0.9"
DROP_PREFIXES = ("package/",)
REVIEW = {
    "decision": "accept",
    "reviewer": "maintainer release (checked, unreviewed)",
    "reason": ("Runnable protocol released from the maintainers' private study as an unreviewed (checked) "
               "release: engine checks pass; no independent fidelity review is claimed."),
    "reviewScope": "Mechanical engine checks of the exact released protocol only.",
}


def dumps(data) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode()


def project(study_dir: Path) -> tuple[dict, dict]:
    protocol = json.loads((study_dir / "protocol.json").read_text())
    sources = {}
    for key, src in ((protocol.get("meta") or {}).get("sources") or {}).items():
        name = isinstance(src, dict) and src.get("file")
        if name and (study_dir / name).is_file():
            sources[key] = (study_dir / name).read_bytes()
    # Strict except for one finding: a text cited to a HOSTED source but not found in that file ships either way
    # (hosted = openly licensed), typically a platform button label. Any other finding fails the paper.
    projected, report = release.project(protocol, sources=sources,
                                        extractors={"pdf": distribution.pdfium_extractor()}, strict=False)
    blocking = [e for e in report.get("errors") or [] if not HOSTED_UNLOCATED.search(e)]
    if blocking:
        raise ProjectionBlocked("; ".join(e[:160] for e in blocking[:4]))
    if projected.get("schema_version") is None and projected.get("schema") in (None, "hsenv-protocol/0.1"):
        projected["schema_version"] = "1"
    holds = {}
    for arm in projected.get("arms") or []:
        why = arm.pop("release_hold", None)
        if why:
            holds[arm["id"]] = why
    return projected, {"held": report.get("held", 0), "holdsResolved": holds}


def run_checks(protocol: dict, sessions: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "protocol.json").write_bytes(dumps(protocol))
        out = subprocess.run([sys.executable, "-m", "hsenv.checks", ".", "--public", "--sessions", str(sessions)],
                             cwd=tmp, capture_output=True, text=True, timeout=1800)
    start = out.stdout.find("{")
    if start < 0:
        raise RuntimeError(f"hsenv checks printed no report: {out.stderr[-400:]}")
    return json.loads(out.stdout[start:])


def licences(protocol: dict) -> dict:
    srcs = (protocol.get("meta") or {}).get("sources") or {}
    tiers = {k: (v.get("distribution") or {}).get("tier", "user") for k, v in srcs.items() if isinstance(v, dict)}
    paper = (srcs.get("paper") or {}) if isinstance(srcs.get("paper"), dict) else {}
    return {"paper": (paper.get("distribution") or {}).get("licence") or "see sources",
            "materials": "hosted" if tiers and set(tiers.values()) == {"hosted"} else "held text is restored locally"}


def build(paper_id: str, slug: str, private: Path, sessions: int) -> dict:
    paper_dir = COMMUNITY / paper_id
    current = json.loads((paper_dir / "current.json").read_text())
    base = load_revision(paper_dir, current["revision"])
    protocol, info = project(private / slug)
    report = run_checks(protocol, sessions)
    if report.get("PASS") is not True:
        return {"paper": paper_id, "slug": slug, "status": "checks_failed"}
    sha = schema_check.canonical_sha256(protocol)
    paper = current["paper"]
    study = {
        "schema_version": "1", "id": protocol["id"], "title": paper.get("title"), "authors": paper.get("authors", []),
        "year": paper.get("year"), "doi": paper.get("doi"), "version": "1.0.0", "quality": "checked",
        "quality_evidence": ["review/checks-latest.json"], "licence": licences(protocol),
        "protocol": {"path": "protocol.json", "sha256": sha, "schema_version": str(protocol.get("schema_version", "1"))},
        "hsenv": HSENV_RANGE, "paperId": paper_id, "visibility": "public",
        "release": {"kind": "checked", "heldTextItems": info["held"],
                    "armHoldsResolvedByProjection": sorted(info["holdsResolved"])},
        "contributors": current.get("contributors") or [{"github": "assassin808"}],
    }
    study = {k: v for k, v in study.items() if v is not None}
    files = {k: v for k, v in base.items() if not k.startswith(DROP_PREFIXES)}
    files["protocol.json"] = dumps(protocol)
    files["study.json"] = dumps(study)
    files["review/checks-latest.json"] = dumps(report)
    validation = validate_assets(files)
    if validation["status"] != "passed":
        return {"paper": paper_id, "slug": slug, "status": "invalid", "errors": validation["errors"][:8]}
    rid = revision_id(files)
    folder = paper_dir / "revisions" / rid
    if not folder.exists():
        staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=paper_dir / "revisions"))
        try:
            entries = []
            for name, raw in sorted(files.items()):
                target = staging / safe_path(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                entries.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
            (staging / "manifest.json").write_text(json.dumps({"schemaVersion": 1, "revision": rid, "files": entries},
                                                              indent=2) + "\n")
            (staging / "projection.json").write_text(json.dumps(validation["projection"], ensure_ascii=False,
                                                                indent=2) + "\n")
            os.replace(staging, folder)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    updated = {"schemaVersion": 1, "revision": rid, "previousRevision": current["revision"], "paper": paper,
               "contributors": current.get("contributors", []), "capabilities": validation["capabilities"],
               "review": {**REVIEW, "candidateRevision": rid},
               "validation": {k: v for k, v in validation.items() if k != "projection"}}
    (paper_dir / "current.json").write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n")
    return {"paper": paper_id, "slug": slug, "status": "released", "revision": rid, "held": info["held"],
            "holdsResolved": len(info["holdsResolved"]), "replayable": validation["capabilities"].get("replayable")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--private", type=Path, required=True, help="private hsenv studies directory")
    ap.add_argument("--map", type=Path, required=True, help="{paper_id: {matches: [slug]}}")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--sessions", type=int, default=1000)
    a = ap.parse_args(argv)
    mapping = json.loads(a.map.read_text())
    results = []
    for paper_id, entry in sorted(mapping.items()):
        if a.only and paper_id not in a.only:
            continue
        slugs = entry.get("matches") or []
        current = json.loads((COMMUNITY / paper_id / "current.json").read_text())
        if len(slugs) != 1:
            results.append({"paper": paper_id, "status": "no_private_study"})
        elif "protocol.json" in load_revision(COMMUNITY / paper_id, current["revision"]):
            results.append({"paper": paper_id, "status": "already_runnable"})
        else:
            try:
                results.append(build(paper_id, slugs[0], a.private, a.sessions))
            except ProjectionBlocked as exc:
                results.append({"paper": paper_id, "slug": slugs[0], "status": "needs_fix", "error": str(exc)[:600]})
            except Exception as exc:  # one paper's failure must not stop the batch
                results.append({"paper": paper_id, "slug": slugs[0], "status": "error", "error": f"{type(exc).__name__}: {exc}"[:400]})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
