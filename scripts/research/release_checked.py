"""Maintainer release of runnable, unreviewed ("checked") protocols for existing library papers.

    python -m scripts.research.release_checked --private <studies-dir> --map <map.json> [--only paper_<id> ...]

For each library paper, `map.json` names exactly one private hsenv study (`{paper_id: {"matches": [slug]}}`). The
private protocol goes through the pinned engine's distribution projection (hsenv.release.project; strict except that a hosted
text missing from its hosted file ships, recorded as distribution.strict false): any
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

from scripts.research.protocols import validate_protocol_files
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


OPEN_LICENCE = re.compile(r"\bCC[- ]?0\b|public domain|\bCC[- ]?BY\b(?![- ]?(?:NC|ND|SA))", re.I)
SHA_PREFIX = re.compile(r"sha256[^0-9a-f]{0,6}([0-9a-f]{8,64})", re.I)
PATH = re.compile(r"((?:/Users/|sources/|\./)?[^\s'\"(),]+\.(?:pdf|qsf|docx|txt|html?|json|csv|md|rmd|r))", re.I)
LICENCE_HOLD = re.compile(r"licen[cs]|copyright|redistribut|publisher|paywall|permission", re.I)
# Anything besides the licence that a hold may name; such a hold is never cleared by the projection.
OTHER_HOLD = re.compile(r"missing|not printed|unprinted|pending|approv|decision|decide|unclear|incomplete|unresolved|"
                        r"todo|to do|not yet|reconstruct|fidelity|review|bug|broken|error", re.I)
TEXT_FORMATS = {"r", "rmd", "txt", "md", "csv", "tsv", "sps", "do", "py"}
URL = re.compile(r"https?://[^\s'\"()<>,]+")
CACHE = Path(os.environ.get("RELEASE_SOURCE_CACHE", tempfile.gettempdir())) / "hs-release-sources"


def open_licence(src) -> bool:
    """CC BY (not NC/ND/SA), CC0 or public domain by the source's structured licence id (distribution.licence), as
    the engine checks it for hosting. Free-text notes never count ("Not CC BY", "CC BY journal page" ...)."""
    dist = src.get("distribution") if isinstance(src, dict) else None
    return isinstance(dist, dict) and release.hosted_licence_ok(dist.get("licence"))


def _download(url: str, sha: str) -> bytes | None:
    """A public source file by its URL, kept only when it hashes to the declared sha256."""
    import urllib.request
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / sha
    if cached.is_file():
        return cached.read_bytes()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "humanstudy-release/1.0"})
        raw = urllib.request.urlopen(req, timeout=60).read()
    except Exception:
        return None
    if hashlib.sha256(raw).hexdigest() != sha:
        return None
    cached.write_bytes(raw)
    return raw


def url_serves(url: str, sha: str) -> bool:
    """This exact URL serves the file with this sha256 (a URL-specific check: the hash cache alone would accept any
    URL once the file is cached). Verified URLs are remembered per (URL, sha256)."""
    import urllib.request
    mark = CACHE / "urls" / hashlib.sha256(f"{url}\n{sha}".encode()).hexdigest()
    if mark.is_file():
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "humanstudy-release/1.0"})
        raw = urllib.request.urlopen(req, timeout=60).read()
    except Exception:
        return False
    if hashlib.sha256(raw).hexdigest() != sha:
        return False
    mark.parent.mkdir(parents=True, exist_ok=True)
    mark.write_text(url)
    (CACHE / sha).write_bytes(raw)
    return True


def resolve_sources(study_dir: Path, protocol: dict) -> dict[str, bytes]:
    """The private copy of every declared source the projection can use: the named file, a file in the study folder
    or at the path written in the declaration whose sha256 (or its printed prefix) matches, else a download of the
    declared URL that hashes to the declared full sha256. Nothing here is published; it only lets the projection
    locate held text and scan notes for quotes."""
    found: dict[str, bytes] = {}
    index: dict[str, Path] | None = None
    declared = (protocol.get("meta") or {}).get("sources") or {}
    # Where a declared relative path may live: the study folder, the build folder the protocol names as its root,
    # the stage-1 run folders of the same slug, and the workspace above them (paths written as "stage1/runs/...").
    workspace = study_dir.parents[3] if len(study_dir.parents) > 3 else study_dir
    bases = [study_dir]
    root = declared.get("root") if isinstance(declared.get("root"), str) else None
    if root:
        bases.append(Path(root) if root.startswith("/") else workspace / root)
    bases += [workspace / "stage1" / run / study_dir.name for run in ("runs", "runs-b2", "runs-b3")]
    bases.append(workspace)
    bases = [b for b in dict.fromkeys(bases) if b.is_dir()]

    def by_hash() -> dict[str, Path]:
        nonlocal index
        if index is None:
            index = {}
            for base in bases[:-1]:   # never hash the whole workspace
                for path in sorted(base.rglob("*")):
                    if path.is_file() and not path.is_symlink() and "runs" not in path.relative_to(base).parts:
                        index.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(), path)
        return index

    for key, src in declared.items():
        text = src if isinstance(src, str) else json.dumps(src, ensure_ascii=False) if isinstance(src, dict) else ""
        dist = (src.get("distribution") or {}) if isinstance(src, dict) else {}
        sha = (src.get("sha256") if isinstance(src, dict) else None) or dist.get("sha256")
        if not isinstance(sha, str):
            m = SHA_PREFIX.search(text)
            sha = m.group(1) if m else None
        sha = sha.lower() if isinstance(sha, str) else None
        paths = []
        if isinstance(src, dict) and isinstance(src.get("file"), str):
            paths.append(src["file"])
        paths += PATH.findall(text)
        raw = None
        for name in paths:
            for path in ([Path(name)] if name.startswith("/") else [b / name for b in bases]):
                if path.is_file():
                    candidate = path.read_bytes()
                    if not sha or hashlib.sha256(candidate).hexdigest().startswith(sha):
                        raw = candidate
                        break
            if raw is not None:
                break
        if raw is None and sha:
            hit = next((p for h, p in by_hash().items() if h.startswith(sha)), None)
            raw = hit.read_bytes() if hit else None
        if raw is None and sha and len(sha) == 64:
            for url in [dist.get("url"), src.get("url") if isinstance(src, dict) else None, *URL.findall(text)]:
                if isinstance(url, str) and url.startswith("http"):
                    raw = _download(url, sha)
                    if raw:
                        break
        if raw is not None and _usable(raw):
            found[key] = raw
    return found


def _usable(raw: bytes) -> bool:
    """Only formats the projection reads (pdf, docx, qsf, html, UTF-8 text): pictures, archives and other binaries
    are left out rather than failing the paper (their texts then count as unlocated, which blocks honestly)."""
    try:
        kind = distribution.sniff(raw)
    except Exception:
        return False
    if kind in ("pdf", "docx"):
        return True
    try:
        raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return False
    return True


def project(study_dir: Path) -> tuple[dict, dict]:
    protocol = json.loads((study_dir / "protocol.json").read_text())
    # Plain-text sources declared by their file type (an R script, R Markdown, a .txt readme) are format "text" to
    # the engine; the projection reads, locates and restores them as text.
    for src in ((protocol.get("meta") or {}).get("sources") or {}).values():
        dist = src.get("distribution") if isinstance(src, dict) else None
        if isinstance(dist, dict) and str(dist.get("format", "")).lower() in TEXT_FORMATS:
            dist["format"] = "text"
    # Hosting is only kept where the engine accepts the licence id and the credit is complete. A CC0 / public-domain
    # source without a written attribution gets one (title, link, licence; no credit is legally required). Anything
    # else that cannot be hosted as written is served as tier "link" instead (never published, restored locally).
    for key, src in ((protocol.get("meta") or {}).get("sources") or {}).items():
        dist = src.get("distribution") if isinstance(src, dict) else None
        if not isinstance(dist, dict) or dist.get("tier") != "hosted":
            continue
        lic = str(dist.get("licence") or "")
        if not release.hosted_licence_ok(lic):
            dist["tier"] = "link"
        elif not str(dist.get("attribution") or "").strip():
            if re.match(r"(?i)(cc0|public.?domain|pdm)", lic):
                dist["attribution"] = "%s. %s. %s." % (str(src.get("what") or key).split(";")[0][:200],
                                                     dist.get("url") or src.get("url") or "", lic)
            else:
                dist["tier"] = "link"
    sources = resolve_sources(study_dir, protocol)
    # Strict except for one finding: a text cited to a HOSTED source but not found in that file ships either way
    # (hosted = openly licensed), typically a platform button label. Any other finding fails the paper.
    projected, report = release.project(protocol, sources=sources,
                                        extractors={"pdf": distribution.pdfium_extractor()}, strict=False)
    blocking = [e for e in report.get("errors") or [] if not HOSTED_UNLOCATED.search(e)]
    if blocking:
        raise ProjectionBlocked("; ".join(e[:160] for e in blocking[:4]))
    # Held items that need a reading of a picture cannot be restored by the desktop helper (it restores text
    # by locators only), so such a release would not run on the researcher's computer.
    if report.get("requires_reading"):
        raise ProjectionBlocked(f"{len(report['requires_reading'])} held item(s) need an image reading the app cannot restore")
    # Notes, conflicts and other operator fields are not withheld by the projection; they are only safe when the
    # engine could scan them against every non-hosted source's text. Unscanned -> do not publish.
    msrc = (protocol.get("meta") or {}).get("sources") or {}
    closed = [k for k in report.get("quote_scan_unchecked") or [] if not open_licence(msrc.get(k))]
    if closed:
        raise ProjectionBlocked(f"source(s) {', '.join(closed[:6])} without an open licence could not be scanned for "
                                "quotes in notes; supply their text before release")
    tiers = report.get("tiers") or {}
    for key, raw in sources.items():
        if tiers.get(key) != "hosted" and raw[:5] == b"%PDF-":
            text = "".join(distribution.pdfium_extractor()(raw))
            if len(text.strip()) < 2000:   # a scan without a text layer: the quote scan saw nothing
                raise ProjectionBlocked(f"non-hosted PDF {key} has no text layer, so notes cannot be scanned for quotes")
    if not any(a.get("runnable", True) is not False for a in projected.get("arms") or []):
        raise ProjectionBlocked("no runnable arm: the paper does not print the materials every arm needs")
    if projected.get("schema_version") is None and projected.get("schema") in (None, "hsenv-protocol/0.1"):
        projected["schema_version"] = "1"
    # An arm hold is cleared only when it is a licence hold, which the projection resolves by holding the text;
    # any other hold (materials missing, a decision pending) keeps the arm out of the runnable set.
    holds = {}
    for arm in projected.get("arms") or []:
        why = arm.pop("release_hold", None)
        if not why:
            continue
        reason = json.dumps(why, ensure_ascii=False)
        if LICENCE_HOLD.search(reason) and not OTHER_HOLD.search(reason):
            holds[arm["id"]] = why
        else:
            arm["runnable"] = False
            arm.setdefault("why", why if isinstance(why, str) else json.dumps(why, ensure_ascii=False))
    if not any(a.get("runnable", True) is not False for a in projected.get("arms") or []):
        raise ProjectionBlocked("no runnable arm: the remaining arm holds are not licence holds")
    restorable(projected, protocol)
    return projected, {"held": report.get("held", 0), "holdsResolved": holds}


def restorable(projected: dict, private: dict) -> None:
    """Every held item must be restorable by the desktop helper: its source is tier "user" (the researcher's own
    file) or tier "link" with a URL that downloads to the declared sha256 (the helper fetches it and checks the hash).
    A held item against a hosted source is never fetched. A link URL that serves something else (an abstract page) is
    replaced by another URL declared for that source that does hash correctly, else the paper is blocked."""
    shipped = (projected.get("meta") or {}).get("sources") or {}
    declared = (private.get("meta") or {}).get("sources") or {}
    used = {e.get("source") for e in ((projected.get("distribution") or {}).get("held") or {}).values()}
    for key in sorted(k for k in used if k):
        dist = (shipped.get(key) or {}).get("distribution") or {}
        tier = dist.get("tier")
        if tier == "user":
            continue
        if tier != "link":
            raise ProjectionBlocked(f"held text against source {key} (tier {tier}) cannot be restored by the app")
        sha = str(dist.get("sha256") or "").lower()
        src = declared.get(key)
        text = json.dumps(src, ensure_ascii=False) if src is not None else ""
        urls = [dist.get("url"), src.get("url") if isinstance(src, dict) else None, *URL.findall(text)]
        ok = next((u for u in dict.fromkeys(urls) if isinstance(u, str) and u.startswith("http") and len(sha) == 64
                   and url_serves(u, sha)), None)
        if not ok:
            raise ProjectionBlocked(f"link source {key}: no declared URL downloads to its sha256, so the app cannot "
                                    "restore its held text")
        dist["url"] = ok


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
    if validation["status"] == "passed":
        # CI replays 300 sessions (validate-studies.yml); a release must pass exactly that, not only our 1,000-session run.
        replayed = validate_protocol_files(files, sessions=300)
        if replayed["status"] != "passed":
            validation = {**validation, "status": "failed", "errors": replayed["errors"]}
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
