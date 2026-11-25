"""Protocol releases in the community store (V080-PLATFORM §1.2, §2.1; B2/B3).

A released v0.8 study arrives in `community/paper_<id>/revisions/<sha256>/` holding study.json and
protocol.json. Everything here reads those files as data and calls the pinned hsenv engine; no file of a
revision is ever imported or executed.

    python -m scripts.research.protocols <revision-dir> [...] [--sessions 300]    exit 1 on any error

Checks, in the design's order:
1. schema: hsenv.schema_check on study.json and protocol.json (schema_version required, no legacy files).
2. release: only allowlisted public paths; visibility public; quality high_fidelity with a signed review
   (G2); no release_hold anywhere; study.json.protocol.sha256 equals the canonical protocol hash; arms in
   study.json agree with protocol.json; image sources are URLs or open assets.
3. review/checks-latest.json reports PASS for this exact protocol sha256 (a stale report fails); with
   --sessions N the engine also replays N random sessions here (refs + fuzz; no model calls).
4. effects.json through hsenv.effects_core; capabilities.json through the capability rules.
5. a revision directory's name is its manifest digest (revisions.revision_id).
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import sys
from pathlib import Path

from hsenv import schema_check

# One list, defined by the engine (hsenv.release.PUBLIC_PATHS, H1). Until hsenv ships that module the
# design's table (§1.2) is spelled out here; test_protocol_release.py fails once both exist and differ.
_DESIGN_PUBLIC_PATHS = (
    "study.json", "protocol.json", "effects.json", "program.json", "capabilities.json",
    "review/fidelity-review-*.md", "review/checks-latest.json",
    "sources/SOURCE.json", "sources/open/**", "sourcetext/*", "variants/*.json", "artifacts/**", "package/**",
)
try:  # pragma: no cover - depends on the hsenv release
    from hsenv.release import PUBLIC_PATHS  # type: ignore
except ImportError:
    PUBLIC_PATHS = _DESIGN_PUBLIC_PATHS

# Paths that are never public, whatever the allowlist says (§1.2 "Never included").
NEVER = ("sources/researcher/**", "sources/private/**", "sources/quarantine/**", "runs/**", "paper/**",
         "bundle.json", "**/runs/**", "**/records.jsonl")

# study.json fields the design adds (§1.1) that the frozen hsenv study.v1 schema does not yet declare
# (additionalProperties: false). Bench checks them itself and hands the rest to schema_check. H1 adds them
# to the engine's schema; then this list goes.
RELEASE_FIELDS = ("slug", "paper", "contributors", "visibility", "hsenv", "fidelity", "arms", "paperId", "release")
SIGNED_VERDICTS = {"HIGH FIDELITY", "high_fidelity"}
# "checked": a runnable release whose fidelity nobody has signed (owner 2026-11-24: every library paper runnable,
# labelled unreviewed). Only catalog/publications.json makes a paper Validated.
UNREVIEWED = "checked"
ASSET = re.compile(r"^asset:([0-9a-f]{16})$")


def _match(name: str, pattern: str) -> bool:
    """`*` stays inside one path segment; `dir/**` is everything below dir; `**/x` is x at any depth."""
    if pattern.startswith("**/"):
        parts = name.split("/")
        return any(_match("/".join(parts[i:]), pattern[3:]) for i in range(len(parts)))
    if pattern.endswith("/**"):
        return name.startswith(pattern[:-2])
    return name.count("/") == pattern.count("/") and fnmatch.fnmatchcase(name, pattern)


def path_problem(name: str, sources: dict) -> str | None:
    """Why `name` may not be published, or None. `sources` is protocol meta.sources."""
    from scripts.research.revisions import safe_path

    try:
        safe_path(name)
    except ValueError as exc:
        return f"{name}: {exc}"
    if any(_match(name, p) for p in NEVER):
        return f"{name}: never public (private source, run data or paper)"
    if not any(_match(name, p) for p in PUBLIC_PATHS):
        return f"{name}: not on the public release allowlist"
    if name.startswith("sourcetext/"):
        key = re.split(r"[#.]", name.split("/", 1)[1], maxsplit=1)[0]
        src = sources.get(key)
        if isinstance(src, dict):
            distribution = src.get("distribution")
            if not isinstance(distribution, dict) or distribution.get("tier") != "hosted":
                return f"{name}: structured source text requires a hosted source"
            src = src.get("file")
        if not (isinstance(src, str) and src.startswith("sources/open/")):
            return f"{name}: source text of a source that is not under sources/open"
        try:
            safe_path(src)
        except ValueError as exc:
            return f"{name}: invalid open source path ({exc})"
    return None


def _json(files: dict[str, bytes], name: str, errors: list[str]):
    if name not in files:
        return None
    try:
        return json.loads(files[name])
    except ValueError:
        errors.append(f"{name}: invalid JSON")
        return None


def _schema(data, kind: str, name: str, errors: list[str]) -> None:
    if not isinstance(data, dict):
        errors.append(f"{name}: must be an object")
        return
    if data.get("schema_version") is None:
        errors.append(f"{name}: schema_version is required (legacy files are not released)")
    result = schema_check.validate(data, kind)
    errors.extend(f"{name}: {path or '/'}: {msg}" for path, msg in result["errors"])


def _version_ok(spec: str, version: str) -> bool:
    def key(v):
        return tuple(int(x) for x in re.findall(r"\d+", v)[:3]) + (0,) * (3 - len(re.findall(r"\d+", v)[:3]))
    ops = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
           "<": lambda a, b: a < b, "==": lambda a, b: a == b}
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        m = re.fullmatch(r"(>=|<=|==|>|<)\s*([\d.]+)", part)
        if not m or not ops[m.group(1)](key(version), key(m.group(2))):
            return False
    return True


def engine_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("hsenv")
    except Exception:
        return None


def protocol_arms(protocol: dict) -> list[dict]:
    """The `arms` summary hsenv study sync writes into study.json (id, runnable, release_hold)."""
    out = []
    for arm in protocol.get("arms") or []:
        item = {"id": arm.get("id"), "runnable": bool(arm.get("runnable", True))}
        if arm.get("release_hold"):
            item["release_hold"] = arm["release_hold"]
        out.append(item)
    return out


def _image_sources(value, path=""):
    if isinstance(value, dict):
        for k, v in value.items():
            if k == "src" and isinstance(v, str):
                yield path + "/src", v
            else:
                yield from _image_sources(v, f"{path}/{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _image_sources(v, f"{path}/{i}")


def validate_protocol_files(files: dict[str, bytes], *, sessions: int = 0, seed: int = 1, draft: bool = False) -> dict:
    """Check data-only structure; draft intake omits publication approval and execution gates."""
    errors: list[str] = []
    warnings: list[str] = []
    study = _json(files, "study.json", errors)
    protocol = _json(files, "protocol.json", errors)
    if study is None or protocol is None:
        if "study.json" not in files or "protocol.json" not in files:
            errors.append("a protocol release needs both study.json and protocol.json")
        return {"status": "failed", "errors": errors, "warnings": warnings}

    # 1. schema
    if isinstance(study, dict):
        _schema({k: v for k, v in study.items() if k not in RELEASE_FIELDS}, "study", "study.json", errors)
    _schema(protocol, "protocol", "protocol.json", errors)
    if errors or not isinstance(study, dict) or not isinstance(protocol, dict):
        return {"status": "failed", "errors": errors, "warnings": warnings}

    # 2. release rules
    sources = (protocol.get("meta") or {}).get("sources") or {}
    errors.extend(p for p in (path_problem(n, sources) for n in sorted(files)) if p)
    if study.get("id") != protocol.get("id") or study.get("slug", study.get("id")) != study.get("id"):
        errors.append("study.json: id (and slug) must equal protocol.json id")
    if not draft and study.get("visibility") != "public":
        errors.append('study.json: visibility must be "public" for a release')
    fidelity = study.get("fidelity") or {}
    if not isinstance(fidelity, dict):
        errors.append("study.json: fidelity must be an object")
        fidelity = {}
    quality = study.get("quality")
    if not draft and quality == UNREVIEWED:
        # Runnable but not independently reviewed: the engine checks below must still PASS, and the release
        # may not claim any fidelity verdict or signature.
        if fidelity:
            errors.append("study.json: a checked (unreviewed) release cannot carry a fidelity verdict or signature")
        release_info = study.get("release") if isinstance(study.get("release"), dict) else {}
        arms = [a for a in protocol.get("arms") or [] if isinstance(a, dict)]
        if release_info.get("readiness") == "reconstruct":
            # Owner 2026-11-24: a paper that does not print every material ships with no runnable arm; every arm
            # states its gap and the researcher reconstructs it in their own copy before running.
            if any(a.get("runnable", True) is not False for a in arms) or not all(a.get("why") for a in arms):
                errors.append("protocol.json: a reconstruct release ships every arm not runnable, with its gap (why)")
        elif not any(a.get("runnable", True) is not False and not a.get("release_hold") for a in arms):
            errors.append("protocol.json: a checked release needs at least one runnable arm")
    elif not draft and (quality != "high_fidelity" or fidelity.get("verdict") not in SIGNED_VERDICTS \
            or not fidelity.get("signed_by")):
        errors.append("study.json: a release needs quality high_fidelity with a HIGH FIDELITY verdict and "
                      "signed_by (G2), or quality checked without a fidelity claim")
    if fidelity.get("review") and fidelity["review"] not in files:
        errors.append(f"study.json: fidelity.review {fidelity['review']} is not in the release")
    if study.get("release_hold"):
        errors.append("study.json: release_hold is set; held content is never released")
    held = [a.get("id") for a in protocol.get("arms") or [] if a.get("release_hold")]
    if held:
        errors.append(f"protocol.json: arms on release_hold {held}; release with hsenv release --drop-held-arms")
    if "arms" in study and study["arms"] != protocol_arms(protocol):
        errors.append("study.json: arms disagree with protocol.json (regenerate with hsenv study sync)")
    sha = schema_check.canonical_sha256(protocol)
    declared = study.get("protocol") or {}
    if declared.get("sha256") != sha:
        errors.append("study.json: protocol.sha256 does not match protocol.json")
    if declared.get("path", "protocol.json") != "protocol.json":
        errors.append("study.json: protocol.path must be protocol.json")
    open_assets = {hashlib.sha256(raw).hexdigest()[:16] for n, raw in files.items() if n.startswith("sources/open/")}
    # a held picture (distribution.held, kind image) is filled in from the researcher's own copy by its hash
    open_assets |= {str(e.get("sha256"))[:16] for e in ((protocol.get("distribution") or {}).get("held") or {}).values()
                    if isinstance(e, dict) and e.get("kind") == "image" and e.get("sha256")}
    for where, src in _image_sources(protocol):
        m = ASSET.match(src)
        if not (src.startswith(("https://", "http://")) or (m and m.group(1) in open_assets)):
            errors.append(f"protocol.json{where}: image source must be a URL or an asset: of a file in sources/open")
    spec, have = study.get("hsenv"), engine_version()
    if isinstance(spec, str) and have and not _version_ok(spec, have):
        errors.append(f"study.json: needs hsenv {spec}, the pinned engine is {have}")
    elif isinstance(spec, str) and not have:
        warnings.append("hsenv version unknown (not installed as a package); engine range not checked")

    # 3. checks
    report = _json(files, "review/checks-latest.json", errors)
    if report is None and draft and "review/checks-latest.json" not in files:
        warnings.append("Draft intake only: no execution or independent fidelity review has been approved.")
    elif not isinstance(report, dict):
        errors.append("review/checks-latest.json: missing; run hsenv checks and commit the report")
    else:
        if report.get("PASS") is not True:
            errors.append("review/checks-latest.json: checks did not PASS")
        if report.get("protocol_sha256") != sha:
            errors.append("review/checks-latest.json: stale (protocol_sha256 is not this protocol's sha256)")
        if report.get("release_hold"):
            errors.append("review/checks-latest.json: reports arms on release_hold")
        if quality == UNREVIEWED and report.get("public") is not True:
            errors.append("review/checks-latest.json: a checked release needs a public (--public) checks report")
    if not draft and quality == UNREVIEWED:
        evidence = study.get("quality_evidence")
        if evidence not in (None, [], ["review/checks-latest.json"]):
            errors.append("study.json: a checked release cites only review/checks-latest.json as quality evidence")
    if sessions and not errors and not draft and quality == UNREVIEWED:
        # The committed report is the maintainer's word; a checked release has no reviewer behind it, so CI runs
        # the full public checks itself and only its own result counts.
        fresh = public_checks(protocol, sessions)
        if fresh.get("PASS") is not True:
            errors.append("hsenv checks --public (re-run): did not PASS: " + json.dumps(
                {k: v for k, v in fresh.items() if k not in ("PASS",) and v}, ensure_ascii=False)[:600])
    fuzz = None
    if sessions and not errors and not draft:
        fuzz = replay(protocol, sessions, seed)
        errors.extend("engine: " + p for p in fuzz["problems"])

    # 4. effects and capabilities
    effects = _json(files, "effects.json", errors)
    if effects is not None:
        from scripts.research.effects_core import check as check_effects
        program = _json(files, "program.json", errors) or {}
        problems, warns = check_effects(Path("."), effects, program)
        errors.extend("effects.json: " + p for p in problems)
        warnings.extend("effects.json: " + w for w in warns)
    if "capabilities.json" in files:
        from scripts.research.revisions import capability_problems
        errors.extend(capability_problems(files))

    return {"status": "failed" if errors else "passed", "errors": errors, "warnings": warnings,
            "protocolSha256": sha, "summary": summary(study, protocol), **({"replay": fuzz} if fuzz else {})}


# A protocol with screening gates (attention checks, comprehension quizzes) completes few random-answer sessions, so
# at a small session count some stage or effect is not reached by chance. Such a failure is re-run once with more
# sessions; a larger run is a stricter coverage test, so this never relaxes the check.
COVERAGE_RETRY_SESSIONS = 1000
COVERAGE_RETRIES = (COVERAGE_RETRY_SESSIONS, 4000)   # a strict screen (1 in 27 random sessions completes) needs 4,000


def public_checks(protocol: dict, sessions: int) -> dict:
    """`python -m hsenv.checks <dir> --public --sessions N` on this protocol; its JSON report. A failure that is
    only random-answer coverage (unreached stages / effects not computed) is re-run at COVERAGE_RETRY_SESSIONS."""
    report = _public_checks(protocol, sessions)
    for more in COVERAGE_RETRIES:
        fuzz = report.get("fuzz") or {}
        if report.get("PASS") is True or sessions >= more or not (
                fuzz.get("unreached_stages") or fuzz.get("effects_not_computed")):
            break
        sessions = more
        report = _public_checks(protocol, sessions)
    return report


def _public_checks(protocol: dict, sessions: int) -> dict:
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False))
        out = subprocess.run([sys.executable, "-m", "hsenv.checks", ".", "--public", "--sessions", str(sessions)],
                             cwd=tmp, capture_output=True, text=True, timeout=1800)
    start = out.stdout.find("{")
    if start < 0:
        return {"PASS": False, "error": (out.stderr or out.stdout)[-400:]}
    try:
        return json.loads(out.stdout[start:])
    except ValueError:
        return {"PASS": False, "error": out.stdout[-400:]}


def replay(protocol: dict, sessions: int, seed: int = 1) -> dict:
    """References and N random sessions through the engine: the part of hsenv checks that needs no source
    text. Replaced by `hsenv check --public` once H1 ships it."""
    from hsenv import checks
    from hsenv.engine import Protocol, ProtocolError
    try:
        p = Protocol(protocol)
    except ProtocolError as exc:
        return {"problems": [f"protocol does not load: {exc}"]}
    problems = list(checks.check_refs(p))
    _, probs, stats = checks.play(p, sessions, seed, "random")
    if sessions < COVERAGE_RETRY_SESSIONS and stats.get("unreached_stages"):
        _, probs, stats = checks.play(p, COVERAGE_RETRY_SESSIONS, seed, "random")   # see COVERAGE_RETRY_SESSIONS
    problems += probs[:10]
    for key in ("unreached_stages", "leaks", "state_expr_leaks", "blocked_effects_not_computed"):
        if stats.get(key):
            problems.append(f"{key}: {stats[key]}")
    if stats.get("n_unanswerable"):
        problems.append(f"{stats['n_unanswerable']} unanswerable screens")
    return {"sessions": stats.get("sessions"), "problems": problems}


def summary(study: dict, protocol: dict) -> dict:
    """The research_index `protocol` block (§3.1). Counts and ids only, never protocol text."""
    analysis = protocol.get("analysis") or {}
    effects = analysis.get("effects") or []
    arms = []
    for arm in protocol.get("arms") or []:
        n = arm.get("n_paper")
        arms.append({"id": arm.get("id"), "runnable": bool(arm.get("runnable", True)),
                     "nPaper": n if isinstance(n, int) else None})
    fidelity = study.get("fidelity") or {}
    return {"slug": study.get("slug") or study.get("id"),
            "schemaVersion": protocol.get("schema_version"),
            "hsenv": study.get("hsenv"),
            "fidelity": {k: fidelity[k] for k in ("verdict", "signed_by") if k in fidelity},
            "arms": arms,
            "runExcluded": [x.get("id") if isinstance(x, dict) else x for x in protocol.get("run_excluded") or []],
            "effects": len(effects)}


def protocol_capabilities(protocol: dict) -> dict:
    """replayable = any arm runnable; humanBenchmark = an effect carries a reported human value (§3.1).
    hsenv.capabilities owns this once H1 ships it; this is the design's rule, nothing more."""
    try:  # pragma: no cover - depends on the hsenv release
        from hsenv.capabilities import protocol_capabilities as engine_rule  # type: ignore
        return engine_rule(protocol)
    except ImportError:
        pass
    arms = protocol.get("arms") or []
    effects = (protocol.get("analysis") or {}).get("effects") or []
    return {"replayable": any(a.get("runnable", True) and not a.get("release_hold") for a in arms),
            "humanBenchmark": sum(1 for e in effects if e.get("reported")) > 0}


def validate_revision_dir(folder: Path, *, sessions: int = 0) -> dict:
    """Check 5 on a stored revision, then checks 1-4 on its manifest files."""
    from scripts.research.revisions import load_revision
    folder = Path(folder)
    try:
        files = load_revision(folder.parent.parent, folder.name)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        return {"status": "failed", "errors": [f"manifest: {exc} (the directory name must be its digest)"],
                "warnings": []}
    return validate_protocol_files(files, sessions=sessions)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate protocol releases in community revisions")
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--sessions", type=int, default=0, help="also replay N random sessions in the engine")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    failed = False
    for d in a.dirs:
        result = validate_revision_dir(d, sessions=a.sessions)
        failed |= result["status"] != "passed"
        if a.json:
            print(json.dumps({"dir": str(d), **result}, ensure_ascii=False))
            continue
        print(f"{d}: {result['status']}")
        for e in result["errors"]:
            print(f"::error file={d}/study.json::{e}")
        for w in result["warnings"]:
            print(f"  warning: {w}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
