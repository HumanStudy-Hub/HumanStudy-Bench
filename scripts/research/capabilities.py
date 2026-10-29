#!/usr/bin/env python3
"""Write capabilities.json: what this study can do, each answer with its reason.

Usage: python capabilities.py <project> [--check]

Every value is derived from files that already exist; nothing is guessed:

- read          program.json passes check_program.py
- run           the replay package, its readiness and the program's blocking issues; `excluded`
                entries are the build agent's own (kept from the previous capabilities.json)
- humanData     package/<slug>/human/ and artifacts/replay/human-check.json (arms covered, not mapped)
- authorCode    package/<slug>/analysis/author.json (the author-code key, HumanStudy-Bench#34)
- reproduced    artifacts/replay/replay-compare.json `humanMatchesPaper` (package analysis on human data
                vs the paper's reported results)
- profiles      the runner does not sample participant profiles yet

`--check` validates an existing capabilities.json and artifacts/failures.json and exits 1 on a problem.
Standard library only, Python 3.8+. A copy of humanstudy-desktop pack/skills/build/capabilities.py
(cross-repo invariant: change both together). HumanStudy-Bench uses validate() for community revisions
and build() for the maintainer backfill.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

STATUS = ("yes", "partial", "no", "unknown")
REPRO = ("pass", "partial", "fail", None)
PATTERNS = ("non-text-stimuli", "materials-not-public", "missing-information", "interaction",
            "sequential-dependence", "modelling", "non-human-participants", "unsimulable-measure",
            "author-code-unusable", "author-code-not-reproducing", "human-data-incomplete", "other")


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _package(root):
    folders = sorted(d for d in (root / "package").glob("*") if (d / "task/adapter.py").is_file()) \
        if (root / "package").is_dir() else []
    return folders[0] if len(folders) == 1 else None


def _gap(item):
    """A not-mapped entry from human-check.json, as {what, reason}."""
    if isinstance(item, str):
        what, _, reason = item.partition(": ")
        return {"what": what, "reason": reason or "not in the human data"}
    what = " ".join(str(item[k]) for k in ("study", "arm", "effect") if item.get(k)) or "unnamed part"
    return {"what": what, "reason": item.get("reason") or "not in the human data"}


def _human_source(package):
    """The SOURCE string a human-data loader declares, read without running its code."""
    for path in sorted((package / "human").glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SOURCE" for t in node.targets):
                try:
                    value = ast.literal_eval(node.value)
                except ValueError:
                    continue
                if isinstance(value, str):
                    return value
    return None


def build(root, previous=None):
    root = Path(root)
    program = _json(root / "program.json")
    if not isinstance(program, dict):
        raise SystemExit("program.json is missing or invalid")
    previous = previous if previous is not None else (_json(root / "capabilities.json") or {})
    package = _package(root)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from check_program import check as check_program
        problems = check_program(program)
    except ImportError:  # outside the pack (an exported skill): reading still works
        problems = []
    caps = {"schemaVersion": 1,
            "derivedFrom": {"program": hashlib.sha256((root / "program.json").read_bytes()).hexdigest(),
                            "package": package.name if package else None},
            "read": {"status": "partial" if problems else "yes",
                     "reason": "%d program check problem(s); see check_program.py" % len(problems) if problems else None}}

    # run
    blocking = [i["id"] for i in program.get("issues", []) if i.get("severity") == "blocking"]
    readiness = (_json(package / "study.json") or {}).get("readiness", {}) if package else {}
    excluded = (previous.get("run") or {}).get("excluded") or []
    if not package:
        run = {"status": "no", "reason": "no replay package was built"}
    elif blocking or readiness.get("status") == "blocked":
        ids = blocking or readiness.get("blocking") or []
        run = {"status": "no", "reason": "blocked by open issues: " + ", ".join(ids), "blocking": ids}
    elif excluded:
        run = {"status": "partial", "reason": "%d part(s) of the paper cannot be run; see excluded" % len(excluded)}
    else:
        run = {"status": "yes", "reason": None}
    # The runner passes one text prompt and reads one text reply per call.
    run.update({"modalities": ["text"], "excluded": excluded})
    caps["run"] = run

    # human data
    check = _json(root / "artifacts/replay/human-check.json")
    has_loader = bool(package and any((package / "human").glob("*.py")))
    if check and has_loader:
        missing = [_gap(x) for x in check.get("notMapped") or []]
        caps["humanData"] = {"status": "partial" if missing else "yes",
                             "coverage": check.get("arms") or [], "sessions": check.get("sessions"),
                             "source": _human_source(package),
                             "missing": missing, "key": None,
                             "reason": "loader in package/%s/human/; source key not pinned yet (#45 milestone 2)" % package.name}
    elif has_loader:
        caps["humanData"] = {"status": "unknown", "coverage": [], "missing": [], "key": None,
                             "reason": "a human-data loader exists but has not been checked (no artifacts/replay/human-check.json)"}
    else:
        caps["humanData"] = {"status": "no", "coverage": [], "missing": [], "key": None,
                             "reason": "no human data connected to this package"}

    # author code
    author = _json(package / "analysis/author.json") if package else None
    if isinstance(author, dict):
        caps["authorCode"] = {"status": "yes" if author.get("route") else "partial", "route": author.get("route"),
                              "reason": author.get("note")}
    else:
        caps["authorCode"] = {"status": "no", "route": None, "reason": "no author-code key recorded"}

    # reproduction: package analysis on the human data vs the paper's reported results
    compare = _json(root / "artifacts/replay/replay-compare.json") or {}
    hm = compare.get("humanMatchesPaper")
    verdict, detail = None, "not checked"
    if isinstance(hm, list) and len(hm) == 2 and hm[1]:
        verdict = "pass" if hm[0] == hm[1] else "fail" if hm[0] == 0 else "partial"
        detail = ("%d of %d reported results recovered by the package analysis from the connected human data "
                  "(see humanData.source: it may be a replication sample, not the original participants)" % tuple(hm))
    elif isinstance(hm, list) and len(hm) == 2:
        detail = "no reported result could be compared with the human data"
    caps["reproduced"] = {"package": verdict, "author": None, "detail": detail}
    caps["profiles"] = {"status": "no", "reason": "the runner does not sample participant profiles yet"}
    return caps


def validate(caps, failures=None):
    """Problems in a capabilities object and an optional failures list (empty when valid)."""
    out = []
    if not isinstance(caps, dict) or caps.get("schemaVersion") != 1:
        return ["capabilities.json: schemaVersion must be 1"]
    for key in ("read", "run", "humanData", "authorCode", "profiles"):
        part = caps.get(key)
        if not isinstance(part, dict) or part.get("status") not in STATUS:
            out.append("capabilities.json: %s.status must be one of %s" % (key, "/".join(STATUS)))
        elif part["status"] in ("partial", "no") and not part.get("reason"):
            out.append("capabilities.json: %s is %s without a reason" % (key, part["status"]))
    for item in (caps.get("run") or {}).get("excluded") or []:
        if not (isinstance(item, dict) and item.get("what") and item.get("reason")):
            out.append("capabilities.json: every run.excluded entry needs what and reason")
    rep = caps.get("reproduced")
    if not isinstance(rep, dict) or rep.get("package") not in REPRO or rep.get("author") not in REPRO:
        out.append("capabilities.json: reproduced.package/author must be pass, partial, fail or null")
    for i, f in enumerate(failures or []):
        if not isinstance(f, dict):
            out.append("failures.json[%d]: not an object" % i)
            continue
        if f.get("pattern") not in PATTERNS:
            out.append("failures.json[%d]: pattern must be one of %s" % (i, ", ".join(PATTERNS)))
        for key in ("where", "what", "fallback"):
            if not f.get(key):
                out.append("failures.json[%d]: missing %s" % (i, key))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = Path(args[0] if args else ".").resolve()
    if "--check" in sys.argv:
        caps = _json(root / "capabilities.json")
        failures = _json(root / "artifacts/failures.json")
        if failures is not None and not isinstance(failures, list):
            print("FAIL artifacts/failures.json must be a list")
            return 1
        bad = validate(caps, failures) if caps is not None else ["capabilities.json is missing or invalid"]
        for line in bad:
            print("FAIL " + line)
        print("PASS capabilities" if not bad else "")
        return 1 if bad else 0
    caps = build(root)
    bad = validate(caps)
    if bad:
        print("\n".join("FAIL " + b for b in bad))
        return 1
    (root / "capabilities.json").write_text(json.dumps(caps, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote capabilities.json: read %s, run %s, human data %s, author code %s, reproduced %s"
          % (caps["read"]["status"], caps["run"]["status"], caps["humanData"]["status"],
             caps["authorCode"]["status"], caps["reproduced"]["package"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
