#!/usr/bin/env python3
"""Write capabilities.json: what this study can do, each answer with its reason.

Usage: python capabilities.py <project> [--check]

Every value is derived from files that already exist; nothing is guessed:

- read          program.json passes check_program.py
- run           the replay package, its readiness and the program's blocking issues; `excluded`
                entries are the build agent's own (kept from the previous capabilities.json)
- humanData     package/<slug>/human/ (loader and key.json: source, sample, pinned files) and
                artifacts/replay/human-check.json (arms covered, not mapped)
- authorCode    package/<slug>/analysis/author.json (the author-code key, HumanStudy-Bench#34)
- reproduced    artifacts/replay/replay-compare.json `humanMatchesPaper` (package analysis on human data
                vs the paper's reported results)
- profiles      the runner does not sample participant profiles yet

`--check` validates an existing capabilities.json and artifacts/failures.json and exits 1 on a problem.

run.excluded entries: {"what", "reason", "arms", "affectsArms", "effects", "field"}.
- `arms`        arms that cannot be run at all; tools refuse them (study_core.require_arms). An arm the package
                runs (task/task.json conditions without status "blocked") is never one of them.
- `affectsArms` arms that still run, with this part left out (a picture, a timer, a questionnaire, demographics).
- `effects`     effects.json effects the replay cannot compute; their results are not_ready. An effect whose arms
                are blocked (task.json status "blocked" or listed in some entry's `arms`) is listed here.
With task.json conditions, --check fails an `arms` entry naming an arm the package runs (above all one that would
leave no arm runnable) and a blocked effect missing from `effects`. These rules came with build contract 3:
for a package whose study.json says "buildContract" below 3 they print as WARN and do not fail.
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
RUN_RULES_CONTRACT = 3     # study.json buildContract from which the run.excluded arm rules fail --check
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


def _names(values):
    return [str(x) for x in values if isinstance(x, (str, int)) and not isinstance(x, bool)] \
        if isinstance(values, list) else []


def package_arms(root):
    """{"runnable": [...], "blocked": [...], "by": {program condition id or arm id: arm}, "study": {arm: study}}
    from package/<slug>/task/task.json `conditions` (a list of {id|arm, status, study, programConditionIds} or a
    mapping keyed by arm); None when the package names no conditions."""
    package = _package(Path(root))
    task = _json(package / "task/task.json") if package else None
    conds = task.get("conditions") if isinstance(task, dict) else None
    items = []
    if isinstance(conds, dict):
        items = [dict(v, id=k) if isinstance(v, dict) else {"id": k} for k, v in conds.items()]
    elif isinstance(conds, list):
        items = [c if isinstance(c, dict) else {"id": c} for c in conds if isinstance(c, (dict, str))]
    out = {"runnable": [], "blocked": [], "by": {}, "study": {}}
    for c in items:
        arm = c.get("id") or c.get("arm") or c.get("name")
        if not isinstance(arm, str) or not arm:
            continue
        blocked = c.get("status") == "blocked" or c.get("runnable") is False
        out["blocked" if blocked else "runnable"].append(arm)
        for alias in [arm, c.get("arm")] + _names(c.get("programConditionIds")):
            if isinstance(alias, str) and alias:
                out["by"].setdefault(alias, arm)
        if c.get("study") is not None:
            out["study"][arm] = str(c["study"])
    return out if out["runnable"] or out["blocked"] else None


def contract_level(root):
    """The highest study.json buildContract of the project's package (0 when none says)."""
    package = _package(Path(root))
    study = _json(package / "study.json") if package else None
    level = study.get("buildContract") if isinstance(study, dict) else None
    return level if isinstance(level, (int, float)) and not isinstance(level, bool) else 0


def effect_arms(effect, arms):
    """The package arms an effects.json effect contrasts (treatment, reference, conditionIds, groups)."""
    con = effect.get("contrast") if isinstance(effect.get("contrast"), dict) else {}
    names = [con.get("treatment"), con.get("reference")] + _names(con.get("conditionIds")) + _names(con.get("groups"))
    return sorted({arms["by"][n] for n in names if isinstance(n, str) and n in arms["by"]})


def blocked_effects(caps, effects, arms):
    """{effect id: why} for effects whose arms cannot run: every arm it contrasts is blocked (task.json status
    "blocked" or named in a run.excluded entry's `arms`), or every arm of its study is."""
    if not isinstance(effects, dict) or not arms:
        return {}
    refused = {a for item in ((caps or {}).get("run") or {}).get("excluded") or [] if isinstance(item, dict)
               for a in _names(item.get("arms"))}
    blocked = (set(arms["blocked"]) | refused) - set(arms["runnable"]) if arms["runnable"] else set(arms["blocked"]) | refused
    out = {}
    for e in effects.get("effects") or []:
        if not isinstance(e, dict) or not isinstance(e.get("id"), str):
            continue
        mine = effect_arms(e, arms)
        study_arms = [a for a, st in arms["study"].items() if st == str(e.get("studyId"))]
        if mine and any(a in blocked for a in mine):
            out[e["id"]] = "arm %s cannot be run" % ", ".join(a for a in mine if a in blocked)
        elif not mine and study_arms and all(a in blocked for a in study_arms):
            out[e["id"]] = "every arm of study %s cannot be run" % e.get("studyId")
    return out


def run_rules(caps, root):
    """(problems, warnings) for run.excluded against the package's arms and effects.json (see the docstring).
    Problems only when the package's study.json says buildContract >= RUN_RULES_CONTRACT."""
    found = []
    run = (caps or {}).get("run") if isinstance((caps or {}).get("run"), dict) else {}
    items = [x for x in run.get("excluded") or [] if isinstance(x, dict)]
    arms = package_arms(root)
    effects = _json(Path(root) / "effects.json")
    for i, item in enumerate(items):
        for key in ("arms", "affectsArms", "effects"):
            if key in item and not (isinstance(item[key], list) and all(isinstance(x, str) for x in item[key])):
                found.append(("problem", "run.excluded[%d].%s must be a list of names" % (i, key)))
    if arms:
        known = set(arms["runnable"]) | set(arms["blocked"])
        refused_all = set()
        for i, item in enumerate(items):
            listed = _names(item.get("arms"))
            refused_all |= set(listed)
            runs = [a for a in listed if a in arms["runnable"]]
            if runs:
                found.append(("rule", "run.excluded[%d] (%s): `arms` lists %s, which the package runs (task.json); "
                              "`arms` means the arm cannot be run and is refused. For a part left out inside arms "
                              "that still run, list them in `affectsArms`; for effects the replay cannot compute, "
                              "in `effects`" % (i, str(item.get("what"))[:60], ", ".join(runs))))
            for a in _names(item.get("arms")) + _names(item.get("affectsArms")):
                if a not in known and a not in arms["by"]:
                    found.append(("warning", "run.excluded[%d] names arm %r, which task.json does not list" % (i, a)))
        if arms["runnable"] and set(arms["runnable"]) <= refused_all and run.get("status") != "no":
            found.append(("rule", "run.excluded `arms` together list every arm the package runs (%s), so every "
                          "named-arm run is refused although run is %s; an exclusion of part of the content belongs "
                          "in `affectsArms`" % (", ".join(arms["runnable"]), run.get("status"))))
    if isinstance(effects, dict):
        ids = {e.get("id") for e in effects.get("effects") or [] if isinstance(e, dict)}
        listed = {x for item in items for x in _names(item.get("effects"))}
        for x in sorted(listed - ids):
            found.append(("warning", "run.excluded lists effect %r, which effects.json does not have" % x))
        for eid, why in sorted(blocked_effects(caps, effects, arms).items()):
            if eid not in listed:
                found.append(("rule", "effect %s cannot be computed (%s) but no run.excluded entry lists it in "
                              "`effects`; add it so its result is not_ready with this reason" % (eid, why)))
    strict = contract_level(root) >= RUN_RULES_CONTRACT
    problems = [m for k, m in found if k == "problem" or (k == "rule" and strict)]
    warnings = [m for k, m in found if k == "warning" or (k == "rule" and not strict)]
    return problems, warnings


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
    hkey = _json(package / "human/key.json") if package else None
    key = None
    if isinstance(hkey, dict):
        data = [f for f in hkey.get("files", []) if f.get("role", "data") == "data"]
        key = {"sample": hkey.get("sample"), "repository": (hkey.get("repository") or {}).get("url"),
               "license": (hkey.get("repository") or {}).get("license"), "dataFiles": len(data),
               "fetchable": sum(1 for f in data if f.get("url"))}
    where = ("source pinned in package/%s/human/key.json (%d of %d data files downloadable from it)"
             % (package.name, key["fetchable"], key["dataFiles"]) if key else
             "loader in package/%s/human/; no source key yet" % package.name) if package else None
    if check and has_loader:
        missing = [_gap(x) for x in check.get("notMapped") or []]
        caps["humanData"] = {"status": "partial" if missing else "yes",
                             "coverage": check.get("arms") or [], "sessions": check.get("sessions"),
                             "source": _human_source(package),
                             "missing": missing, "key": key, "reason": where}
    elif has_loader:
        caps["humanData"] = {"status": "unknown", "coverage": [], "missing": [], "key": key,
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
        warn = []
        if caps is not None and not bad:
            more, warn = run_rules(caps, root)
            bad += more
        for line in warn:
            print("WARN " + line + (" (older build: blocking from buildContract %d)" % RUN_RULES_CONTRACT
                                    if contract_level(root) < RUN_RULES_CONTRACT else ""))
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
