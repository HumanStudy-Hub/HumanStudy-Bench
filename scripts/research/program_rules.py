#!/usr/bin/env python3
"""Check program.json against the Human Program rules, offline, and print EVERY problem with a fix hint.

Usage: python .opencode/skills/build/check_program.py program.json

Mirrors the app's validateHumanProgram (app/src/shared/human-program.ts, collectProblems): unique ids,
every reference resolves (evidence, nodes, steps, studies, conditions, issue nodeId/stepId/studyId/fieldId),
condition rules, cycles, derivation, evidence sources, and "unresolved field/rule needs a review issue".
When the `jsonschema` package is importable it also validates schema/human-program.schema.json.
Exit 0 when clean, 1 when any problem is found.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def dups(ids):
    seen, out = set(), []
    for i in ids:
        if i in seen and i not in out:
            out.append(i)
        seen.add(i)
    return out


def problems_for(p: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    add = lambda msg, hint="": out.append((msg, hint))
    L = lambda k: [x for x in (p.get(k) or []) if isinstance(x, dict)]
    studies_l, nodes_l, steps_l, rels, evid_l, issues = (L(k) for k in ("studies", "nodes", "steps", "relations", "evidence", "issues"))
    for key, items in (("studies", studies_l), ("nodes", nodes_l), ("steps", steps_l), ("relations", rels), ("evidence", evid_l), ("issues", issues)):
        if dups([i.get("id") for i in items]):
            add(f"{key}: duplicate ID", f"give every entry in {key} its own id: {', '.join(map(str, dups([i.get('id') for i in items])))}")
    nodes = {n.get("id"): n for n in nodes_l}
    studies = {s.get("id") for s in studies_l}
    steps = {s.get("id"): s for s in steps_l}
    evidence = {e.get("id") for e in evid_l}

    def refs(ids, known, where, hint_for=None):
        ids = list(ids or [])
        if dups(ids):
            add(f"{where}: duplicate ID", "list each id once")
        for i in dict.fromkeys(ids):
            if i not in known:
                add(f"{where}: unknown reference {i}", (hint_for(i) if hint_for else None) or "the id does not exist; remove it or add the missing entry")

    def field(f, where):
        refs(f.get("evidenceIds"), evidence, where)
        refs(f.get("studyIds"), studies, where)
        if f.get("origin") == "derived" and not str(f.get("derivation") or "").strip():
            add(f"{where}: derived value requires derivation", "fill derivation or change origin to reported")

    for n in nodes_l:
        nid = n.get("id")
        if str(nid).startswith("flow:"):
            add("Node ID prefix flow: is reserved", f"rename node {nid}")
        refs(n.get("studyIds"), studies, nid)
        refs(n.get("evidenceIds"), evidence, nid)
        if dups([f.get("id") for f in n.get("fields") or []]):
            add(f"{nid}: duplicate ID", "give every field its own id")
        for f in n.get("fields") or []:
            field(f, f"{nid}.{f.get('id')}")
    node_hint = lambda i: f"{i} is a step, but this list holds node ids" if i in steps else None
    for s in steps_l:
        sid = s.get("id")
        if s.get("studyId") not in studies:
            add(f"{sid}: unknown reference {s.get('studyId')}", "studyId must name an entry of studies")
        for k in ("actorIds", "inputIds", "outputIds", "visibleIds"):
            refs(s.get(k), nodes, sid, node_hint)
        refs(s.get("children"), steps, sid)
        to = [e.get("to") for e in s.get("next") or []]
        refs(to, steps, sid)
        refs(s.get("evidenceIds"), evidence, sid)
        if s.get("rule"):
            field(s["rule"], sid)
        for i in [*(s.get("children") or []), *to]:
            if i in steps and steps[i].get("studyId") != s.get("studyId"):
                add(f"{sid}: cross-study flow edge", f"{i} belongs to another study")
    done: set = set()

    def visit(i, active):
        if i in done or i not in steps:
            return
        for c in steps[i].get("children") or []:
            if c in active or c == i:
                add("Procedure containment cycle", f"step {i} contains {c}, which already contains it")
            else:
                visit(c, active | {i})
        done.add(i)

    for s in steps_l:
        visit(s.get("id"), frozenset())
    finished: set = set()
    by_study = {s.get("id"): s for s in studies_l}

    def svisit(i, active):
        if i in finished or i not in by_study:
            return
        deps = by_study[i].get("dependsOn") or []
        refs(deps, studies, i)
        for d in deps:
            if d in active or d == i:
                add("Study dependency cycle", f"{i} depends on {d}, which already depends on it")
            else:
                svisit(d, active | {i})
        finished.add(i)

    for s in studies_l:
        svisit(s.get("id"), frozenset())

    def conditions(ids, scope, where):
        if not ids:
            return
        refs(ids, nodes, where)
        for i in ids:
            c = nodes.get(i)
            if c is None:
                continue
            if c.get("kind") != "condition":
                add(f"{where}: {i} is not a condition", 'conditionIds may only name nodes with kind "condition"')
            elif c.get("studyIds") and scope and not any(st in c["studyIds"] for st in scope):
                add(f"{where}: condition {i} belongs to another study", "use a condition of the same study")

    for n in nodes_l:
        conditions(n.get("conditionIds"), n.get("studyIds") or [], n.get("id"))
    for s in steps_l:
        conditions(s.get("conditionIds"), [s.get("studyId")], s.get("id"))
    for r in rels:
        for e in ([r.get("from")] if r.get("to") == r.get("from") else [r.get("from"), r.get("to")]):
            if e not in nodes:
                add(f"{r.get('id')}: unknown reference {e}", "relations connect node ids")
    for e in evid_l:
        if not (e.get("sourceId") or e.get("path") or e.get("url")):
            add(f"{e.get('id')}: source reference is required", "set sourceId, path or url")
        loc = e.get("locator") or {}
        if loc.get("start") is not None and loc.get("end") is not None and loc["end"] < loc["start"]:
            add(f"{e.get('id')}: reversed source range", "start must not exceed end")
    for i in issues:
        iid = i.get("id")
        refs(i.get("evidenceIds"), evidence, iid)
        if i.get("studyId"):
            refs([i["studyId"]], studies, iid)
        nid = i.get("nodeId")
        if nid and nid not in nodes:
            hint = f"nodeId {nid} is a step; use stepId" if nid in steps else f"nodeId {nid} is a study; use studyId" if nid in studies else "nodeId must be a node id"
            add(f"{iid}: unknown reference {nid}", hint)
        if i.get("stepId"):
            refs([i["stepId"]], steps, iid, lambda x: f"stepId {x} is a node; use nodeId" if x in nodes else None)
        fid = i.get("fieldId")
        if fid and (not nid or not any(f.get("id") == fid for f in (nodes.get(nid) or {}).get("fields") or [])):
            add(f"{iid}: unknown field reference", "fieldId needs a nodeId whose fields include it")
    for n in nodes_l:
        for f in n.get("fields") or []:
            if f.get("state") in ("missing", "decision") and not any(i.get("nodeId") == n.get("id") and (not i.get("fieldId") or i.get("fieldId") == f.get("id")) for i in issues):
                add(f"{n.get('id')}.{f.get('id')}: unresolved field requires a review issue", f'add an issue with nodeId "{n.get("id")}" and fieldId "{f.get("id")}"')
    for s in steps_l:
        r = s.get("rule")
        if r and r.get("state") in ("missing", "decision") and not any(i.get("stepId") == s.get("id") for i in issues):
            add(f"{s.get('id')}: unresolved rule requires a review issue", f'add an issue with stepId "{s.get("id")}"')
    return out


def schema_problems(p, path: Path) -> list[tuple[str, str]]:
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return []
    candidates = [path.resolve().parent / "schema/human-program.schema.json", Path(__file__).resolve().parents[3] / "schema/human-program.schema.json", Path(__file__).resolve().parents[2] / "schema/human-program.schema.json"]
    schema_path = next((c for c in candidates if c.is_file()), None)
    if schema_path is None:
        return []
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.validators.validator_for(schema)(schema)
    return [("schema: " + "/".join(map(str, e.absolute_path)) + ": " + e.message[:200], "match schema/human-program.schema.json") for e in sorted(validator.iter_errors(p), key=lambda e: list(map(str, e.absolute_path)))[:50]]


def check(p) -> list[tuple[str, str]]:
    if not isinstance(p, dict):
        return [("program.json must be a JSON object", "")]
    if p.get("schemaVersion") != 2:
        return [(f"unsupported schemaVersion {json.dumps(p.get('schemaVersion'))}", "set schemaVersion to 2")]
    return problems_for(p)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    path = Path(args[0] if args else "program.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL {path}: cannot read as JSON: {exc}")
        sys.exit(1)
    found = (check(data) if isinstance(data, dict) and data.get("schemaVersion") != 2 else schema_problems(data, path) + check(data))
    for msg, hint in found:
        print(f"PROBLEM {msg}" + (f"\n        fix: {hint}" if hint else ""))
    if found:
        print(f"FAIL {len(found)} problem(s) in {path}; fix them all and run this check again")
        sys.exit(1)
    print(f"PASS {path}")


if __name__ == "__main__":
    main()
