"""A deterministic, rebuildable index of research assets. No model calls or network access."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

VERSION = 1


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def normalise_doi(value: str) -> str:
    value = re.sub(r"^(https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value.strip(), flags=re.I).lower()
    return value if re.fullmatch(r"10\.\d{4,9}/\S+", value) else ""


def normalise(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def community_identity(meta: dict, paper_key: str) -> tuple[str, list[str]]:
    # Human-readable legacy slugs are not identities: long/non-Latin titles can collide.
    source = "community-title:" + digest({"title": normalise(meta.get("title", "")),
             "year": meta.get("year"), "authors": [normalise(str(a)) for a in meta.get("authors", [])]})
    pid, aliases = paper_identity(meta, source)
    return pid, list(dict.fromkeys(aliases + ["community:" + paper_key]))


def paper_identity(meta: dict, source_id: str) -> tuple[str, list[str]]:
    """DOI links exact identities. Titles produce candidates, never automatic identity merges."""
    doi = normalise_doi(str(meta.get("doi") or ""))
    arxiv = re.sub(r"v\d+$", "", str(meta.get("arxiv") or ""))
    alias = f"doi:{doi}" if doi else f"arxiv:{arxiv}" if re.fullmatch(r"\d{4}\.\d{4,5}", arxiv) else source_id
    return "paper_" + hashlib.sha256(alias.encode()).hexdigest()[:24], list(dict.fromkeys([alias, source_id]))


def strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        return [part for k, v in value.items() if k not in {"extensions", "verification"} for part in strings(v)]
    if isinstance(value, list):
        return [part for item in value for part in strings(item)]
    return []


def community_description(revision: Path, manifest: dict, program: dict) -> str:
    """Card text from the package's own overview, else the program's study titles; never new prose."""
    for entry in manifest["files"]:
        if re.fullmatch(r"package/[^/]+/study\.json", entry["path"]):
            overview = json.loads((revision / entry["path"]).read_text()).get("overview")
            if isinstance(overview, str) and overview.strip():
                return overview.strip()
    titles = [s.get("title") for s in program.get("studies", []) if isinstance(s.get("title"), str)]
    return f"{len(titles)} {'study' if len(titles) == 1 else 'studies'}: " + "; ".join(titles) if titles else ""


def project_program(program: dict) -> dict:
    """Keep complete searchable content and evidence IDs; never infer an effect estimate."""
    nodes = program.get("nodes", [])
    studies = []
    for study in program.get("studies", []):
        sid = study["id"]
        selected = [n for n in nodes if not n.get("studyIds") or sid in n["studyIds"]]
        studies.append({
            "id": sid, "title": study.get("title", sid),
            "description": study.get("description", ""),
            "dependsOn": study.get("dependsOn", []),
            "nodeIds": [n["id"] for n in selected],
            "evidenceIds": sorted({eid for n in selected for eid in n.get("evidenceIds", [])}),
            "searchText": "\n".join(strings(study) + [s for n in selected for s in strings(n)]),
        })
    # A result node's prose is searchable, but not a machine-ready Cohen's d or independent sample.
    return {"programId": program["id"], "schemaVersion": program["schemaVersion"],
            "studies": studies, "searchText": "\n".join(strings(program)),
            "issues": [{k: v for k, v in issue.items() if k != "extensions"}
                       for issue in program.get("issues", [])],
            "effectEstimates": []}


def taxonomy_errors(taxonomy: dict) -> list[str]:
    errors, concepts = [], taxonomy.get("concepts", [])
    ids = [c.get("id") for c in concepts]
    if len(set(ids)) != len(ids):
        errors.append("duplicate concept identity")
    known = set(ids)
    for c in concepts:
        if not c.get("id") or not c.get("definition") or not c.get("label"):
            errors.append("each concept needs id, label and definition")
        if c.get("status") == "reviewed" and not c.get("reviewedBy"):
            errors.append(f"{c.get('id')}: reviewed concepts need an accountable reviewer")
        for target in c.get("broader", []) + c.get("related", []):
            if target not in known or target == c.get("id"):
                errors.append(f"{c.get('id')}: invalid concept relation {target}")
    graph = {c.get("id"): c.get("broader", []) for c in concepts}
    def visit(cid, path):
        if cid in path:
            errors.append("taxonomy broader relationship contains a cycle")
            return
        for target in graph.get(cid, []):
            visit(target, path | {cid})
    for cid in graph:
        visit(cid, set())
    return sorted(set(errors))


def concept_matches(query: str, taxonomy: dict) -> list[str]:
    q = normalise(query)
    return [c["id"] for c in taxonomy.get("concepts", [])
            if c.get("status") == "reviewed" and q in
            {normalise(s) for s in [c["label"], *c.get("aliases", [])]}]


from scripts.research.effects_core import phenomena_of  # noqa: E402  (desktop copy)


def concept_suggestions(phrases: list[str], taxonomy: dict) -> list[str]:
    """Concept ids whose label or alias equals a phenomenon keyword. Suggestions only: a paper's
    reviewed concepts stay in conceptIds/conceptAssignments."""
    wanted = {normalise(p) for p in phrases if p}
    return sorted(c["id"] for c in taxonomy.get("concepts", [])
                  if wanted & {normalise(x) for x in [c.get("label", "")] + list(c.get("aliases", []))})


def effect_records(paper_id: str, effects: dict, program: dict, taxonomy: dict) -> list[dict]:
    """One flat record per effect of effects.json, for filtering across papers (by phenomenon, concept,
    design, outcome type, metric) and for pooling later. Reported numbers are copied as printed; nothing
    is converted here."""
    nodes = {n["id"]: n for n in program.get("nodes", [])}
    evidence = {e["id"]: e for e in program.get("evidence", [])}

    def statement(node_id: str):
        node = nodes.get(node_id, {})
        for f in node.get("fields", []):
            if f.get("id") == "statement" and isinstance(f.get("value"), str):
                return f["value"]
        return node.get("title")
    hyps = {h["id"]: h for h in effects.get("hypotheses", [])}
    out = []
    for e in effects.get("effects", []):
        hs = [hyps[h] for h in e.get("hypothesisIds", []) if h in hyps]
        phenomena = phenomena_of(e, effects.get("hypotheses", []))
        reported = e.get("reported")
        pages = sorted({evidence[x]["locator"].get("page") for x in (reported or {}).get("evidenceIds", [])
                        if x in evidence and evidence[x].get("locator", {}).get("page")})
        out.append({"id": paper_id + "#" + e["id"], "effectId": e["id"], "label": e.get("label"),
                    "role": e.get("role"), "studyId": e.get("studyId"),
                    "population": e.get("population") or {"kind": "human"},
                    "hypotheses": [{"id": h["id"], "nodeId": h.get("nodeId"), "statement": statement(h.get("nodeId"))}
                                   for h in hs],
                    "phenomena": phenomena, "conceptSuggestions": concept_suggestions(phenomena, taxonomy),
                    "topics": effects.get("topics", []), "design": e.get("design"), "unit": e.get("unit"),
                    "outcome": e.get("outcome"), "contrast": e.get("contrast"), "prediction": e.get("prediction"),
                    "computeOp": (e.get("compute") or {}).get("op"), "reported": reported,
                    "reportedPages": pages})
    return out


def build_catalog(root: Path, legacy: list[dict]) -> dict:
    taxonomy = json.loads((root / "catalog/taxonomy.json").read_text())
    errors = taxonomy_errors(taxonomy)
    if errors:
        raise ValueError("; ".join(errors))
    papers: dict[str, dict] = {}
    for item in legacy:
        sid = item["study_id"]
        meta = dict(item)
        # Use a DOI only when present in an actual source metadata field.
        metadata = root / "studies" / sid / "source/metadata.json"
        if metadata.is_file():
            raw = json.loads(metadata.read_text())
            if isinstance(raw, dict):
                meta["doi"] = raw.get("doi") or raw.get("DOI") or ""
        pid, aliases = paper_identity(meta, "bench:" + sid)
        paper = {"id": pid, "aliases": aliases, "title": item["title"],
                 "authors": item.get("authors", []), "year": item.get("year"),
                 "description": item.get("description", ""), "legacyStudyId": sid,
                 "contributors": item.get("contributors", []), "currentRevision": None,
                 "assetRoot": "studies/" + sid, "conceptIds": [], "conceptAssignments": [],
                 "capabilities": {"discoverable": True, "traceable": False,
                                  "reanalyzable": False, "replayable": False, "humanBenchmark": False},
                 "studies": [], "resources": [], "effectEstimates": [],
                 "searchText": "\n".join(strings(item)), "coverage": "legacy-metadata"}
        if pid in papers:
            papers[pid]["aliases"] = sorted(set(papers[pid]["aliases"] + aliases))
        else:
            papers[pid] = paper

    sources = root / "catalog/imports"
    for path in sorted(sources.glob("*.json")) if sources.exists() else []:
        batch = json.loads(path.read_text())
        for item in batch.get("papers", []):
            pid = item["id"]
            if pid in papers:
                target = papers[pid]
                target["aliases"] = sorted(set(target["aliases"] + item["aliases"]))
                target["resources"] += item.get("resources", [])
                target.setdefault("sourceRecords", []).append(item)
                target["searchText"] += "\n" + item.get("searchText", "")
            else:
                papers[pid] = item

    # Only current.json publishes a community asset. Proposals and older snapshots never become cards.
    for pointer in sorted((root / "community").glob("*/current.json")):
        current = json.loads(pointer.read_text())
        if not re.fullmatch(r"[a-f0-9]{64}", current.get("revision", "")):
            raise ValueError("invalid community revision identity")
        revision = pointer.parent / "revisions" / current["revision"]
        manifest = json.loads((revision / "manifest.json").read_text())
        if manifest.get("revision") != current["revision"] or digest([
                {"path": entry["path"], "sha256": entry["sha256"]} for entry in manifest["files"]]) != current["revision"]:
            raise ValueError("community revision manifest digest mismatch")
        for entry in manifest["files"]:
            file = revision / entry["path"]
            if not file.resolve().is_relative_to(revision.resolve()) or file.is_symlink() or hashlib.sha256(file.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"community revision content changed: {file}")
        program = json.loads((revision / "program.json").read_text())
        projection = project_program(program)
        paper = current["paper"]
        if not paper.get("description"):
            paper = {**paper, "description": community_description(revision, manifest, program)}
        pid = paper["id"]
        previous = papers.get(pid, {})
        declared = revision / "capabilities.json"  # what the study can do, with reasons (desktop #45)
        study_caps = json.loads(declared.read_text()) if any(e["path"] == "capabilities.json" for e in manifest["files"]) else None
        has_effects = any(e["path"] == "effects.json" for e in manifest["files"])
        estimates = effect_records(pid, json.loads((revision / "effects.json").read_text()), program, taxonomy) \
            if has_effects else previous.get("effectEstimates", [])
        papers[pid] = {**previous, **paper, **projection,
                       "aliases": sorted(set(previous.get("aliases", []) + paper.get("aliases", []))),
                       "currentRevision": current["revision"],
                       "assetRoot": revision.relative_to(root).as_posix(), "assetFiles": manifest["files"],
                       "coverage": "human-program", "capabilities": current["capabilities"],
                       **({"studyCapabilities": study_caps} if study_caps else {}),
                       "effectEstimates": estimates,
                       "conceptIds": previous.get("conceptIds", []),
                       "conceptAssignments": previous.get("conceptAssignments", [])}

    by_alias = {alias: p for p in papers.values() for alias in p.get("aliases", [])}
    for assignment in taxonomy.get("assignments", []):
        paper = by_alias.get(assignment["paperAlias"])
        if paper and assignment["status"] == "reviewed" and assignment.get("reviewedBy"):
            paper["conceptIds"] = sorted(set(paper.get("conceptIds", []) + [assignment["conceptId"]]))
            paper.setdefault("conceptAssignments", []).append(assignment)
    ordered = sorted(papers.values(), key=lambda p: p["id"])
    source_versions = {path.stem: json.loads(path.read_text())["sourceVersion"]
                       for path in sorted(sources.glob("*.json"))} if sources.exists() else {}
    return {"schemaVersion": VERSION, "taxonomyVersion": taxonomy["version"],
            "snapshot": digest({"papers": ordered, "taxonomy": taxonomy}),
            "sourceVersions": source_versions, "papers": ordered, "taxonomy": taxonomy,
            "coverage": {"papers": len(ordered), "scope": "indexed-corpus",
                         "builtPrograms": sum(bool(p.get("currentRevision")) for p in ordered)}}


def write_catalog(root: Path, legacy: list[dict]) -> dict:
    value = build_catalog(root, legacy)
    output = root / "co_website/data/research_index.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return value
