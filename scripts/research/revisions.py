"""Immutable community revisions and a three-way merge. Never execute contributed code here."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator
from scripts.research.catalog import community_identity, digest, normalise_doi, project_program
from scripts.research.capabilities import validate as check_capabilities
from scripts.research.effects_core import check as check_effects
from scripts.research.program_rules import problems_for

ROOT = Path(__file__).resolve().parents[2]
MISSING = object()
# A package in a community paper is slim (desktop #45): study.json refers to ../../program.json and the
# evidence and missing-information lists are read from program.json. Packages that still hold a program
# copy must also hold the two view files (the standalone eight-file contract).
CORE = ("study.json", "source/paper_metadata.json", "materials/materials.json", "task/task.json",
        "task/adapter.py", "evaluation/evaluation.py")
STANDALONE_VIEWS = ("source/evidence.json", "audit/missing_information.json")
PROGRAM_REF = "../../program.json"
TOP_LEVEL = {"program.json", "capabilities.json", "effects.json"}
PRIVATE = {".env", ".git", ".opencode", ".humanstudy", "node_modules", "__pycache__",
           "threads", "conversations", "chat", "chats", "credentials", "account.json", "settings.json"}


def safe_path(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name or ":" in name:
        raise ValueError("invalid asset path")
    parts = name.split("/")
    if any(not p or p in {".", ".."} for p in parts) or PurePosixPath(name).is_absolute():
        raise ValueError("asset paths must stay inside the contribution")
    if any(p.lower() in PRIVATE or p.lower().startswith(".env.") for p in parts):
        raise ValueError("private project files are not community assets")
    if name not in TOP_LEVEL and parts[0] not in {"package", "artifacts", "sources"}:
        raise ValueError(f"not a public research asset: {name}")
    return name


def read_assets(directory: Path) -> dict[str, bytes]:
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlinks are not community assets")
        if path.is_file():
            name = path.relative_to(directory).as_posix()
            if name == "CONTRIBUTION.json":
                continue
            files[safe_path(name)] = path.read_bytes()
    return files


def validate_assets(files: dict[str, bytes]) -> dict:
    errors = []
    for name in files:
        try:
            safe_path(name)
        except ValueError as exc:
            errors.append(str(exc))
    try:
        program = json.loads(files["program.json"])
        schema = json.loads((ROOT / "contracts/human-program.schema.json").read_text())
        errors.extend("program: " + e.message for e in Draft202012Validator(schema).iter_errors(program))
        if not errors:
            errors.extend(f"{message}: {hint}" for message, hint in problems_for(program))
    except (KeyError, ValueError, TypeError) as exc:
        return {"status": "failed", "errors": errors + [f"program.json is missing or invalid: {exc}"]}
    if errors:
        return {"status": "failed", "errors": errors}
    package_roots = sorted({"/".join(name.split("/")[:2]) for name in files
                            if name.startswith("package/") and len(name.split("/")) >= 3})
    for package in package_roots:
        for rel in CORE:
            if package + "/" + rel not in files:
                errors.append(f"{package}: missing {rel}")
        parsed = {}
        for name, raw in files.items():
            if name.startswith(package + "/") and name.endswith(".json"):
                try:
                    parsed[name[len(package) + 1:]] = json.loads(raw)
                except ValueError:
                    errors.append(f"{name}: invalid JSON")
        study, task = parsed.get("study.json", {}), parsed.get("task/task.json", {})
        if "program" in study:
            if study["program"] != program:
                errors.append(f"{package}: study.json.program must equal program.json")
            errors.extend(f"{package}: missing {rel} (needed while study.json holds a program copy)"
                          for rel in STANDALONE_VIEWS if package + "/" + rel not in files)
        elif study.get("programRef") != {"path": PROGRAM_REF, "id": program["id"]}:
            errors.append(f'{package}: study.json needs programRef {{"path": "{PROGRAM_REF}", "id": "{program["id"]}"}}')
        expected = {"programId": program["id"], "stepIds": [s["id"] for s in program["steps"]],
                    "recordNodeIds": [n["id"] for n in program["nodes"] if n["kind"] in {"record", "variable"}],
                    "analysisNodeIds": [n["id"] for n in program["nodes"] if n["kind"] == "analysis"]}
        if task.get("programBindings") != expected:
            errors.append(f"{package}: task bindings disagree with the program")
        for rel, function in [("task/adapter.py", "run_sessions"), ("evaluation/evaluation.py", "evaluate")]:
            try:
                tree = ast.parse(files.get(package + "/" + rel, b""))
                if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function for n in tree.body):
                    errors.append(f"{package}/{rel}: missing {function}")
            except (SyntaxError, ValueError):
                errors.append(f"{package}/{rel}: invalid Python")
    errors.extend(capability_problems(files))
    errors.extend(effects_problems(files, program))
    return {"status": "failed" if errors else "passed", "errors": errors,
            "checks": {"schema": not errors, "packageBindings": bool(package_roots) and not errors,
                       "codeExecuted": False, "scientificReview": False},
            "capabilities": {"discoverable": True, "traceable": False,
                             "reanalyzable": False, "replayable": False, "humanBenchmark": False},
            "programDigest": digest(program), "projection": project_program(program)}


def effects_problems(files: dict[str, bytes], program: dict) -> list[str]:
    """effects.json, when present, follows the desktop contract (desktop effects_core.py): every program
    hypothesis listed, effects linked to program nodes, every reported number written in a cited quote."""
    raw = files.get("effects.json")
    if raw is None:
        return []
    try:
        effects = json.loads(raw)
    except ValueError:
        return ["effects.json: invalid JSON"]
    problems, _warnings = check_effects(Path("."), effects, program)
    return ["effects.json: " + p for p in problems]


def capability_problems(files: dict[str, bytes]) -> list[str]:
    """capabilities.json and artifacts/failures.json, when present, follow the desktop contract."""
    caps, failures = files.get("capabilities.json"), files.get("artifacts/failures.json")
    if caps is None and failures is None:
        return []
    try:
        caps_data = json.loads(caps) if caps is not None else None
        fail_data = json.loads(failures) if failures is not None else []
    except ValueError:
        return ["capabilities.json or artifacts/failures.json: invalid JSON"]
    if not isinstance(fail_data, list):
        return ["artifacts/failures.json must be a list"]
    if caps_data is None:
        caps_data = {"schemaVersion": 1, **{k: {"status": "unknown"} for k in ("read", "run", "humanData", "authorCode", "profiles")},
                     "reproduced": {"package": None, "author": None}}
    return check_capabilities(caps_data, fail_data)


def merge_json(base: Any, current: Any, proposed: Any, path: str = "") -> tuple[Any, list[str]]:
    """Lists with IDs merge by identity. Ordinary lists remain atomic (order has meaning)."""
    if current == proposed or proposed == base:
        return (MISSING if current is MISSING else deepcopy(current)), []
    if current == base:
        return (MISSING if proposed is MISSING else deepcopy(proposed)), []
    if all(isinstance(v, dict) for v in (base, current, proposed)):
        result, conflicts = {}, []
        for key in sorted(set(base) | set(current) | set(proposed)):
            value, found = merge_json(base.get(key, MISSING), current.get(key, MISSING),
                                      proposed.get(key, MISSING), path + "/" + key)
            if value is not MISSING:
                result[key] = value
            conflicts += found
        return result, conflicts
    if all(isinstance(v, list) for v in (base, current, proposed)):
        if all(all(isinstance(x, dict) and isinstance(x.get("id"), str) for x in v) for v in (base, current, proposed)):
            if all(len({x["id"] for x in v}) == len(v) for v in (base, current, proposed)):
                maps = [{x["id"]: x for x in v} for v in (base, current, proposed)]
                merged, conflicts = merge_json(*maps, path)
                order = list(dict.fromkeys([x["id"] for x in current] + [x["id"] for x in proposed]))
                return [merged[key] for key in order if key in merged], conflicts
    # Never automatically select a scientific interpretation in a conflict.
    return current, [path or "/"]


def merge_assets(base: dict[str, bytes], current: dict[str, bytes], proposed: dict[str, bytes]) -> tuple[dict, list]:
    result, conflicts = {}, []
    for name in sorted(set(base) | set(current) | set(proposed)):
        b, c, p = base.get(name, MISSING), current.get(name, MISSING), proposed.get(name, MISSING)
        if c == p or p == b:
            value = c
        elif c == b:
            value = p
        elif name.endswith(".json") and all(isinstance(v, bytes) for v in (b, c, p)):
            try:
                merged, found = merge_json(json.loads(b), json.loads(c), json.loads(p), name)
                value = (json.dumps(merged, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
                conflicts += found
            except ValueError:
                value = c
                conflicts.append(name)
        else:
            value = c
            conflicts.append(name)
        if value is not MISSING:
            result[name] = value
    # study.json.program (standalone layout) is a derived copy, not a competing editable design; a
    # slim study.json only names the program.
    if not conflicts and "program.json" in result:
        program = json.loads(result["program.json"])
        for name in result:
            if name.startswith("package/") and name.endswith("/study.json"):
                data = json.loads(result[name])
                if "program" in data:
                    data["program"] = program
                else:
                    data["programRef"] = {"path": PROGRAM_REF, "id": program["id"]}
                result[name] = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode()
    return result, conflicts


def revision_id(files: dict[str, bytes]) -> str:
    return digest([{ "path": name, "sha256": hashlib.sha256(raw).hexdigest()}
                   for name, raw in sorted(files.items())])


def load_revision(paper_dir: Path, revision: str | None) -> dict[str, bytes]:
    if revision is None:
        return {}
    if not re.fullmatch(r"[a-f0-9]{64}", revision):
        raise ValueError("invalid base revision")
    folder = paper_dir / "revisions" / revision
    manifest = json.loads((folder / "manifest.json").read_text())
    files = {}
    for item in manifest["files"]:
        name = safe_path(item["path"])
        path = folder / name
        if path.is_symlink():
            raise ValueError("revision contains a symlink")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("immutable revision changed")
        files[name] = raw
    if revision_id(files) != revision:
        raise ValueError("revision manifest digest mismatch")
    return files


def integrate(paper_dir: Path, metadata: dict, proposed: dict[str, bytes], *, review: dict | None = None) -> dict:
    """Prepare a full revision and publish pointer under a per-paper lock, with expected-base checks.

    A trusted reviewer must independently attest the exact merged content. The worker never
    trusts `checks` or `review` uploaded with a contribution. On GitHub, the main-ref CAS is
    the final concurrency boundary; retry by recomputing against the new main, not force-push.
    """
    paper_dir.mkdir(parents=True, exist_ok=True)
    lock = paper_dir / ".integrating"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return {"status": "retry", "reason": "Another contribution is being integrated."}
    os.close(fd)
    try:
        pointer = paper_dir / "current.json"
        key = metadata.get("paperKey", "")
        if key.startswith("doi:") and normalise_doi(str(metadata.get("paper", {}).get("doi", ""))) != key[4:]:
            return {"status": "checks_failed", "reason": "The paper DOI and contribution identity disagree."}
        current = json.loads(pointer.read_text()) if pointer.exists() else None
        current_id = current["revision"] if current else None
        if current_id == revision_id(proposed):
            return {"status": "unchanged", "revision": current_id}
        if metadata.get("_identityCandidates"):
            return {"status": "needs_review", "reason": "A catalog paper may already exist. Resolve its identity before publishing another entry.", "paperCandidates": metadata["_identityCandidates"]}
        baseline = metadata.get("baseRevision")
        if current and not baseline:
            return {"status": "needs_review", "reason": "The original contribution baseline is unknown."}
        if metadata.get("intent", "reconstruct") != "reconstruct":
            return {"status": "needs_review", "reason": "A new research design cannot replace the original paper."}
        try:
            base_files = load_revision(paper_dir, baseline)
            current_files = load_revision(paper_dir, current_id)
        except (ValueError, FileNotFoundError):
            return {"status": "needs_review", "reason": "The contribution baseline is unavailable or invalid."}
        merged, conflicts = merge_assets(base_files, current_files, proposed)
        if conflicts:
            return {"status": "needs_review", "conflicts": conflicts,
                    "reason": "These changes disagree with the current research; the previous version is retained."}
        validation = validate_assets(merged)
        if validation["status"] != "passed":
            return {"status": "checks_failed", "validation": validation}
        source_doi = json.loads(merged["program.json"])["source"].get("doi")
        if source_doi and key.startswith("doi:") and normalise_doi(source_doi) != key[4:]:
            return {"status": "needs_review", "reason": "The program and contribution refer to different source papers."}
        rid = revision_id(merged)
        if rid == current_id:
            return {"status": "unchanged", "revision": rid}
        # An independent agent or curator reviews an exact candidate; approval cannot be replayed.
        if not review or review.get("candidateRevision") != rid or review.get("decision") != "accept" or not review.get("reviewer"):
            return {"status": "needs_review", "candidateRevision": rid, "validation": validation,
                    "reason": "Structural checks passed; source fidelity and scientific claims still need independent review."}
        folder = paper_dir / "revisions" / rid
        if not folder.exists():
            folder.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=folder.parent))
            try:
                entries = []
                for name, raw in sorted(merged.items()):
                    target = staging / safe_path(name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
                    entries.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
                (staging / "manifest.json").write_text(json.dumps({"schemaVersion": 1, "revision": rid, "files": entries}, indent=2) + "\n")
                os.replace(staging, folder)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        else:
            load_revision(paper_dir, rid)
        source = json.loads(merged["program.json"])["source"]
        pid, aliases = community_identity(metadata.get("paper", source), metadata["paperKey"])
        resolved = metadata.get("_resolvedPaper", {})
        pid = resolved.get("id", current["paper"]["id"] if current else pid)
        aliases = list(dict.fromkeys(aliases + resolved.get("aliases", []) + (current["paper"].get("aliases", []) if current else [])))
        paper = {**metadata.get("paper", {}), **resolved, "id": pid, "paperKey": metadata["paperKey"], "aliases": aliases}
        authors = list(current.get("contributors", [])) if current else []
        contributor = metadata.get("contributor")
        if contributor and contributor not in authors:
            authors.append(contributor)
        updated = {"schemaVersion": 1, "revision": rid, "previousRevision": current_id,
                   "paper": {**paper, "contributors": authors}, "contributors": authors,
                   "capabilities": {**validation["capabilities"], **review.get("verifiedCapabilities", {})},
                   "review": review, "validation": {k: v for k, v in validation.items() if k != "projection"}}
        # Projection is written before the authoritative pointer; a failed write cannot publish half a revision.
        (folder / "projection.json").write_text(json.dumps(validation["projection"], ensure_ascii=False, indent=2) + "\n")
        temporary = paper_dir / ".current.tmp"
        temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, pointer)
        return {"status": "accepted", "revision": rid, "previousRevision": current_id}
    finally:
        lock.unlink(missing_ok=True)
