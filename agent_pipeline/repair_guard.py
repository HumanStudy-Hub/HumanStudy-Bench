"""Deterministic fidelity guard for the validator repair loop.

The repair turn is told, in its prompt, never to fill a gap or rewrite a
transcribed instrument. This module checks that it did not, by comparing the
package snapshot taken before repair with the package after repair. The repair
is rejected when:

- an unresolved item (labelled ``missing``, or in state ``missing``/``decision``)
  stops being unresolved, is removed, or changes anything other than its
  evidence/study ids, its label (while staying unresolved), its
  ``extensions.material`` classification or the removal of a derivation;
- a ``verbatim`` item is relabelled, removed, or any part of it other than its
  own top-level metadata changes (nested text such as scale anchors counts);
- an evidence record changes, is removed, or is added (unless it copies a record
  that already existed elsewhere in the package);
- an entry of ``audit/missing_information.json`` changes or is removed;
- the repair adds an item, or changes an item's value or label, and the result
  is resolved without an evidence basis: it must be ``verbatim``, ``reported`` or
  ``derived`` and cite evidence ids that exist in the package. ``researcher``,
  ``implementation`` and ``unknown`` values cannot be introduced by a repair;
- a JSON file that did not parse before repair was changed, because nothing in
  it could be compared.

Items are found structurally in every JSON file of the paper folder: any object
that carries an evidence label (``origin``, ``evidence_label``, ``evidenceLabel``
or ``label`` set to a known origin) or a ``state``. Items are matched between
the two snapshots by file and JSON path. List entries are keyed by ``id`` when
they have one and by a hash of their content otherwise, so reordering or
inserting into a list does not misalign the comparison.

Evidence records are objects with an ``id`` and a ``quote`` anywhere in the
package. Legacy Studio sidecars (``status`` reported/implementation/unresolved)
are guarded like v2 fields. ``studio-model.json`` is skipped only when it and
``study.json`` both carry a v2 program, because the validator then requires the
two to be identical and ``study.json`` is guarded.

Known limits: id-less list entries are matched by a content hash, so an edit
inside one entry is seen as that entry being replaced; the guard cannot judge
whether a cited, existing evidence record supports a value; non-JSON files are
listed and diffed for reviewers but not judged.
"""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

EVIDENCE_LABELS = ("verbatim", "reported", "derived")
ORIGINS = EVIDENCE_LABELS + ("missing", "researcher", "implementation", "unknown")
LABEL_KEYS = ("origin", "evidence_label", "evidenceLabel", "label")
UNRESOLVED_STATES = ("missing", "decision")
ITEM_STATES = ("confirmed", "check") + UNRESOLVED_STATES
LEGACY_STATUSES = ("reported", "implementation", "unresolved")  # Studio StudySchema sidecars
EVIDENCE_ID_KEYS = ("evidenceIds", "evidence_ids")
ID_KEYS = EVIDENCE_ID_KEYS + ("studyIds",)
# Top-level keys that describe an item rather than carry its value.
ITEM_METADATA = frozenset(LABEL_KEYS + ID_KEYS + ("state", "status", "derivation", "note", "notes"))
# Keys ignored when hashing an id-less list entry to match it across snapshots.
HASH_IGNORED = ITEM_METADATA | {"extensions"}
EVIDENCE_FIELDS = ("quote", "locator", "path", "url", "sourceId", "verification", "page")
AUDIT_FILE = "audit/missing_information.json"
SKIPPED_FILES = frozenset(("studio-model.json",))

Path_ = Tuple[str, ...]


def _origin(item: dict) -> Optional[str]:
    for key in LABEL_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value in ORIGINS:
            return value
    return None


def _unresolved(item: dict) -> bool:
    return (_origin(item) == "missing" or item.get("state") in UNRESOLVED_STATES
            or item.get("status") == "unresolved")


def _value(item: dict) -> dict:
    """Everything except the item's own top-level metadata. Nested objects are
    kept whole, so a nested ``label`` or ``note`` is treated as content. A
    top-level ``label`` is metadata only when it holds an evidence label;
    otherwise it is text (a display name or a response option)."""
    return {k: v for k, v in item.items()
            if k not in ITEM_METADATA or (k == "label" and not (isinstance(v, str) and v in ORIGINS))}


def _unresolved_value(item: dict, before: Optional[dict] = None) -> dict:
    """The value of an unresolved item, ignoring an ``extensions.material``
    classification the repair adds to a gap that had none (the package check
    requires one). An existing classification, with its search record, is
    compared like any other content."""
    value = _value(item)
    if before is not None and isinstance(before.get("extensions"), dict) and "material" in before["extensions"]:
        return value
    extensions = value.get("extensions")
    if isinstance(extensions, dict) and "material" in extensions:
        rest = {k: v for k, v in extensions.items() if k != "material"}
        if rest:
            value["extensions"] = rest
        else:
            del value["extensions"]
    return value


def _empty(value: object) -> bool:
    return value is None or (isinstance(value, (str, list, dict)) and not value)


def _ids(item: dict, keys=EVIDENCE_ID_KEYS) -> List[str]:
    for key in keys:
        ids = item.get(key)
        if isinstance(ids, list):
            return [i for i in ids if isinstance(i, str) and i.strip()]
    return []


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in HASH_IGNORED}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _segment(child: Any, index: int, seen: Dict[str, int]) -> str:
    if isinstance(child, dict) and isinstance(child.get("id"), str) and child["id"]:
        base = "#" + child["id"]
    elif isinstance(child, (dict, list)):
        digest = hashlib.sha1(json.dumps(_strip(child), sort_keys=True, default=str).encode()).hexdigest()[:10]
        base = "~" + digest
    else:
        return f"[{index}]"
    seen[base] = seen.get(base, 0) + 1
    return base if seen[base] == 1 else f"{base}@{seen[base]}"


def _walk(value: Any, path: Path_, items: Dict[Path_, dict], evidence: Dict[Path_, dict]) -> None:
    if isinstance(value, dict):
        if _origin(value) is not None or value.get("state") in ITEM_STATES or value.get("status") in LEGACY_STATUSES:
            items[path] = value
        # An evidence record is any object with an id and a quote, wherever it
        # sits; evidence lists also register quote-less records so ids resolve.
        if isinstance(value.get("id"), str) and value["id"] and "quote" in value:
            evidence[path] = value
        for key, child in value.items():
            if key == "evidence" and isinstance(child, list):
                for entry in child:
                    if isinstance(entry, dict) and isinstance(entry.get("id"), str) and entry["id"] \
                            and "quote" not in entry:
                        evidence[path + ("evidence", "#" + entry["id"])] = entry
            _walk(child, path + (key,), items, evidence)
    elif isinstance(value, list):
        seen: Dict[str, int] = {}
        for index, child in enumerate(value):
            _walk(child, path + (_segment(child, index, seen),), items, evidence)


def _audit_entries(document: Any) -> Dict[Tuple[str, str], Any]:
    entries: Dict[Tuple[str, str], Any] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("field"), str):
                entries[(str(value.get("study", "")), value["field"])] = value
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(document)
    return entries


def _where(path: Path_) -> str:
    return path[0] + ":" + ".".join(path[1:]) if len(path) > 1 else path[0]


def _evidence_record(entry: dict) -> dict:
    return {k: entry.get(k) for k in EVIDENCE_FIELDS}


def _skipped(documents: Dict[str, Any]) -> set:
    """studio-model.json is skipped only when it and study.json both carry a v2
    program; the validator then requires the two to be identical."""
    study, model = documents.get("study.json"), documents.get("studio-model.json")
    if isinstance(study, dict) and "program" in study and isinstance(model, dict) \
            and ("program" in model or "schemaVersion" in model):
        return set(SKIPPED_FILES)
    return set()


Removable = Callable[[dict, Optional[dict]], bool]


def _under(path: Path_, roots: set) -> bool:
    return any(path[:len(root)] == root for root in roots)


def compare_packages(pre: Dict[str, Any], post: Dict[str, Any], removable: Optional[Removable] = None) -> dict:
    """Compare two packages given as ``{relative json path: parsed json}``.

    ``removable(before, after)`` may allow a verbatim item to be removed (``after`` is None)
    or marked as not shown: the survey flow check's repair for text participants never saw.

    A file that did not parse is given as ``Unparsed(text)``. Pure function:
    returns ``{"passed", "violations", "checked"}`` without touching the disk.
    """
    violations: List[str] = []
    pre_items: Dict[Path_, dict] = {}
    post_items: Dict[Path_, dict] = {}
    pre_evidence: Dict[Path_, dict] = {}
    post_evidence: Dict[Path_, dict] = {}
    skipped = _skipped(pre) & _skipped(post)
    for name, doc in pre.items():
        if name in skipped:
            continue
        if isinstance(doc, Unparsed):
            if post.get(name) != doc:
                violations.append(f"{name}: was not valid JSON before repair and was changed; "
                                  "the repair cannot be verified")
            continue
        _walk(doc, (name,), pre_items, pre_evidence)
    for name, doc in post.items():
        if name not in skipped and not isinstance(doc, Unparsed) and not isinstance(pre.get(name), Unparsed):
            _walk(doc, (name,), post_items, post_evidence)
    known_evidence = {entry["id"] for entry in post_evidence.values()}

    unresolved = verbatim = 0
    allowed: set = set()
    for path, before in pre_items.items():
        after = post_items.get(path)
        where = _where(path)
        if removable is not None and _origin(before) == "verbatim" and not _unresolved(before) \
                and removable(before, after):
            verbatim += 1
            allowed.add(path)
            continue
        if _unresolved(before):
            unresolved += 1
            if after is None:
                violations.append(f"{where}: unresolved item was removed or changed")
                continue
            if not _unresolved(after):
                violations.append(f"{where}: moved out of missing/decision "
                                  f"(state {before.get('state')} -> {after.get('state')}, "
                                  f"origin {_origin(before)} -> {_origin(after)})")
            if _unresolved_value(before, before) != _unresolved_value(after, before):
                violations.append(f"{where}: value of an unresolved item was changed")
            if after.get("derivation") not in (None, "", before.get("derivation")):
                violations.append(f"{where}: derivation was added to an unresolved item")
            for key in ("note", "notes"):
                if after.get(key) != before.get(key):
                    violations.append(f"{where}: {key} of an unresolved item was changed")
        if _origin(before) == "verbatim":
            verbatim += 1
            if after is None:
                violations.append(f"{where}: verbatim item was removed or changed")
            elif _origin(after) != "verbatim":
                violations.append(f"{where}: verbatim item was relabelled {_origin(after)}")
            elif _value(before) != _value(after):
                violations.append(f"{where}: verbatim text was changed")

    touched = 0
    for path, after in post_items.items():
        if path in allowed:
            continue
        before = pre_items.get(path)
        where = _where(path)
        origin = _origin(after)
        if before is not None and _value(before) == _value(after) and _origin(before) == origin:
            if origin in ("verbatim", "reported") and _ids(before) and not _ids(after):
                violations.append(f"{where}: evidence ids were removed from a {origin} item")
            continue
        touched += 1
        if _unresolved(after):
            if before is None and not _empty(after.get("value")):
                violations.append(f"{where}: repair added an unresolved item that already carries a value")
            continue
        if origin not in EVIDENCE_LABELS:
            violations.append(f"{where}: repair introduced a resolved value with origin {origin}; "
                              "only evidence-backed values may be added")
            continue
        ids = _ids(after)
        if not ids:
            violations.append(f"{where}: {origin} item added or changed by repair cites no evidence ids")
        elif [i for i in ids if i not in known_evidence]:
            unknown = [i for i in ids if i not in known_evidence]
            violations.append(f"{where}: cites unknown evidence ids {', '.join(unknown)}")

    pre_records: Dict[str, List[dict]] = {}
    for path, entry in pre_evidence.items():
        pre_records.setdefault(entry["id"], []).append(_evidence_record(entry))
    for path, entry in pre_evidence.items():
        after = post_evidence.get(path)
        if after is None and _under(path, allowed):
            continue
        if after is None:
            violations.append(f"{_where(path)}: evidence record was removed")
        elif _evidence_record(after) != _evidence_record(entry):
            changed = [k for k in EVIDENCE_FIELDS if after.get(k) != entry.get(k)]
            violations.append(f"{_where(path)}: evidence record was changed ({', '.join(changed)})")
    for path, entry in post_evidence.items():
        if path not in pre_evidence and _evidence_record(entry) not in pre_records.get(entry["id"], []):
            violations.append(f"{_where(path)}: repair added a new evidence record")

    pre_audit = _audit_entries(pre.get(AUDIT_FILE))
    post_audit = _audit_entries(post.get(AUDIT_FILE))
    for (study, field), entry in sorted(pre_audit.items()):
        if (study, field) not in post_audit:
            violations.append(f"{AUDIT_FILE}: entry for {study or '?'} / {field} was removed")
        elif post_audit[(study, field)] != entry:
            violations.append(f"{AUDIT_FILE}: entry for {study or '?'} / {field} was changed")

    return {
        "passed": not violations,
        "violations": violations,
        "checked": {
            "files": len(pre),
            "unresolvedItems": unresolved,
            "verbatimItems": verbatim,
            "evidenceRecords": len(pre_evidence),
            "auditEntries": len(pre_audit),
            "touchedItems": touched,
        },
    }


class Unparsed(str):
    """Raw text of a JSON file that did not parse."""


def load_package(root: Optional[Path]) -> Dict[str, Any]:
    """Every JSON file under a paper folder, keyed by its relative path."""
    documents: Dict[str, Any] = {}
    if root is None or not root.is_dir():
        return documents
    for path in sorted(root.rglob("*.json")):
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        try:
            documents[path.relative_to(root).as_posix()] = json.loads(text)
        except json.JSONDecodeError:
            documents[path.relative_to(root).as_posix()] = Unparsed(text)
    return documents


def changed_files(pre_root: Path, post_root: Path) -> List[str]:
    """Relative paths of every file added, removed or modified, JSON or not."""
    def files(root: Path) -> Dict[str, bytes]:
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    before, after = files(pre_root), files(post_root)
    return sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))


def package_diff(pre_root: Path, post_root: Path, limit: int = 2_000_000) -> str:
    """Unified diff of every changed text file, for reviewers (logs/repair.diff)."""
    chunks: List[str] = []
    for name in changed_files(pre_root, post_root):
        old, new = pre_root / name, post_root / name
        try:
            a = old.read_text().splitlines(keepends=True) if old.is_file() else []
            b = new.read_text().splitlines(keepends=True) if new.is_file() else []
        except UnicodeDecodeError:
            chunks.append(f"Binary file {name} changed\n")
            continue
        chunks.extend(difflib.unified_diff(a, b, f"pre_repair/{name}", f"package/{name}"))
    text = "".join(chunks)
    return text if len(text) <= limit else text[:limit] + "\n[truncated]\n"


def guard_repair(pre_root: Optional[Path], post_root: Optional[Path], removable: Optional[Removable] = None) -> dict:
    """Compare the paper folder before repair with the one after it."""
    if pre_root is None:
        return {"passed": False, "checked": {}, "changedFiles": [],
                "violations": ["the package before repair did not contain exactly one paper folder"]}
    if post_root is None:
        return {"passed": False, "checked": {}, "changedFiles": [],
                "violations": ["after repair the package does not contain exactly one paper folder"]}
    if post_root.name != pre_root.name:
        return {"passed": False, "checked": {}, "changedFiles": [],
                "violations": [f"repair renamed the paper folder {pre_root.name} to {post_root.name}"]}
    result = compare_packages(load_package(pre_root), load_package(post_root), removable)
    result["changedFiles"] = changed_files(pre_root, post_root)
    return result
