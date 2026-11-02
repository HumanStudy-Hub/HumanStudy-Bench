"""Deterministic fidelity guard for the validator repair loop.

The repair turn is told, in its prompt, never to fill a gap or rewrite a
transcribed instrument. This module checks that it did not, by comparing the
package snapshot taken before repair with the package after repair:

- a field or item that was ``missing`` (label) or in a ``missing``/``decision``
  state stays unresolved, keeps its content, and is not deleted;
- a ``verbatim`` item keeps its label and its text exactly;
- an evidence quote keeps its text exactly;
- an entry of ``audit/missing_information.json`` is not removed;
- every item the repair added or changed that is labelled ``verbatim``,
  ``reported`` or ``derived`` cites at least one evidence id, and every cited id
  resolves to an evidence record in the package.

Items are found structurally in every JSON file of the paper folder: any object
that carries an evidence label (``origin``, ``evidence_label``, ``evidenceLabel``
or ``label`` set to one of the four labels) or a ``state``. Items are matched
between the two snapshots by file and JSON path, using ``id`` for list entries
that have one, so reordering a list does not register as a change.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LABELS = ("verbatim", "reported", "derived", "missing")
LABEL_KEYS = ("origin", "evidence_label", "evidenceLabel", "label")
UNRESOLVED_STATES = ("missing", "decision")
EVIDENCE_ID_KEYS = ("evidenceIds", "evidence_ids")
# Keys that describe an item rather than carry its transcribed or reported
# content. Changing them is not a change of text.
METADATA_KEYS = frozenset(
    ("origin", "evidence_label", "evidenceLabel", "label", "state", "evidenceIds", "evidence_ids",
     "studyIds", "derivation", "note", "notes", "extensions")
)
AUDIT_FILE = "audit/missing_information.json"

Path_ = Tuple[str, ...]


def _label(item: dict) -> Optional[str]:
    for key in LABEL_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value in LABELS:
            return value
    return None


def _content(value: Any) -> Any:
    """The item's text payload: the value with descriptive metadata removed."""
    if isinstance(value, dict):
        return {k: _content(v) for k, v in value.items() if k not in METADATA_KEYS}
    if isinstance(value, list):
        return [_content(v) for v in value]
    return value


def _evidence_ids(item: dict) -> List[str]:
    for key in EVIDENCE_ID_KEYS:
        ids = item.get(key)
        if isinstance(ids, list):
            return [i for i in ids if isinstance(i, str) and i.strip()]
    return []


def _walk(value: Any, path: Path_, out: Dict[Path_, dict], evidence: Dict[Path_, dict]) -> None:
    if isinstance(value, dict):
        if _label(value) is not None or value.get("state") in ("confirmed", "check") + UNRESOLVED_STATES:
            out[path] = value
        for key, child in value.items():
            if key == "evidence" and isinstance(child, list):
                for entry in child:
                    if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                        evidence[path + ("evidence", "#" + entry["id"])] = entry
            _walk(child, path + (key,), out, evidence)
    elif isinstance(value, list):
        seen: Dict[str, int] = {}
        for index, child in enumerate(value):
            ident = child.get("id") if isinstance(child, dict) else None
            if isinstance(ident, str) and ident:
                seen[ident] = seen.get(ident, 0) + 1
                segment = "#" + ident if seen[ident] == 1 else f"#{ident}@{seen[ident]}"
            else:
                segment = f"[{index}]"
            _walk(child, path + (segment,), out, evidence)


def _audit_keys(documents: Dict[str, Any]) -> set:
    keys = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("field"), str):
                keys.add((str(value.get("study", "")), value["field"]))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(documents.get(AUDIT_FILE))
    return keys


def _where(path: Path_) -> str:
    return path[0] + ":" + ".".join(path[1:]) if len(path) > 1 else path[0]


def compare_packages(pre: Dict[str, Any], post: Dict[str, Any]) -> dict:
    """Compare two packages given as ``{relative json path: parsed json}``.

    Pure function: returns ``{"passed", "violations", "checked"}`` and never
    touches the file system.
    """
    pre_items: Dict[Path_, dict] = {}
    post_items: Dict[Path_, dict] = {}
    pre_evidence: Dict[Path_, dict] = {}
    post_evidence: Dict[Path_, dict] = {}
    for name, doc in pre.items():
        _walk(doc, (name,), pre_items, pre_evidence)
    for name, doc in post.items():
        _walk(doc, (name,), post_items, post_evidence)
    known_evidence = {path[-1][1:] for path in post_evidence}

    violations: List[str] = []
    unresolved = verbatim = 0
    for path, before in pre_items.items():
        after = post_items.get(path)
        where = _where(path)
        label = _label(before)
        if label == "missing" or before.get("state") in UNRESOLVED_STATES:
            unresolved += 1
            if after is None:
                violations.append(f"{where}: unresolved item was removed")
                continue
            if label == "missing" and _label(after) != "missing":
                violations.append(f"{where}: label moved out of missing (missing -> {_label(after)})")
            if before.get("state") in UNRESOLVED_STATES and after.get("state") not in UNRESOLVED_STATES:
                violations.append(
                    f"{where}: state moved out of {before.get('state')} (-> {after.get('state')})")
            if _content(before) != _content(after):
                violations.append(f"{where}: content of an unresolved item was changed")
        if label == "verbatim":
            verbatim += 1
            if after is None:
                violations.append(f"{where}: verbatim item was removed")
            elif _label(after) != "verbatim":
                violations.append(f"{where}: verbatim item was relabelled {_label(after)}")
            elif _content(before) != _content(after):
                violations.append(f"{where}: verbatim text was changed")

    for path, before in pre_evidence.items():
        after = post_evidence.get(path)
        if "quote" in before and (after is None or after.get("quote") != before.get("quote")):
            violations.append(f"{_where(path)}: evidence quote was {'removed' if after is None else 'changed'}")

    changed = 0
    for path, after in post_items.items():
        before = pre_items.get(path)
        if before == after:
            continue
        changed += 1
        label = _label(after)
        if label not in ("verbatim", "reported", "derived"):
            continue
        ids = _evidence_ids(after)
        where = _where(path)
        if not ids:
            violations.append(f"{where}: {label} item added or changed by repair cites no evidence ids")
        elif known_evidence:
            unknown = [i for i in ids if i not in known_evidence]
            if unknown:
                violations.append(f"{where}: cites unknown evidence ids {', '.join(unknown)}")

    removed_audit = sorted(_audit_keys(pre) - _audit_keys(post))
    for study, field in removed_audit:
        violations.append(f"{AUDIT_FILE}: entry for {study or '?'} / {field} was removed")

    return {
        "passed": not violations,
        "violations": violations,
        "checked": {
            "files": len(pre),
            "unresolvedItems": unresolved,
            "verbatimItems": verbatim,
            "evidenceQuotes": sum(1 for e in pre_evidence.values() if "quote" in e),
            "changedItems": changed,
        },
    }


def load_package(root: Optional[Path]) -> Dict[str, Any]:
    """Every JSON file under a paper folder, keyed by its relative path.
    A file that does not parse is kept as its raw text so a change still shows."""
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
            documents[path.relative_to(root).as_posix()] = text
    return documents


def guard_repair(pre_root: Optional[Path], post_root: Optional[Path]) -> dict:
    """Compare the paper folder before repair with the one after it."""
    if pre_root is None:
        return {"passed": False, "violations": ["no pre-repair snapshot to compare against"], "checked": {}}
    if post_root is None or post_root.name != pre_root.name:
        return {"passed": False, "violations": ["repair removed or renamed the paper folder"], "checked": {}}
    return compare_packages(load_package(pre_root), load_package(post_root))
