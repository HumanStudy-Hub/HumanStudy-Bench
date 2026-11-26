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
import copy
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

from hsenv import distribution, release, schema_check

from scripts.research import approx_text
from scripts.research import protocols
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
READINESS_REASON = {
    "exact": "Every held text is restored on the researcher's computer from their copy of the source, by its hash.",
    "approx": ("Some held passages are restored from the researcher's copy approximately (a scanned or corrected "
               "text) and labelled so."),
    "needs_text": ("Some held texts cannot be restored by the app (a picture, a file the researcher must supply, or "
                   "text that cannot be located): the researcher pastes or captures them before running."),
    "reconstruct": ("The paper does not print every material: no arm runs as published; each arm states its gap and "
                    "the researcher reconstructs it, labelled reconstructed, before running."),
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
# Wording of a licence hold that defers to the owner, who decided on 2026-11-24 to release every library paper.
OWNER_RELEASED = re.compile(r"release is an owner decision|owner decides(?=\s*[;.(\"]|\s*$)|"
                            r"owner decision(?=\s*[;.(\"]|\s*$)|"
                            r"private research (?:use |runs )?only", re.I)
UNLOCATED = re.compile(r"held text\(s\) cannot be located|PDF locators must be written by the extractor")
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
    workspace = Path(os.environ["RELEASE_WORKSPACE"]) if os.environ.get("RELEASE_WORKSPACE") else (
        study_dir.parents[3] if len(study_dir.parents) > 3 else study_dir)
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
        if dist.get("tier") == "hosted" and isinstance(dist.get("sha256"), str) and \
                release.hosted_licence_ok(dist.get("licence")):
            sha = dist["sha256"]   # a hosted source is judged by the exact file that is hosted
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
        declared_fmt = str(dist.get("format") or "").lower()
        if raw is not None and _usable(raw) and format_agrees(declared_fmt, raw):
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


# Owner 2026-11-24: a label of at most five words (a scale anchor, a button, "Set 1", "Strongly agree") is not
# protectable expression; it ships as written instead of being held.
SHORT_WORDS = 5
# Fields of meta the engine and the app read. Everything else in meta (review notes, conflicts, deviations, search
# logs, readings) is private build material: dropped before projection, never published.
META_KEEP = frozenset(("citation", "title", "sources", "language", "languages", "language_cite", "licence", "doi",
                       "sample", "image_read", "designs", "reviewable", "reviewable_on", "official",
                       "participant_language", "foreign_text_allowed", "pages", "assignment", "n_paper"))
SOURCE_KEEP = frozenset(("what", "title", "authors", "year", "doi", "url", "file", "member", "zip_member", "sha256",
                         "bytes", "licence", "distribution", "kind", "format", "pages", "path", "osf_file"))
# Projection findings this release resolves itself: unlocatable or picture-only held text ships as a placeholder the
# researcher fills (tier "needs_text"); the rest are recomputed below with the short-label rule.
RESOLVED = re.compile(r"^(?:\d+ held text\(s\) cannot be located|PDF locators must be written by the extractor|"
                      r"held text inside shipped strings|\d+ on-screen string\(s\) \(data / stages / roles / flow\) "
                      r"are source text without a cite|\d+ source text\(s\) inside expressions / computed positions|"
                      r"\d+ dict key\(s\) / structural value\(s\) are source text|cites name sources missing|"
                      r"\d+ held text\(s\) are not NFC)")
KEY_QUOTES = re.compile(r"^(\d+) dict key\(s\) / structural value\(s\) are source text \(cannot be withheld\): (.*)$",
                        re.S)


def _value(doc, path):
    try:
        return _pointer(doc, path)
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def nwords(text) -> int:
    return len(approx_text.words(approx_text.clean(text))) if isinstance(text, str) else 10 ** 6


def _walk_strings(o, fn):
    if isinstance(o, dict):
        for k in list(o):
            if isinstance(o[k], str):
                o[k] = fn(o[k])
            else:
                _walk_strings(o[k], fn)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            if isinstance(v, str):
                o[i] = fn(v)
            else:
                _walk_strings(v, fn)


def _cited_sources(o, out: set) -> set:
    if isinstance(o, dict):
        if isinstance(o.get("source"), str) and "label" in o:
            out.add(o["source"])
        for v in o.values():
            _cited_sources(v, out)
    elif isinstance(o, list):
        for v in o:
            _cited_sources(v, out)
    return out


# a path on a maintainer's machine (it names the account): only its file name may ship
LOCAL_PATH = re.compile(r"(?:~|/Users/[^/\s\"']+|/home/[^/\s\"']+|/private/(?:tmp|var)|/var/folders|/tmp)"
                        r"(?:/[^\s\"',;()]+)+")
# operator records of the build and its reviews (who decided what, probe files, tuning history): not the study
CODING_NOTES = frozenset(("decision", "tuning_probes", "heldout_probes", "bias_note", "known_weak_spots"))


def _drop_operator_notes(protocol: dict) -> None:
    for item in protocol.get("run_excluded") or []:
        if isinstance(item, dict) and isinstance(item.get("decision"), str):
            item.pop("decision")

    def coding(o):
        if isinstance(o, dict):
            for k in list(o):
                if k in CODING_NOTES and isinstance(o[k], str):
                    o.pop(k)
                else:
                    coding(o[k])
        elif isinstance(o, list):
            for v in o:
                coding(v)
    coding(protocol.get("coding"))


def local_paths(files: dict) -> list[str]:
    """Shipped files that still name a path on a maintainer's machine."""
    return sorted(name for name, raw in files.items()
                  if re.search(rb"/Users/[^/\s\"']+/|/home/[^/\s\"']+/|/private/tmp/|/var/folders/", raw))


# Maintainer prose (cite notes, reasons, licence and clearance records) names internal review files and process
# roles that a reader of the release cannot open; those references are reworded, never the participant wording.
PROSE_KEYS = frozenset(("note", "why", "what", "licence", "license", "clearance", "rule", "description", "image_read",
                        "reason", "notes"))
_PROSE_SUBS = (
    (re.compile(r"\s*[;,]?\s*\bLESSONS\s+\d+(?:\s*[,/]\s*\d+)*"), ""),
    (re.compile(r"\b(?:review|reviews)/(?:[\w.#-]+/)*[\w.#-]+\.\w+"), "the maintainers' review record"),
    (re.compile(r"\blessons-pending(?:\.md)?\b"), "the maintainers' review record"),
    (re.compile(r"\bsourcetext/(?:[\w.#-]+/)*([\w.#-]+)"), r"\1"),
    (re.compile(r"\bcoordinators\b"), "maintainers"), (re.compile(r"\bCoordinators\b"), "Maintainers"),
    (re.compile(r"\bcoordinator\b"), "maintainer"), (re.compile(r"\bCoordinator\b"), "Maintainer"),
    (re.compile(r"\(\s*[;,]\s*"), "("), (re.compile(r"\s*[;,]\s*\)"), ")"), (re.compile(r"\s*\(\s*\)"), ""),
    (re.compile(r"[ \t]{2,}"), " "),
)


def scrub_prose(text: str) -> str:
    for pat, rep in _PROSE_SUBS:
        text = pat.sub(rep, text)
    return text.strip() if text.strip() else text


def _scrub_prose_fields(o) -> None:
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, str) and k in PROSE_KEYS:
                o[k] = scrub_prose(v)
            elif isinstance(v, list) and k in PROSE_KEYS and all(isinstance(x, str) for x in v):
                o[k] = [scrub_prose(x) for x in v]
            else:
                _scrub_prose_fields(v)
    elif isinstance(o, list):
        for v in o:
            _scrub_prose_fields(v)


# A reason the projection withheld because it quoted held text reads as an opaque token; a reader gets a plain sentence.
_WITHHELD = re.compile(r"withheld:[0-9a-f]+")
READABLE = {
    ("run_excluded", "why"): "The reason quotes source wording that is held back; this arm needs the materials listed "
                             "as missing.",
    ("omitted", "what"): "A part of the procedure the paper describes (its description quotes held-back wording)",
    ("omitted", "why"): "The reason quotes source wording that is held back.",
    ("unsupported", "why"): "The reason quotes source wording that is held back.",
}


def readable_reasons(protocol: dict) -> int:
    rows = [("run_excluded", e) for e in protocol.get("run_excluded") or []]
    rows += [("omitted", e) for e in protocol.get("omitted") or []]
    rows += [("unsupported", e) for e in (protocol.get("analysis") or {}).get("unsupported") or []]
    n = 0
    for where, e in rows:
        for key in ("what", "why"):
            text = READABLE.get((where, key))
            if text and isinstance(e, dict) and isinstance(e.get(key), str) and _WITHHELD.fullmatch(e[key]):
                e[key] = text
                n += 1
    return n


def slim(protocol: dict) -> dict:
    """The private protocol as it may be projected: operator-only meta and top-level notes dropped, every string NFC,
    and every cited source declared (an undeclared one becomes a tier "user" source the researcher supplies)."""
    meta = protocol.get("meta") or {}
    protocol["meta"] = {k: v for k, v in meta.items() if k in META_KEEP}
    srcs = protocol["meta"].setdefault("sources", {})
    for k, v in list(srcs.items()):
        if isinstance(v, dict):
            srcs[k] = {kk: vv for kk, vv in v.items() if kk in SOURCE_KEEP}
        elif k == "root":   # the build folder, not a source
            srcs.pop(k)
        else:   # a source declared by a string: a researcher-supplied source (its bytes may still be scanned)
            srcs[k] = {"what": str(v)[:300], "distribution": {"tier": "user"}}
    for k in list(protocol):
        if release._NOTE_KEY.search(k) or k in ("deviations", "alternatives"):
            protocol.pop(k)
    _drop_operator_notes(protocol)
    _scrub_prose_fields(protocol)
    _walk_strings(protocol, lambda t: LOCAL_PATH.sub(lambda m: m.group(0).rstrip("/").rsplit("/", 1)[-1],
                                                     unicodedata.normalize("NFC", t)))
    for key in sorted(_cited_sources({k: v for k, v in protocol.items() if k != "meta"}, set()) - set(srcs)):
        srcs[key] = {"what": key, "distribution": {"tier": "user"}}
    return protocol


# Pieces of one running text (the lines of a picture, the spans of a paragraph): each may be short while together
# they are a passage, so they stay held.
FRAGMENT_KEYS = frozenset(("lines", "spans", "segments", "runs", "words", "tokens", "fragments", "chunks",
                           "paragraphs", "sentences", "passage", "body", "story", "vignette"))
# scripts written without spaces between words: every character counts as a word (CJK incl. supplementary Han and
# half-width kana, Thai, Lao, Khmer, Myanmar, Tibetan)
_UNSPACED = re.compile("[\u0e00-\u0eff\u0f00-\u0fff\u1000-\u109f\u1780-\u17ff\u3040-\u30ff\u3400-\u4dbf"
                       "\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff65-\uff9f\U00020000-\U0003134f]")
_SENTENCE_END = re.compile(r"[.!?\u3002\uff01\uff1f]['\"\u201d\u2019)]*\s*$")
# keys under which short texts are separate choices / labels (a scale, a table header), not pieces of prose
LABEL_KEYS = frozenset(("options", "option", "anchors", "anchor", "choices", "choice", "labels", "label", "scale",
                        "columns", "column", "rows", "row", "headers", "header", "buttons", "button", "left", "right",
                        "levels", "items_labels", "cells", "legend", "units", "unit", "equals", "validate"))


NUMERIC = re.compile(r"[0-9\s.,:;/+\-\u2212\u2013%()$\u20ac\u00a3#]*")


def nwords_all(text) -> int:
    if not isinstance(text, str):
        return 10 ** 6
    return nwords(_UNSPACED.sub(" ", text)) + len(_UNSPACED.findall(text))


def is_label(text) -> bool:
    """A short label the owner's rule ships: at most SHORT_WORDS words (a character of an unspaced script counts as
    a word), and not a sentence of three or more words."""
    if nwords_all(text) > SHORT_WORDS:
        return False
    return not (nwords_all(text) >= 3 and _SENTENCE_END.search(text.strip()))


def _sequence_key(path: str):
    """Held texts anywhere inside sibling elements of one list (content blocks, list items; any field) share the
    list as their key."""
    parts = path.split("/")
    for i in range(len(parts) - 1, 0, -1):
        if parts[i].isdigit():
            return "/".join(parts[:i])
    return None


def unhold_short(projected: dict, private: dict, unheld: list | None = None) -> int:
    """Put every held text that is a label (is_label) back as written; returns how many. Not put back: pieces of a
    running text (FRAGMENT_KEYS), and short texts in sibling elements of one list unless they sit under a choice /
    label key: adjacent short blocks (even single words) may be one sentence."""
    held = (projected.get("distribution") or {}).get("held") or {}
    unheld = [] if unheld is None else unheld
    texts = {hid: _value(private, e["path"]) for hid, e in held.items() if e.get("kind") != "image" and e.get("path")}
    groups: dict[str, list] = {}
    for hid, e in held.items():
        if hid in texts:
            # siblings in one list (any field) and siblings in one object (a prefix and a suffix) are one sequence
            for key in (_sequence_key(e["path"]), "obj:" + e["path"].rsplit("/", 1)[0]):
                if key:
                    groups.setdefault(key, []).append(hid)
    n = 0
    for hid, entry in list(held.items()):
        if hid not in texts or FRAGMENT_KEYS & set(entry["path"].split("/")):
            continue
        text = texts[hid]
        if not is_label(text):
            continue
        seq = sorted(set(groups.get(_sequence_key(entry["path"]) or "", [hid])
                         + groups.get("obj:" + entry["path"].rsplit("/", 1)[0], [hid])))
        if len(seq) > 1 and not (LABEL_KEYS & set(entry["path"].split("/"))) \
                and not all(isinstance(texts[h], str) and NUMERIC.fullmatch(texts[h].strip()) for h in seq):
            continue   # (a list of numbers / codes without letters is never prose)
        parent, key = entry["path"].rsplit("/", 1)
        cont = _pointer(projected, parent)
        key = key.replace("~1", "/").replace("~0", "~")
        cont[int(key) if isinstance(cont, list) else key] = text
        unheld.append((text, entry.get("source")))
        del held[hid]
        n += 1
    return n


def _case_lists(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            sub = f"{path}/{k.replace('~', '~0').replace('/', '~1')}"
            if k == "cases" and isinstance(v, list):
                yield sub, v
            yield from _case_lists(v, sub)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _case_lists(v, f"{path}/{i}")


def drop_held_cases(projected: dict) -> int:
    """Coding cases (worked examples that test a coding rule) whose text is held or withheld cannot be coded in the
    public copy; they are test fixtures, not material, so they are left out of it (with their held entries).
    Returns how many."""
    held = (projected.get("distribution") or {}).get("held") or {}
    placeholder = (projected.get("distribution") or {}).get("placeholder") or "\u2591"
    n = 0
    lists = list(_case_lists({k: v for k, v in projected.items() if k != "distribution"}))
    for lpath, cases in sorted(lists, key=lambda x: -x[0].count("/")):   # inner lists first: outer drops shift them
        drop = {i for i, c in enumerate(cases) if isinstance(c, dict) and isinstance(c.get("text"), str)
                and (release._is_withheld(c["text"]) or (c["text"] and set(c["text"]) == {placeholder}))}
        if not drop:
            continue
        keep = [i for i in range(len(cases)) if i not in drop]
        new_index = {old: new for new, old in enumerate(keep)}
        cases[:] = [cases[i] for i in keep]
        n += len(drop)
        for hid in list(held):
            path = str(held[hid].get("path") or "")
            if path.startswith(lpath + "/"):
                rest = path[len(lpath) + 1:].split("/", 1)
                old = int(rest[0])
                if old in drop:
                    del held[hid]
                else:
                    held[hid]["path"] = f"{lpath}/{new_index[old]}" + (f"/{rest[1]}" if len(rest) > 1 else "")
    return n


QSF_SHOWN = frozenset(("QuestionText", "Display"))   # what a respondent sees (incl. translations); not descriptions


class _Visible(__import__("html.parser").parser.HTMLParser):
    """The text a browser shows: nothing inside script/style/template/... or any hidden element (nested too)."""
    HIDE = {"script", "style", "template", "noscript", "textarea", "select", "object", "iframe", "svg", "head", "title"}
    VOID = {"br", "img", "input", "hr", "meta", "link", "area", "base", "col", "embed", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.out = [], []

    def handle_starttag(self, tag, attrs):
        if tag in self.VOID:
            return
        a = " ".join(f"{k}={v}" for k, v in attrs).lower()
        hidden = tag in self.HIDE or "hidden" in a or re.search(r"display\s*:\s*none|visibility\s*:\s*hidden", a)
        self.stack.append(bool(hidden) or (self.stack[-1] if self.stack else False))

    def handle_endtag(self, tag):
        if tag not in self.VOID and self.stack:
            self.stack.pop()

    def handle_data(self, data):
        if not (self.stack and self.stack[-1]):
            self.out.append(data)

    def handle_comment(self, data):
        pass


def visible_text(fragment: str) -> str:
    parser = _Visible()
    try:
        parser.feed(fragment)
        parser.close()
    except Exception:
        return ""
    return " ".join(parser.out)


def _all_strings(o, shown=False):
    """Strings of a survey file that respondents see: QuestionText and Display values, in every language."""
    if isinstance(o, str):
        if shown:
            yield o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from _all_strings(v, k in QSF_SHOWN)
    elif isinstance(o, list):
        for v in o:
            yield from _all_strings(v, shown)


def _canon(text: str) -> str:
    """The engine's canonical text with invisible characters removed (a PDF reader can put U+FFFE inside a word)."""
    return distribution.canon_only(approx_text._INVISIBLE.sub("", text))


def hosted_unconfirmed(message: str, private: dict, sources: dict) -> list[str]:
    """Of a 'cited to a hosted source but not in the hosted file' finding, the paths that may not ship: anything but
    a label or a text whose own cite (the engine's slot cite) says it was read from a picture of that source (a
    picture's words are not in a file's text layer). The engine lists at most ten paths; when it lists fewer than
    it counts, every slot cited only to hosted sources is checked here against the hosted file's text."""
    head, _, rest = message.partition(": ")
    listed = re.split(r", (?=/)", rest) if rest else []
    msrc = (private.get("meta") or {}).get("sources") or {}

    def hosted(key):
        d = (msrc.get(key) or {}).get("distribution") if isinstance(msrc.get(key), dict) else None
        return isinstance(d, dict) and d.get("tier") == "hosted"
    slots = {}
    for path, _cont, _key, cf, kind in release._slots(copy.deepcopy(private)):
        cites = [c for c in release._cites(cf) if c.get("label") in ("verbatim", "reported")]
        if kind != "image" and cites and all(hosted(c.get("source")) for c in cites):
            slots[release._ptr(path)] = cites
    truncated = not listed or not head.split(" ", 1)[0].isdigit() or int(head.split(" ", 1)[0]) != len(listed)
    candidates = list(slots) if truncated else listed
    texts: dict[str, list] = {}
    bad = []
    for path in candidates:
        cites = slots.get(path, [])
        value = _value(private, path)
        if is_label(value) or (cites and all(c.get("read") == "image" for c in cites)):
            continue
        if isinstance(value, str):
            found = False
            for c in cites:
                key = c.get("source")
                want = str(((msrc.get(key) or {}).get("distribution") or {}).get("sha256") or "").lower()
                if key in sources and key not in texts and (
                        not want or hashlib.sha256(sources[key]).hexdigest() != want):
                    texts[key] = []   # only the exact hosted file (its distribution sha256) can confirm a text
                elif key in sources and key not in texts:
                    fmt = ((msrc.get(key) or {}).get("distribution") or {}).get("format") or distribution.sniff(sources[key])
                    try:
                        units = [] if fmt == "qsf" else distribution.units(
                            "text" if fmt in TEXT_FORMATS else fmt, sources[key], {"pdf": distribution.pdfium_extractor()})
                        if fmt == "qsf":   # what respondents see, translations included (hidden elements removed)
                            units += [visible_text(x)
                                      for x in _all_strings(json.loads(sources[key].decode("utf-8-sig")))]
                        texts[key] = [_canon(u) for u in units]   # field by field, never joined
                    except Exception:
                        texts[key] = []
                if any(_canon(value) in u for u in texts.get(key, [])):
                    found = True
            if found:
                continue
        bad.append(path)
    return bad


def _key_text(item: str):
    return item[:-len(" (key)")].rsplit("/", 1)[1].replace("~1", "/").replace("~0", "~")


def blocking_errors(report: dict, projected: dict, private: dict, unheld=(), sources=None) -> list[str]:
    """Projection findings that would publish source text, after the short-label rule. `unheld`: the (text, source)
    of every short label put back; `sources`: the source bytes the projection had."""
    sources = sources or {}
    cite_paths = release._cite_paths(private)
    out = [e for e in report.get("errors") or [] if not HOSTED_UNLOCATED.search(e) and not RESOLVED.search(e)]
    for e in report.get("errors") or []:
        if HOSTED_UNLOCATED.search(e):
            bad = hosted_unconfirmed(e, private, sources)
            if bad:
                out.append("text(s) cited to a hosted source but not in its file, neither a label nor read from a "
                           "picture of it: " + ", ".join(bad[:10]))
    texts = []
    for e in ((projected.get("distribution") or {}).get("held") or {}).values():
        if e.get("kind") != "image" and e.get("path"):
            texts.append(_value(private, e["path"]))
    _, leaks, _ = release._scan(projected, [t for t in texts if isinstance(t, str)], cite_paths)
    # A short label put back whose source the projection could not read: a longer screen string containing it may
    # be more of that unread source, so it still counts as a leak (the projection's own evidence is kept).
    unread = [t for t, _ in unheld]   # the projection flagged strings containing them: keep that evidence
    if unread:
        _, more, _ = release._scan(copy.deepcopy(projected), unread, cite_paths)
        leaks += [x for x in more if x not in leaks]
    # a shipped string that is itself a short label reveals only those words
    leaks = [x for x in leaks if not is_label(_key_text(x) if x.endswith(" (key)") else _value(projected, x))]
    if leaks:
        out.append("held text inside shipped strings that cannot be withheld: " + ", ".join(leaks[:10]))
    long_uncited = [u["path"] for u in report.get("uncited_source_text") or []
                    if not is_label(_value(private, u["path"]))]
    if long_uncited:
        out.append(f"{len(long_uncited)} on-screen string(s) of more than {SHORT_WORDS} words are source text without "
                   "a cite: " + ", ".join(long_uncited[:10]))
    # an unlocatable text the projection did not replace (a structural field) still ships as written
    held_paths = {e.get("path") for e in ((projected.get("distribution") or {}).get("held") or {}).values()}
    placeholder = (projected.get("distribution") or {}).get("placeholder") or "\u2591"
    shipped = [u["path"] for u in report.get("unlocated") or [] if u["path"] not in held_paths
               and isinstance(_value(projected, u["path"]), str) and set(_value(projected, u["path"])) != {placeholder}
               and not release._is_withheld(_value(projected, u["path"])) and not is_label(_value(projected, u["path"]))]
    if shipped:
        out.append(f"{len(shipped)} unlocatable source text(s) ship as written: " + ", ".join(shipped[:10]))
    lits: dict[str, list] = {}
    for lp, t in release._literals(private, cite_paths):
        lits.setdefault(lp if isinstance(lp, str) else release._ptr(list(lp)), []).append(t)
    long_lits = [lp for lp in report.get("literals_in_source") or []
                 if not lits.get(lp) or any(not is_label(t) for t in lits[lp])]
    if long_lits:
        out.append(f"{len(long_lits)} source text(s) of more than {SHORT_WORDS} words inside expressions / computed "
                   "positions (move the text into data with a cite): " + ", ".join(long_lits[:10]))
    for e in report.get("errors") or []:
        m = KEY_QUOTES.match(e)
        if not m:
            if re.match(r"^\d+ dict key\(s\) / structural value", e):
                out.append(e[:300])   # a key finding this wrapper cannot read: never waived
            continue
        listed = re.split(r", (?=/)", m.group(2))
        bad = []
        for item in listed:
            if item.endswith(" (structural value)"):
                text = _value(private, item[:-len(" (structural value)")])
            else:
                text = item.rsplit("/", 1)[-1].replace("~1", "/").replace("~0", "~")
            if not is_label(text):
                bad.append(item)
        if bad or int(m.group(1)) != len(listed):
            out.append(f"{m.group(1)} dict key(s) / structural value(s) of more than {SHORT_WORDS} words are source "
                       "text (cannot be withheld): " + ", ".join(bad[:10] or listed[:10]))
    return out


def format_agrees(declared: str, raw: bytes) -> bool:
    """The bytes are the declared kind of file (a hash prefix can match the wrong file, e.g. a .docx for an image)."""
    if not declared:
        return True
    kind = distribution.sniff(raw)
    if declared in ("pdf", "docx", "qsf", "html"):
        return kind == declared
    if declared in TEXT_FORMATS or declared == "text":
        return kind in ("text", "html", "qsf")
    return False   # an image, an archive or another binary format is never read as text


def project(study_dir: Path) -> tuple[dict, dict]:
    protocol = json.loads((study_dir / "protocol.json").read_text())
    declared = copy.deepcopy(protocol)   # the full declarations: URLs and file paths may be written in notes
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
    # after the final tiers: a hosted source is resolved to its hosted file, any other to the private copy
    sources = resolve_sources(study_dir, protocol)
    protocol = slim(protocol)
    projected, report = release.project(protocol, sources=sources,
                                        extractors={"pdf": distribution.pdfium_extractor()}, strict=False)
    unheld: list = []
    short = unhold_short(projected, protocol, unheld)
    blocking = blocking_errors(report, projected, protocol, unheld, sources)
    if blocking:
        raise ProjectionBlocked("; ".join(e[:200] for e in blocking[:4]))
    added, _ = add_approx(projected, protocol, sources)
    held_now = projected.setdefault("distribution", {}).setdefault("held", {})
    # Held items the app cannot fill from the researcher's file by itself: unlocatable text, text read from a picture
    # without an approx fingerprint, pictures, and text against a source the app cannot fetch. They ship as
    # placeholders; the app asks the researcher to paste or capture them (tier "needs_text"). Nothing of them ships.
    unlocated = {u["path"] for u in report.get("unlocated") or []}
    needs = {h for h, e in held_now.items() if "approx" not in e and (
        e.get("path") in unlocated or e.get("kind") == "image"
        or (e.get("locator") or {}).get("format") in ("image_reading", "region", "image"))}
    needs |= unfetchable(projected, declared)
    # A held picture keeps no private file path: its src becomes asset:<sha256 prefix> of the picture (the id the engine
    # gives the file), which the researcher's own copy fills in.
    for e in held_now.values():
        if e.get("kind") != "image" or not e.get("path"):
            continue
        cont = _value(projected, e["path"])
        holder, key = (cont, "src") if isinstance(cont, dict) else (_value(projected, e["path"].rsplit("/", 1)[0]),
                                                                      e["path"].rsplit("/", 1)[1])
        if isinstance(holder, dict) and isinstance(holder.get(key), str):
            e["sha256"] = picture_sha(study_dir, holder[key]) or e.get("sha256")   # the bytes, when they are here
            if not holder[key].startswith("asset:") and e.get("sha256"):
                holder[key] = "asset:" + str(e["sha256"])[:16]
    # A picture with a private file path that is not held (an openly licensed figure) cannot ship as a path: it is
    # held like the others (the researcher's copy of the source fills it in by its hash).
    held_srcs = {e.get("path") for e in held_now.values() if e.get("kind") == "image"}
    for path, holder in _local_pictures(projected):
        if path in held_srcs or path.rsplit("/", 1)[0] in held_srcs:
            continue
        if path.startswith("/meta/"):   # an operator's picture (a design note): not needed to run, never shipped
            holder.pop("src")
            continue
        digest = picture_sha(study_dir, holder["src"]) or holder.get("sha256")
        if not digest:   # the file is not here: nothing can verify it, so it is not shipped at all
            holder.pop("src")
            continue
        cite = (release._cites(_own_cite(holder)) or [{}])[0]
        if not isinstance(cite.get("source"), str):
            cite = {"source": "materials"}
            projected["meta"].setdefault("sources", {}).setdefault("materials", {
                "what": "the researcher's own copy of the study materials", "distribution": {"tier": "user"}})
        n_ = len(held_now) + 1
        while f"h{n_}" in held_now:
            n_ += 1
        hid = f"h{n_}"
        held_now[hid] = {"path": path.rsplit("/", 1)[0], "source": cite.get("source"), "kind": "image",
                         "sha256": digest, "locator": {"format": "image", "page": cite.get("page")}}
        holder["src"] = "asset:" + digest[:16]
        needs.add(hid)
    # after every lookup by path into the private protocol (dropping shifts the paths of later cases)
    drop_held_cases(projected)
    needs &= set(held_now)
    # Notes are dropped (slim); a source the engine could not scan for quotes is recorded, not a blocker.
    unscanned = sorted(report.get("quote_scan_unchecked") or [])
    tiers = report.get("tiers") or {}
    for key, raw in sources.items():
        if tiers.get(key) != "hosted" and raw[:5] == b"%PDF-" and key not in unscanned:
            if len("".join(distribution.pdfium_extractor()(raw)).strip()) < 2000:
                unscanned.append(key)   # a scan without a text layer: the quote scan saw nothing
    if unscanned:
        loose = uncited_screen_text(protocol) + derived_unscanned(report, protocol, set(unscanned))
        if loose:
            raise ProjectionBlocked(f"source(s) {', '.join(unscanned[:6])} could not be scanned (no text or file), so "
                                    f"{len(loose)} uncited on-screen text(s) of more than {SHORT_WORDS} words cannot be "
                                    "checked against them: " + ", ".join(loose[:8]))
    # The engine reads one protocol format, schema_version "1"; builders wrote their own convention labels
    # ("hsenv-protocol/3.2", 3, 1). The schema check below decides whether the file is that format.
    projected["schema_version"] = "1"
    # An arm hold is cleared only when it is a licence hold, which the projection resolves by holding the text;
    # any other hold (materials missing, a decision pending) keeps the arm out of the runnable set.
    holds = {}
    for arm in projected.get("arms") or []:
        why = arm.pop("release_hold", None)
        if not why:
            continue
        reason = OWNER_RELEASED.sub(" ", json.dumps(why, ensure_ascii=False))
        if LICENCE_HOLD.search(reason) and not OTHER_HOLD.search(reason):
            holds[arm["id"]] = why
            # the builders also set runnable false for a licence hold; the projection resolved it
            # (only an arm with its own cite and no other recorded reason: not in run_excluded, no `why`)
            excluded = json.dumps(protocol.get("run_excluded") or [], ensure_ascii=False)
            if arm.get("runnable") is False and not arm.get("why") and arm.get("cite") \
                    and f'"{arm.get("id")}"' not in excluded:
                arm.pop("runnable")
        else:
            arm["runnable"] = False
            arm.setdefault("why", why if isinstance(why, str) else json.dumps(why, ensure_ascii=False))
    # Owner 2026-11-24: a paper that does not print every material still ships. Its arms stay not runnable as
    # published (the engine refuses to serve an item labelled missing) with the gap stated (`why`); the researcher and
    # the assistant reconstruct the missing material in their own copy, labelled reconstructed (tier "reconstruct").
    reconstruct = []
    if not any(a.get("runnable", True) is not False for a in projected.get("arms") or []):
        for arm in projected.get("arms") or []:
            arm.setdefault("why", "materials not printed in the paper: reconstruct them before running")
            reconstruct.append(arm["id"])
    readiness = "reconstruct" if reconstruct else "needs_text" if needs else "approx" if added else "exact"
    return projected, {"held": len(held_now), "holdsResolved": holds, "approx": added, "short": short,
                       "needsText": len(needs), "unscanned": unscanned, "reconstruct": reconstruct,
                       "readiness": readiness}


def _pointer(doc, path: str):
    cur = doc
    for part in path.split("/")[1:] if path else []:
        part = part.replace("~1", "/").replace("~0", "~")
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def add_approx(projected: dict, private: dict, sources: dict) -> tuple[int, int]:
    """Give every held PDF text that could not be located exactly an approx fingerprint (approx_text.py), when our
    own copy of the source proves it can be placed: (added, still unplaceable). The app then restores such a passage
    from the researcher's copy and reports it as approximate; the exact path is unchanged for every other item."""
    held = (projected.get("distribution") or {}).get("held") or {}
    pages_of: dict[str, list[str]] = {}
    added = missing = 0
    for entry in held.values():
        loc = entry.get("locator") or {}
        if entry.get("kind") == "image" or loc.get("parts"):
            continue
        src = entry.get("source")
        raw = sources.get(src)
        if raw is None or raw[:5] != b"%PDF-":
            missing += 1
            continue
        if src not in pages_of:
            pages_of[src] = distribution.pdfium_extractor()(raw)
        try:
            text = _pointer(private, entry["path"])
            parent = _pointer(private, entry["path"].rsplit("/", 1)[0])
        except (KeyError, IndexError, ValueError, TypeError):
            missing += 1
            continue
        cite = parent.get("cite") if isinstance(parent, dict) else None
        pages = [loc.get("page")] if isinstance(loc.get("page"), int) else list(loc.get("pages") or [])
        if not pages and isinstance(cite, dict) and isinstance(cite.get("page"), int):
            pages = [cite["page"]]
        fp = approx_text.fingerprint(text, pages) if isinstance(text, str) and pages else None
        hit = approx_text.locate(pages_of[src], fp) if fp else None
        if hit is None:
            missing += 1
            continue
        entry["approx"] = fp
        added += 1
    return added, missing


def picture_sha(study_dir: Path, src: str) -> str | None:
    """sha256 of a picture's bytes from its local path in the study folder (None when it is not there)."""
    rel = Path(os.path.normpath(src.lstrip("/")))
    if rel.is_absolute() or rel.parts[:1] == ("..",):
        return None
    path = study_dir / rel   # a study's sources/ may be a link to the private checkout: followed, never escaped by ..
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


# parts of data / stages that participants never see: coding rules (patterns), cite evidence, presentation notes
NOT_SHOWN = frozenset(("coding", "presentation", "evidence", "patterns", "pattern", "regex", "dcite", "cites",
                       "keyed", "notes", "note", "widget_params"))


def uncited_screen_text(private: dict) -> list[str]:
    """Participant-facing strings (under data / stages / roles / flow) of more than SHORT_WORDS words with no cite on
    their own object or any object above them, outside structural fields and computed positions: our own wording,
    or a quote nobody cited."""
    cite_paths = release._cite_paths(private)
    out: list[str] = []

    def cited(o):   # an object with its own cite (covers all of it), or a cite itself (label + source)
        return isinstance(o, dict) and (bool(o.get("cite") or o.get("cites")) or ("label" in o and "source" in o))

    def walk(o, path, covered):
        if isinstance(o, dict):
            covered = covered or cited(o)
            items = list(o.items())
        elif isinstance(o, list):
            items = list(enumerate(o))
        else:
            return
        if NOT_SHOWN & {str(x) for x in path}:
            return
        for k, v in items:
            if isinstance(v, str):
                own = isinstance(k, str) and isinstance(o, dict) and bool(o.get(k + "_cite") or o.get(k + "_cites"))
                if covered or own or (isinstance(k, str) and release._NOTE_KEY.search(k)) or (isinstance(k, str) and (k in release._STRUCT or k.endswith("_cite") or k in ("cite",
                        "cites", "note", "notes"))) or release._computed(path, k, cite_paths) or release._is_withheld(v) \
                        or release._TOKEN.fullmatch(v.strip()) or is_label(v):
                    continue
                out.append(release._ptr(path + [k]))
            else:
                child = covered or (isinstance(k, str) and isinstance(o, dict)
                                    and bool(o.get(k + "_cite") or o.get(k + "_cites")))
                walk(v, path + [k], child)
    for top in ("data", "stages", "roles", "flow"):
        walk(private.get(top), [top], False)
    return out


def derived_unscanned(report: dict, private: dict, unscanned) -> list[str]:
    """Derived texts the engine shipped as our own wording while citing a source it could not scan."""
    out = []
    for path in report.get("derived_shipped") or []:
        parts = path.split("/")
        key = parts[-1]
        cites = []
        for depth in range(len(parts) - 1, 0, -1):   # the text's own cite and every cite above it
            cont = _value(private, "/".join(parts[:depth]))
            if isinstance(cont, dict):
                for k in ("cite", "cites") + ((key + "_cite", key + "_cites") if depth == len(parts) - 1 else ()):
                    if cont.get(k):
                        cites += release._cites(cont[k])
        if any(c.get("source") in unscanned for c in cites) and not is_label(_value(private, path)):
            out.append(path)
    return out


def _own_cite(holder: dict):
    return holder.get("cite") or holder.get("cites")


def _local_pictures(o, path=""):
    """(pointer of src, the picture dict) of every picture whose src is a local path."""
    if isinstance(o, dict):
        src = o.get("src")
        if isinstance(src, str) and not src.startswith(("http://", "https://", "asset:")):
            yield path + "/src", o
        for k, v in o.items():
            if k != "distribution":
                yield from _local_pictures(v, f"{path}/{str(k).replace('~', '~0').replace('/', '~1')}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _local_pictures(v, f"{path}/{i}")


def unfetchable(projected: dict, private: dict) -> set[str]:
    """Held items the app cannot fetch a source for: a source of tier "user" is the researcher's own file (the app
    asks for it, fine); tier "link" needs a URL that downloads to the declared sha256 (another declared URL that does
    replaces a wrong one); a held item against a hosted source is never fetched. Returns the held ids whose source
    cannot be fetched (the researcher drops in their copy, or pastes the text)."""
    shipped = (projected.get("meta") or {}).get("sources") or {}
    declared = (private.get("meta") or {}).get("sources") or {}
    held = (projected.get("distribution") or {}).get("held") or {}
    bad = set()
    for key in sorted({e.get("source") for e in held.values()} - {None}):
        dist = (shipped.get(key) or {}).get("distribution") or {}
        tier = dist.get("tier")
        if tier == "user":
            continue
        ok = None
        if tier == "link":
            sha = str(dist.get("sha256") or "").lower()
            src = declared.get(key)
            text = json.dumps(src, ensure_ascii=False) if src is not None else ""
            urls = [dist.get("url"), src.get("url") if isinstance(src, dict) else None, *URL.findall(text)]
            ok = next((u for u in dict.fromkeys(urls) if isinstance(u, str) and u.startswith("http") and len(sha) == 64
                       and url_serves(u, sha)), None)
        if ok:
            dist["url"] = ok
        else:
            bad |= {h for h, e in held.items() if e.get("source") == key}
    return bad


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


def build(paper_id: str, slug: str, private: Path, sessions: int, dry: bool = False) -> dict:
    paper_dir = COMMUNITY / paper_id
    current = json.loads((paper_dir / "current.json").read_text())
    base = load_revision(paper_dir, current["revision"])
    protocol, info = project(private / slug)
    readable_reasons(protocol)
    report = protocols.public_checks(protocol, sessions)   # with the same coverage re-runs as CI
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
                    # exact: every held text restores by hash; approx: some from the researcher's copy's text layer,
                    # labelled approximate; needs_text: some the researcher pastes or captures; reconstruct: the
                    # paper does not print every material (owner 2026-11-24)
                    "readiness": info["readiness"],
                    **({"approxTextItems": info["approx"]} if info.get("approx") else {}),
                    **({"userTextItems": info["needsText"]} if info.get("needsText") else {}),
                    **({"shortLabelsShipped": info["short"]} if info.get("short") else {}),
                    **({"reconstructArms": info["reconstruct"]} if info.get("reconstruct") else {}),
                    **({"unscannedSources": info["unscanned"]} if info.get("unscanned") else {}),
                    "armHoldsResolvedByProjection": sorted(info["holdsResolved"])},
        "contributors": current.get("contributors") or [{"github": "assassin808"}],
    }
    study = {k: v for k, v in study.items() if v is not None}
    # a protocol release ships only the engine's public paths (a legacy revision may carry a paper PDF)
    files = {k: v for k, v in base.items() if not k.startswith(DROP_PREFIXES)
             and any(protocols._match(k, pat) for pat in protocols.PUBLIC_PATHS)}
    files["protocol.json"] = dumps(protocol)
    files["study.json"] = dumps(study)
    files["review/checks-latest.json"] = dumps(report)
    if local_paths(files):
        return {"paper": paper_id, "slug": slug, "status": "invalid",
                "errors": ["a shipped file names a path on a maintainer's machine: " + ", ".join(local_paths(files))]}
    validation = validate_assets(files)
    if validation["status"] == "passed":
        # CI replays 300 sessions (validate-studies.yml); a release must pass exactly that, not only our 1,000-session run.
        replayed = validate_protocol_files(files, sessions=300)
        if replayed["status"] != "passed":
            validation = {**validation, "status": "failed", "errors": replayed["errors"]}
    if validation["status"] != "passed":
        return {"paper": paper_id, "slug": slug, "status": "invalid", "errors": validation["errors"][:8]}
    rid = revision_id(files)
    if dry:
        return {"paper": paper_id, "slug": slug, "status": "would_release", "readiness": info["readiness"],
                "held": info["held"], "short": info["short"], "needsText": info["needsText"], "approx": info["approx"]}
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
               "review": {**REVIEW, "candidateRevision": rid,
                          "reason": ("Protocol released from the maintainers' private study as an unreviewed (checked) "
                                     "release: engine checks pass; no independent fidelity review is claimed. "
                                     + READINESS_REASON[info["readiness"]])},
               "validation": {k: v for k, v in validation.items() if k != "projection"}}
    (paper_dir / "current.json").write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n")
    return {"paper": paper_id, "slug": slug, "status": "released", "revision": rid, "held": info["held"],
            "readiness": info["readiness"],
            "holdsResolved": len(info["holdsResolved"]), "replayable": validation["capabilities"].get("replayable")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--private", type=Path, required=True, help="private hsenv studies directory")
    ap.add_argument("--map", type=Path, required=True, help="{paper_id: {matches: [slug]}}")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--sessions", type=int, default=1000)
    ap.add_argument("--dry", action="store_true", help="check only; write nothing")
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
                results.append(build(paper_id, slugs[0], a.private, a.sessions, a.dry))
            except ProjectionBlocked as exc:
                results.append({"paper": paper_id, "slug": slugs[0], "status": "needs_fix", "error": str(exc)[:600]})
            except Exception as exc:  # one paper's failure must not stop the batch
                results.append({"paper": paper_id, "slug": slugs[0], "status": "error", "error": f"{type(exc).__name__}: {exc}"[:400]})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
