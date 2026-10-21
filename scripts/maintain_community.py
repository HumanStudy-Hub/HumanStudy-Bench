#!/usr/bin/env python3
"""Integrate a proposal using only trusted code from the target branch.

CI fetches a PR ref but NEVER checks it out, imports it or executes contributed adapters.
This script reads blobs via git show; only accepted research assets and trusted projections
are written. The final commit uses the current main as parent and a non-forced push.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research.catalog import community_identity, digest, normalise, normalise_doi, write_catalog
from scripts.research.revisions import integrate, load_revision, merge_assets, revision_id, safe_path
from scripts.research.review import review_candidate
from scripts.build_studies_index import build_index


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def proposal_from_ref(ref: str, base: str, author: str) -> tuple[Path, dict, dict]:
    if not re.fullmatch(r"[a-f0-9]{40}", ref) or not re.fullmatch(r"[a-f0-9]{40}", base):
        raise ValueError("exact commit identities are required")
    names = git("diff", "--name-only", "--no-renames", "-z", base, ref).decode().split("\0")
    names = [n for n in names if n]
    pattern = re.compile(r"^community/([A-Za-z0-9_.-]+)/proposals/([a-f0-9-]+)/(.+)$")
    matches = [pattern.fullmatch(n) for n in names]
    if not names or not all(matches):
        raise ValueError("a contribution may only add files to one proposal directory")
    roots = {(m[1], m[2]) for m in matches}
    if len(roots) != 1:
        raise ValueError("one proposal per contribution")
    paper_dir, proposal_id = next(iter(roots))
    prefix = f"community/{paper_dir}/proposals/{proposal_id}/"
    if git("ls-tree", base, prefix):
        raise ValueError("published proposal identities cannot be overwritten")
    files, metadata = {}, None
    for name, match in zip(names, matches):
        mode = git("ls-tree", ref, name).decode().split(" ", 1)[0]
        if mode != "100644":
            raise ValueError("only regular research files may be contributed")
        raw = git("show", f"{ref}:{name}")
        rel = match[3]
        if rel == "CONTRIBUTION.json":
            metadata = json.loads(raw)
        else:
            files[safe_path(rel)] = raw
    if not isinstance(metadata, dict) or not metadata.get("paperKey") or not metadata.get("paper"):
        raise ValueError("proposal lacks paper identity and metadata")
    import os
    contributor = metadata.get("contributor", {})
    bot = os.environ.get("COMMUNITY_BOT_LOGIN", "humanstudy-bot")
    if author.casefold() == bot.casefold():
        if not isinstance(contributor.get("hub"), dict) or not contributor["hub"].get("signature"):
            raise ValueError("Hub submissions require their server-signed contributor identity")
    elif str(contributor.get("github", "")).casefold() != author.casefold():
        raise ValueError("the contributor must match the authenticated GitHub PR author")
    # Existing Hub key encoding remains compatible, but no contribution may write another paper's path.
    expected = "paper_" + hashlib.sha256(metadata["paperKey"].encode()).hexdigest()[:24]
    if paper_dir != expected or len(paper_dir) > 300:
        raise ValueError("proposal paper identity does not match its directory")
    metadata["proposalId"] = proposal_id
    metadata.pop("_resolvedPaper", None)
    metadata.pop("_identityCandidates", None)
    catalog_path = ROOT / "co_website/data/research_index.json"
    known = json.loads(catalog_path.read_text()).get("papers", []) if catalog_path.exists() else []
    requested_id = metadata.get("paperId")
    if not requested_id and metadata.get("sourceAlias"):
        source_matches = [p for p in known if metadata["sourceAlias"] in p.get("aliases", [])]
        if len(source_matches) != 1:
            raise ValueError("the original source alias is unknown or ambiguous")
        requested_id = source_matches[0]["id"]
    exact = [p for p in known if metadata["paperKey"] in p.get("aliases", []) or "community:" + metadata["paperKey"] in p.get("aliases", [])]
    if metadata["paperKey"].startswith("title:"):
        exact = [p for p in exact if normalise(p["title"]) == normalise(metadata["paper"]["title"]) and p.get("year") == metadata["paper"].get("year")]
    if requested_id:
        found = next((p for p in known if p["id"] == requested_id), None)
        if found is None:
            raise ValueError("the original catalog paper identity is unknown")
        known_dois = [a[4:] for a in found.get("aliases", []) if a.startswith("doi:")]
        if known_dois and normalise_doi(str(metadata["paper"].get("doi", ""))) not in known_dois:
            raise ValueError("the original paper DOI cannot be silently reassigned")
        source_title = normalise(found["title"])
        proposed_title = normalise(metadata["paper"]["title"])
        title_matches = source_title == proposed_title or (found.get("coverage") == "source-directory-not-complete-paper" and proposed_title in source_title)
        if found not in exact and (not title_matches or found.get("year") is not None and found.get("year") != metadata["paper"].get("year")):
            raise ValueError("the declared source paper and original catalog identity disagree")
        exact = [found]
    if len(exact) == 1:
        metadata["_resolvedPaper"] = {k: exact[0][k] for k in ["id", "aliases", "legacyStudyId"] if k in exact[0]}
    elif not exact:
        candidates = [p["id"] for p in known if normalise(p["title"]) == normalise(metadata["paper"]["title"])
                      and p.get("year") == metadata["paper"].get("year")]
        if candidates:
            metadata["_identityCandidates"] = candidates
    canonical_id = metadata.get("_resolvedPaper", {}).get("id") or community_identity(metadata["paper"], metadata["paperKey"])[0]
    return ROOT / "community" / canonical_id, metadata, files


def maintain(ref: str, base: str, author: str, *, agent_review: bool) -> dict:
    folder, metadata, files = proposal_from_ref(ref, base, author)
    result = integrate(folder, metadata, files)
    if result.get("candidateRevision") and agent_review:
        pointer = folder / "current.json"
        current = json.loads(pointer.read_text()) if pointer.exists() else None
        candidate, conflicts = merge_assets(load_revision(folder, metadata.get("baseRevision")),
                                           load_revision(folder, current["revision"] if current else None), files)
        if not conflicts:
            reviewed = review_candidate(candidate, revision_id(candidate), metadata["paper"])
            result = integrate(folder, metadata, files, review=reviewed)
            result["review"] = reviewed
    report = {"schemaVersion": 1, "proposalId": metadata["proposalId"], "headCommit": ref,
              "paperKey": metadata["paperKey"], **{k: v for k, v in result.items() if k != "validation"}}
    if "validation" in result:
        report["validation"] = {k: v for k, v in result["validation"].items() if k != "projection"}
    reports = folder / "reviews"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / (metadata["proposalId"] + ".json")).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    # Keep the submitted assets immutable for attribution and future revalidation, never as visible cards.
    proposal = folder / "proposals" / metadata["proposalId"]
    proposal.mkdir(parents=True, exist_ok=True)
    for name, raw in files.items():
        path = proposal / safe_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (proposal / "CONTRIBUTION.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    write_catalog(ROOT, build_index())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--author", required=True)
    parser.add_argument("--agent-review", action="store_true")
    args = parser.parse_args()
    print(json.dumps(maintain(args.head, args.base, args.author, agent_review=args.agent_review), ensure_ascii=False))
