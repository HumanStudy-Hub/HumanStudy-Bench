"""Metadata-only public corpus adapters. Never download gated tests or execute source scripts."""
from __future__ import annotations

import argparse
import csv
import io
import hashlib
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.research.catalog import digest, paper_identity

PSYCH_REPO = "marcelbinz/Psych-201"
REANALYSIS_REPO = "tobihol/agentic-reproducibility"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "HumanStudy-public-metadata-import/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def text(raw: dict, *names) -> str:
    return next((str(raw[k]).strip() for k in names if raw.get(k) is not None and str(raw[k]).strip()), "")


def record(meta: dict, source_id: str, resources: list[dict], *, search_text: str = "", coverage: str = "source-metadata") -> dict:
    pid, aliases = paper_identity(meta, source_id)
    return {"id": pid, "aliases": aliases, "title": meta["title"], "authors": meta.get("authors", []),
            "year": meta.get("year"), "description": meta.get("description", ""),
            "currentRevision": None, "conceptIds": [], "conceptAssignments": [],
            "capabilities": {"discoverable": True, "traceable": False, "reanalyzable": False,
                             "replayable": False, "humanBenchmark": False},
            "studies": [], "effectEstimates": [], "resources": resources, "coverage": coverage,
            "searchText": "\n".join([meta["title"], meta.get("description", ""), search_text])}


def psych201_records(tree: dict, read: callable) -> list[dict]:
    if tree.get("truncated"):
        raise ValueError("GitHub returned a truncated source tree; do not silently import a partial corpus")
    sha = tree.get("commitSha", tree["sha"])
    paths = {item["path"]: item for item in tree["tree"] if item["type"] == "blob"}
    folders = sorted(path.split("/")[0] for path in paths if path.count("/") == 1 and path.endswith("/README.md"))
    def one(folder):
        readme_path = folder + "/README.md"
        raw = read(readme_path).decode("utf-8")
        heading = re.search(r"(?mi)^#{1,6}[\t ]*Reference:?[\t ]*(?:\n|$)", raw)
        reference_body = raw[heading.end():] if heading else raw
        reference_body = re.split(r"(?m)^#{1,6}[\t ]+", reference_body.strip())[0].strip()
        # A README can contain an entire methods section. Its citation is not that whole document.
        # Preserve all source content in searchText; the card displays the complete citation paragraph.
        reference = re.split(r"\n[\t ]*\n", reference_body)[0].strip()
        # The full citation is retained rather than hallucinating a title/author parse.
        dois = re.findall(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s<>]+)", reference)
        years = re.findall(r"\((19\d{2}|20\d{2})(?:[^)]*)\)", reference)
        resources = [{"id": "readme", "kind": "metadata", "url": f"https://github.com/{PSYCH_REPO}/blob/{sha}/{readme_path}",
                      "sourceVersion": sha, "sha256": hashlib.sha256(raw.encode()).hexdigest(), "access": "public", "license": "Apache-2.0 (repository)",
                      "reference": reference}]
        generator = folder + "/generate_prompts.py"
        if generator in paths:
            resources.append({"id": "generator", "kind": "transcription-code",
                              "url": f"https://github.com/{PSYCH_REPO}/blob/{sha}/{generator}",
                              "sourceVersion": sha, "gitBlob": paths[generator]["sha"],
                              "access": "public", "license": "Apache-2.0 (repository)", "executed": False})
        for url in dict.fromkeys(re.findall(r"https?://[^\s<>]+", raw)):
            resources.append({"id": "original-" + digest(url)[:12], "kind": "original-data-link", "url": url,
                              "access": "unverified", "license": "unverified", "sourceVersion": sha})
        item = record({"title": reference or folder, "doi": dois[0] if len(dois) == 1 else "",
                       "year": int(years[0]) if years else None}, "psych201:" + folder, resources,
                      search_text=raw, coverage="source-directory-not-complete-paper")
        item["sourceMetadata"] = {"corpus": "Psych-201", "directory": folder,
                                  "transcriptsAreObservedHumanBehavior": True,
                                  "overlapWithPsych101": "must-check-participant-flags-before-analysis",
                                  "substudyCoverage": "not-yet-verified", "reference": reference}
        return item
    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(one, folders))


def reanalysis_records(raw: bytes, corpus: str, sha: str) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    papers = []
    for row in rows:
        source_id = f"reanalysis:{corpus}:" + text(row, "paper_id")
        if not text(row, "paper_id") or not text(row, "title"):
            raise ValueError("source labeling row lacks a stable identity/title")
        resource = {"id": "claim-label", "kind": "claim-metadata", "access": "public",
                    "license": "MIT (repository); underlying papers/data separately licensed",
                    "url": f"https://github.com/{REANALYSIS_REPO}/blob/{sha}/data/labeling/{corpus}_labeling.csv",
                    "sourceVersion": sha, "rowId": text(row, "paper_id"), "fields": row}
        item = record({"title": row["title"], "doi": text(row, "doi")}, source_id, [resource],
                      search_text="\n".join(row.values()), coverage="focal-claim-metadata")
        item["sourceMetadata"] = {"corpus": corpus, "sourceId": text(row, "paper_id"),
                                  "researchType": text(row, "labeler_data_type") or "unverified",
                                  "substudyCoverage": "focal-claim-not-all-paper-studies"}
        papers.append(item)
    return papers


def import_source(source: str, output: Path, *, source_ref: str = "main") -> dict:
    repo = PSYCH_REPO if source == "psych201" else REANALYSIS_REPO
    commit_sha = json.loads(fetch(f"https://api.github.com/repos/{repo}/commits/{source_ref}"))["sha"]
    tree = json.loads(fetch(f"https://api.github.com/repos/{repo}/git/trees/{commit_sha}?recursive=1"))
    if tree.get("truncated"):
        raise ValueError("truncated GitHub source tree")
    sha = commit_sha
    tree["commitSha"] = commit_sha
    def read(path):
        return fetch(f"https://raw.githubusercontent.com/{repo}/{sha}/{path}")
    if source == "psych201":
        papers = psych201_records(tree, read)
    else:
        papers = [p for corpus in ["multi100", "score"]
                  for p in reanalysis_records(read(f"data/labeling/{corpus}_labeling.csv"), corpus, sha)]
    result = {"schemaVersion": 1, "source": source, "sourceRepository": repo,
              "sourceVersion": sha, "importerVersion": 1, "kind": "metadata-only",
              "recordCount": len(papers), "papers": papers}
    result["snapshot"] = digest(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["psych201", "reanalysis"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-ref", default="main", help="Commit SHA for an exactly repeatable import")
    args = parser.parse_args()
    batch = import_source(args.source, args.output, source_ref=args.source_ref)
    print(json.dumps({k: v for k, v in batch.items() if k != "papers"}))
