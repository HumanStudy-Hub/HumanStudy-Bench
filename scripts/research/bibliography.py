"""Resolve display metadata (title, authors, year) for imported papers whose source gives only a citation.

Psych-201 READMEs give a free-text reference, which the importer keeps verbatim instead of guessing
a title. This script looks each reference up in Crossref and writes `catalog/bibliography.json`,
which `build_catalog` applies. A record is accepted only when the Crossref title appears in the
source reference (or the DOI came from the source), so nothing is invented. Unresolved papers are
listed for manual review and keep their source text.

    python -m scripts.research.bibliography --output catalog/bibliography.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

CROSSREF = "https://api.crossref.org/works"
USER_AGENT = "HumanStudy-catalog-bibliography/1 (https://github.com/HumanStudy-Hub/HumanStudy-Bench)"
CITATION = re.compile(r"https?://|doi\.org|\bdoi:|\(\d{4}[a-z]?[^)]*\)|^\W*(?:\*\*)?reference", re.I)


def key(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def get(url: str) -> dict | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            time.sleep(2 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 * (attempt + 1))
    return None


def work_metadata(work: dict) -> dict | None:
    titles = [t for t in work.get("title", []) if t.strip()]
    if not titles:
        return None
    title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", titles[0])).strip().rstrip(".")
    title = re.sub(r"\s+([?!:;,])", r"\1", title)
    subtitle = [s for s in work.get("subtitle", []) if s.strip()]
    if subtitle and key(subtitle[0]) not in key(title):
        title = f"{title}: {re.sub(r'<[^>]+>', '', subtitle[0]).strip()}"
    authors = [" ".join(p for p in (a.get("given"), a.get("family")) if p) or a.get("name", "")
               for a in work.get("author", [])]
    parts = next((work[k]["date-parts"][0] for k in ("published-print", "published-online", "issued", "created")
                  if work.get(k, {}).get("date-parts", [[None]])[0][0]), [None])
    return {"title": title, "authors": [a for a in authors if a], "year": parts[0], "doi": work.get("DOI", "").lower()}


def dois_in(text: str) -> list[str]:
    """DOIs written in a reference, including ones carried by publisher and preprint URLs."""
    found = re.findall(r"\b(10\.\d{4,9}/[^\s<>\"\)\]#?]+)", text)
    found += ["10.1038/" + m for m in re.findall(r"nature\.com/articles/([a-z0-9.-]+)", text)]
    found += ["10.31234/osf.io/" + m for m in re.findall(r"osf\.io/preprints/psyarxiv/([a-z0-9]+?)(?:_v\d+)?\b", text)]
    return list(dict.fromkeys(d.rstrip(".,;\\").lower() for d in found))


def reference_text(paper: dict) -> str:
    """The citation paragraph. Some READMEs put it under a bare "Reference" heading line."""
    reference = paper.get("sourceMetadata", {}).get("reference") or paper["title"]
    if len(re.sub(r"[*#:\s]|references?", "", reference, flags=re.I)) >= 20:
        return reference
    raw = paper.get("searchText", "")
    after = re.split(r"(?im)^[*#\s]*references?[*:\s]*$", raw, maxsplit=2)
    for block in after[1:]:
        for paragraph in re.split(r"\n\s*\n", block):
            paragraph = paragraph.strip()
            if len(re.sub(r"[*#:\s]|references?", "", paragraph, flags=re.I)) >= 20:
                return paragraph
    return reference


def resolve(reference: str, known_doi: str) -> tuple[dict | None, str]:
    for doi in dict.fromkeys([d for d in [known_doi, *dois_in(reference)] if d]):
        work = get(f"{CROSSREF}/{urllib.parse.quote(doi, safe='/')}")
        meta = work_metadata(work["message"]) if work else None
        if meta:
            return meta, "crossref-doi"
    query = re.sub(r"https?://\S+", " ", reference)
    found = get(f"{CROSSREF}?rows=3&query.bibliographic={urllib.parse.quote(query[:600])}")
    for item in (found or {}).get("message", {}).get("items", []):
        meta = work_metadata(item)
        # Accept a search hit only when its title is actually written in the source reference.
        if meta and len(key(meta["title"])) >= 12 and key(meta["title"]) in key(reference):
            return meta, "crossref-search"
    return None, "unresolved"


def build(root: Path, previous: dict) -> dict:
    entries, unresolved = dict(previous.get("papers", {})), []
    psych = json.loads((root / "catalog/imports/psych201.json").read_text())
    for paper in psych["papers"]:
        if paper["id"] in entries or not CITATION.search(paper["title"]):
            continue
        reference = reference_text(paper)
        doi = next((a[4:] for a in paper["aliases"] if a.startswith("doi:")), "")
        meta, method = resolve(reference, doi)
        if meta:
            # The year the source cites (e.g. the published version) wins over Crossref's record date.
            cited = re.search(r"\((19\d{2}|20\d{2})", reference)
            year = int(cited.group(1)) if cited else meta["year"]
            entries[paper["id"]] = {**meta, "year": year, "method": method, "reference": reference}
        else:
            unresolved.append({"id": paper["id"], "reference": reference})
        time.sleep(0.2)
    # Reanalysis titles arrive force-capitalised ("Ict", "Covid-19"). Take Crossref's casing for
    # the same title, by DOI only; a different title is never swapped in.
    reanalysis = json.loads((root / "catalog/imports/reanalysis.json").read_text())
    for paper in reanalysis["papers"]:
        if paper["id"] in entries:
            continue
        words = re.findall(r"[A-Za-z][A-Za-z'’-]*", paper["title"])
        if len(words) < 4 or sum(w[0].isupper() for w in words) < len(words) * 0.9:
            continue
        doi = next((a[4:] for a in paper["aliases"] if a.startswith("doi:")), "")
        work = get(f"{CROSSREF}/{urllib.parse.quote(doi, safe='/')}") if doi else None
        meta = work_metadata(work["message"]) if work else None
        same = meta and key(meta["title"]).replace(" ", "") == key(paper["title"]).replace(" ", "")
        if (same and meta["title"] != paper["title"]
                and meta["title"] != meta["title"].upper()):
            entries[paper["id"]] = {"title": meta["title"], "method": "crossref-doi-casing", "doi": meta["doi"]}
        time.sleep(0.2)
    return {"schemaVersion": 1, "source": "Crossref", "papers": dict(sorted(entries.items())),
            "unresolved": sorted(unresolved, key=lambda u: u["id"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("catalog/bibliography.json"))
    args = parser.parse_args()
    previous = json.loads(args.output.read_text()) if args.output.exists() else {}
    result = build(Path("."), previous)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"resolved": len(result["papers"]), "unresolved": len(result["unresolved"])}))
