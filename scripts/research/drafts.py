"""Hosted paired-draft intake checks. No execution, release approval or scientific review."""
import hashlib
import json
import re
import unicodedata

from hsenv.release import hosted_licence_ok


def draft_problems(files):
    from scripts.research.revisions import safe_path
    from scripts.research.catalog import normalise_doi

    errors = []
    try:
        study, protocol, program, manifest = (json.loads(files[n]) for n in
            ("study.json", "protocol.json", "program.json", "sources/SOURCE.json"))
        if not all(isinstance(x, dict) for x in (study, protocol, program, manifest)):
            raise ValueError("paired documents must be objects")
        distribution = protocol.get("distribution")
        if not isinstance(distribution, dict) or distribution.get("format") != "hsenv-projection/1" \
                or distribution.get("strict") is not True or distribution.get("held") != {}:
            errors.append("draft: requires a strict hosted projection without held material")
        if "░" in json.dumps({k: v for k, v in protocol.items() if k != "distribution"}, ensure_ascii=False):
            errors.append("draft: contains withheld material")
        if study.get("quality") != "draft" or any(k in study for k in ("fidelity", "visibility")):
            errors.append("draft: must be a draft without contributor-supplied release approval")
        scholarly = program.get("extensions", {}).get("scholarly", {})
        expected = {"path": "protocol.json", "id": protocol.get("id"),
                    "sha256": hashlib.sha256(files["protocol.json"]).hexdigest()}
        if scholarly.get("profile") != "scholarly-v1" or scholarly.get("protocol") != expected:
            errors.append("draft: program protocol binding does not match the submitted bytes")
        source = program.get("source", {})
        title = lambda s: " ".join(unicodedata.normalize("NFKC", str(s)).casefold().split())
        if not title(source.get("title", "")) or title(source.get("title")) != title(study.get("title")):
            errors.append("draft: study and Program paper titles disagree")
        if normalise_doi(str(source.get("doi") or "")) != normalise_doi(str(study.get("doi") or "")):
            errors.append("draft: study and Program paper DOIs disagree")
        records = manifest.get("files")
        if manifest.get("schemaVersion") != 1 or not isinstance(records, list) or not records:
            raise ValueError("source manifest needs public files")
        declared = {}
        for record in records:
            path = safe_path(record["path"])
            if not path.startswith("sources/open/") or path in declared:
                raise ValueError("source manifest contains a duplicate or non-open path")
            data = files.get(path)
            if data is None or record.get("bytes") != len(data) or record.get("sha256") != hashlib.sha256(data).hexdigest() \
                    or not hosted_licence_ok(record.get("license", "")):
                raise ValueError("source manifest bytes, hash or redistribution license is invalid")
            declared[path] = record
        if any(n.startswith("sources/open/") and n not in declared for n in files):
            errors.append("draft: an open file lacks source metadata")
        sources = protocol.get("meta", {}).get("sources")
        if not isinstance(sources, dict) or not sources:
            raise ValueError("protocol needs hosted sources")
        for entry in sources.values():
            item = declared.get(entry.get("file")) if isinstance(entry, dict) else None
            dist = entry.get("distribution") if isinstance(entry, dict) else None
            if not item or not isinstance(dist, dict) or dist.get("tier") != "hosted" \
                    or dist.get("sha256") != item["sha256"] or dist.get("licence") != item["license"] \
                    or not isinstance(dist.get("attribution"), str) or not dist["attribution"].strip():
                errors.append("draft: each protocol source needs matching hosted bytes, license and attribution")
        for path in [source.get("filename"), *(e.get("path") for e in program.get("evidence", []))]:
            if path and path not in declared:
                errors.append("draft: a Program source is absent from the public source manifest")
        private = {"private", "researcher", "quarantine", "runs", "human-data", "original-text", "rehydrated", "restored"}
        for name in files:
            lower = name.lower().split("/")
            if any(p in private for p in lower) or lower[-1].endswith((".annotations.json", ".jsonl", ".ndjson", ".log", ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar")) \
                    or re.match(r"^(?:records|responses|raw[-_](?:data|responses|records)|(?:participant|human)[-_](?:data|responses|records))\.", lower[-1]):
                errors.append("draft: private/raw record paths are not accepted")
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        errors.append(f"draft: invalid paired source metadata ({exc})")
    return errors
