#!/usr/bin/env python3
import argparse
import json
import os
import urllib.request
import zipfile
from pathlib import Path


def fetch_materials(job_dir: Path, job: dict) -> None:
    """Download and extract optional uploaded open materials.

    The web app uploads a zip (a chosen folder is zipped in the browser) and
    stores a short-lived signed URL in job.json. This runs in the same step as
    the prompt build so the workflow never needs its own download step. A
    missing or stale link must not sink an otherwise healthy paper-only build.
    """
    url = job.get("openMaterialsUrl")
    if not url:
        return
    input_dir = job_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    archive = input_dir / "open_materials.zip"
    try:
        with urllib.request.urlopen(url, timeout=300) as response:
            archive.write_bytes(response.read())
    except Exception as exc:
        print(f"Could not download open materials ({exc}); continuing without them.")
        return
    out = (input_dir / "open_materials").resolve()
    out.mkdir(parents=True, exist_ok=True)
    # The archive is user-supplied and untrusted: refuse any member that would
    # escape the target directory before extracting.
    try:
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                member = (out / name).resolve()
                if member != out and not str(member).startswith(str(out) + os.sep):
                    raise ValueError(f"unsafe path in archive: {name}")
            zf.extractall(out)
    except (zipfile.BadZipFile, ValueError) as exc:
        print(f"Could not extract open materials ({exc}); continuing without them.")
        return
    print(f"Extracted open materials into {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    job = json.loads((args.job / "job.json").read_text())
    fetch_materials(args.job, job)
    contract = args.contract.read_text()
    osf = job.get("osfUrl")
    materials_dir = (args.job / "input" / "open_materials").resolve()
    has_materials = materials_dir.is_dir()
    if has_materials and osf:
        external_rule = (
            f"External material was supplied both as an uploaded archive and as a URL. "
            f"Read the uploaded files under `{materials_dir}`, and you may also follow `{osf}` "
            "and links directly contained in those materials."
        )
    elif has_materials:
        external_rule = (
            f"Uploaded open materials were supplied and extracted under `{materials_dir}`. "
            "Read those local files; do not fetch anything over the network."
        )
    elif osf:
        external_rule = (
            f"External material was explicitly supplied: `{osf}`. You may access this URL and links directly contained in its materials."
        )
    else:
        external_rule = (
            "No external source was supplied. Network research is not authorized: do not search, "
            "browse, fetch websites, resolve the DOI, or discover OSF materials. Use only the uploaded PDF."
        )
    materials_line = f"\n- Open materials: `{materials_dir}` (read these local files)" if has_materials else ""
    prompt = f"""{contract}

## Current job

- Job directory: `{args.job.resolve()}`
- Paper: `{(args.job / 'input/paper.pdf').resolve()}`
- Original filename: `{job.get('paperName', 'paper.pdf')}`
- Contributor: `{job.get('contributorName', 'Unknown')}`
- External-source policy: {external_rule}{materials_line}

Complete the full extraction and package build now. Write all deliverables under
`{(args.job / 'package').resolve()}`. Do not modify files outside the job directory.
"""
    args.output.write_text(prompt)


if __name__ == "__main__":
    main()
