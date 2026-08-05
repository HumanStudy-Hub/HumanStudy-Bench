#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path


REQUIRED = (
    "README.md",
    "study.json",
    "source/paper_metadata.json",
    "source/extraction.json",
    "source/evidence.json",
    "source/open_materials.json",
    "materials/materials.json",
    "task/task.json",
    "task/adapter.py",
    "evaluation/evaluation.py",
    "audit/provenance.json",
    "audit/missing_information.json",
    "audit/agent_report.md",
)


def main() -> None:
    package = Path(sys.argv[1]).resolve()
    roots = [path for path in package.iterdir() if path.is_dir()] if package.exists() else []
    if len(roots) != 1:
        raise SystemExit("package must contain exactly one paper folder")
    root = roots[0]
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise SystemExit("missing required package files: " + ", ".join(missing))
    for path in root.rglob("*.json"):
        try:
            json.loads(path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid JSON in {path.relative_to(root)}: {exc}") from exc
    result = subprocess.run(
        [sys.executable, str(root / "task/adapter.py"), "--smoke-test"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise SystemExit("adapter smoke test failed:\n" + result.stdout + result.stderr)
    print(f"Validated agent package: {root.name}")


if __name__ == "__main__":
    main()
