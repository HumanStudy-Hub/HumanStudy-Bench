#!/usr/bin/env python3
"""Add the standard run interface to an existing buffer package via Claude Code.

Legacy packages expose a study-specific adapter/evaluator interface. This script
spawns Claude Code with `migrate_standard.md` to add the standard
`run_sessions(agent_fn, seed)` + `evaluate(sessions)` interface, then commits the
result back to the job branch.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, type=Path, help="Job directory containing package/ and job.json")
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-5"))
    parser.add_argument("--jobs-repo", default="HumanStudy-Hub/humanstudy-hub-jobs")
    parser.add_argument("--branch", help="Job branch name (defaults to jobs/<job id from job.json>)")
    args = parser.parse_args()

    job = args.job.resolve()
    import json
    job_id = json.loads((job / "job.json").read_text(encoding="utf-8")).get("id") if (job / "job.json").exists() else job.name
    branch = args.branch or f"jobs/{job_id}"

    contract = REPO_ROOT / "agent_pipeline" / "migrate_standard.md"
    prompt = (
        f"{contract.read_text()}\n\n"
        f"## This package\n\n"
        f"The job directory is `{job}`. The package is under `{job}/package/`. "
        f"Add the standard run interface and verify it imports and runs.\n"
    )

    env = os.environ.copy()
    for secret in ("PIPELINE_PROGRESS_TOKEN", "HUMANSTUDY_PIPELINE_TOKEN", "GITHUB_TOKEN", "PLAYGROUND_KEY_SECRET"):
        env.pop(secret, None)
    env["ANTHROPIC_BASE_URL"] = "https://openrouter.ai/api"
    env["ANTHROPIC_AUTH_TOKEN"] = os.environ.get("OPENROUTER_API_KEY", "")
    env["ANTHROPIC_API_KEY"] = ""
    env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = args.model
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = args.model
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = args.model
    env["CLAUDE_CODE_SUBAGENT_MODEL"] = args.model
    env["OPENROUTER_MODEL"] = args.model

    process = subprocess.Popen(
        [
            "claude",
            "--print",
            "--model", args.model,
            "--add-dir", str(job),
            "--dangerously-skip-permissions",
            prompt,
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    process.wait()

    if process.returncode != 0:
        raise SystemExit(f"migration agent exited with {process.returncode}")

    token = os.environ.get("PIPELINE_PROGRESS_TOKEN", "") or os.environ.get("HUMANSTUDY_PIPELINE_TOKEN", "")
    if token:
        subprocess.run(["git", "config", "user.name", "HumanStudy-Hub Agent"], cwd=job, check=False)
        subprocess.run(["git", "config", "user.email", "pipeline@humanstudy-hub.org"], cwd=job, check=False)
        subprocess.run(["git", "add", "package"], cwd=job, check=False)
        subprocess.run(["git", "commit", "-m", f"agent: add standard run interface {job_id}"], cwd=job, check=False)
        remote = f"https://x-access-token:{token}@github.com/{args.jobs_repo}.git"
        subprocess.run(["git", "push", remote, f"HEAD:{branch}"], cwd=job, check=False)
        print("committed and pushed the migration")
    else:
        print("no pipeline token; skipping commit/push. Review the package changes locally.")


if __name__ == "__main__":
    main()
