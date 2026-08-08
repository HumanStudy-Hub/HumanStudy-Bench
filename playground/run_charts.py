#!/usr/bin/env python3
"""Have Claude Code chart and interpret a finished playground run.

The agent is given the run directory and the contract in `playground/CLAUDE.md`,
and writes `output/charts.json`. Whatever it produces is validated before it is
kept; if it is missing, invalid, or the agent runs out of time, the deterministic
charts from `default_charts.py` are used instead. A run therefore always ends
with charts a researcher can read.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from playground.default_charts import build_charts
from playground.validate_charts import ChartError, validate


def build_prompt(contract: Path, run_dir: Path) -> str:
    return (
        f"{contract.read_text()}\n\n"
        "## This run\n\n"
        f"The run directory is `{run_dir}`. Read `run.json`, `output/analysis.json`, "
        "`output/evaluation.json`, `output/transcript_sample.json`, and "
        "`output/charts.default.json` there.\n\n"
        f"Write your charts and interpretation to `{run_dir / 'output' / 'charts.json'}`. "
        "Write that file exactly once, when it is complete and valid, then stop.\n"
    )


def stream_output(process: "subprocess.Popen[str]", log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()


def stop_process(process: "subprocess.Popen[str]") -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)


def accept_charts(path: Path) -> Dict[str, Any] | None:
    """Return the agent's charts when they are present and valid."""
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text())
        validate(document)
    except (OSError, json.JSONDecodeError) as error:
        print(f"[charts] agent output could not be read: {error}", flush=True)
        return None
    except ChartError as error:
        print(f"[charts] agent output rejected: {error}", flush=True)
        return None
    document["source"] = "agent"
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--contract", type=Path, default=Path(__file__).resolve().parent / "CLAUDE.md")
    parser.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-5"))
    parser.add_argument("--timeout-minutes", type=float, default=8)
    args = parser.parse_args()

    run_dir = args.run.resolve()
    output_dir = run_dir / "output"
    logs_dir = run_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = output_dir / "analysis.json"
    if not analysis_path.exists():
        raise SystemExit("This run has no analysis.json, so there is nothing to chart.")
    analysis = json.loads(analysis_path.read_text())

    # The baseline is written first: it is the agent's reference and the fallback.
    fallback = build_charts(analysis)
    (output_dir / "charts.default.json").write_text(json.dumps(fallback, indent=2) + "\n")

    charts_path = output_dir / "charts.json"
    prompt_path = run_dir / "charts-prompt.md"
    prompt_path.write_text(build_prompt(args.contract, run_dir))

    agent_env = os.environ.copy()
    for secret in ("PIPELINE_PROGRESS_TOKEN", "HUMANSTUDY_PIPELINE_TOKEN", "GITHUB_TOKEN", "OPENROUTER_API_KEY", "PLAYGROUND_KEY_SECRET"):
        agent_env.pop(secret, None)
    agent_env["ANTHROPIC_AUTH_TOKEN"] = os.environ.get("CHARTS_AGENT_TOKEN", os.environ.get("ANTHROPIC_AUTH_TOKEN", ""))

    process = subprocess.Popen(
        [
            "claude",
            "--print",
            "--model", args.model,
            "--add-dir", str(run_dir),
            "--dangerously-skip-permissions",
            prompt_path.read_text(),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=agent_env,
    )
    output_thread = threading.Thread(target=stream_output, args=(process, logs_dir / "charts.log"), daemon=True)
    output_thread.start()
    deadline = time.monotonic() + args.timeout_minutes * 60

    try:
        while process.poll() is None:
            if time.monotonic() >= deadline:
                print("[charts] the charting agent ran out of time", flush=True)
                stop_process(process)
                break
            time.sleep(5)
    finally:
        stop_process(process)
        output_thread.join(timeout=5)

    document = accept_charts(charts_path)
    if document is None:
        print("[charts] using the deterministic charts for this run", flush=True)
        document = fallback
    charts_path.write_text(json.dumps(document, indent=2) + "\n")
    print(f"[charts] wrote {len(document['charts'])} charts from the {document['source']} source", flush=True)


if __name__ == "__main__":
    main()
