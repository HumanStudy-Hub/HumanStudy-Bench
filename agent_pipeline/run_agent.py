#!/usr/bin/env python3
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
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


def package_root(package: Path) -> Path | None:
    roots = [path for path in package.iterdir() if path.is_dir()] if package.exists() else []
    return roots[0] if len(roots) == 1 else None


def package_progress(package: Path) -> tuple[int, int, list[str]]:
    root = package_root(package)
    missing = list(REQUIRED) if root is None else [name for name in REQUIRED if not (root / name).is_file()]
    total = sum(1 for path in package.rglob("*") if path.is_file()) if package.exists() else 0
    return len(REQUIRED) - len(missing), total, missing


def validate(validator: Path, package: Path) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, str(validator), str(package)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def stop_process(process: subprocess.Popen[str]) -> None:
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


def stream_output(process: subprocess.Popen[str], log_path: Path) -> None:
    with log_path.open("w") as log:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()


def write_result(job: Path, reason: str, valid: bool, detail: str) -> None:
    payload = {
        "reason": reason,
        "package_valid": valid,
        "detail": detail,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (job / "logs/watchdog.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--validator", required=True, type=Path)
    parser.add_argument("--timeout-minutes", type=float, default=25)
    parser.add_argument("--check-interval", type=float, default=30)
    args = parser.parse_args()

    args.job = args.job.resolve()
    package = args.job / "package"
    logs = args.job / "logs"
    package.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    command = [
        "claude",
        "--print",
        "--model",
        args.model,
        "--add-dir",
        str(args.job),
        "--dangerously-skip-permissions",
        args.prompt.read_text(),
    ]
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    output_thread = threading.Thread(target=stream_output, args=(process, logs / "agent.log"), daemon=True)
    output_thread.start()
    deadline = time.monotonic() + args.timeout_minutes * 60

    try:
        while process.poll() is None:
            completed, total, missing = package_progress(package)
            print(f"[package-progress] required={completed}/{len(REQUIRED)} total={total}", flush=True)
            if missing:
                print(f"[package-progress] missing: {' '.join(missing)}", flush=True)
            else:
                try:
                    valid, detail = validate(args.validator, package)
                except subprocess.TimeoutExpired:
                    valid, detail = False, "Validator timed out; retrying."
                print(f"[package-progress] validator={'passed' if valid else 'not-ready'} {detail}", flush=True)
                if valid:
                    write_result(args.job, "validator_passed", True, detail)
                    stop_process(process)
                    output_thread.join(timeout=5)
                    return
            if time.monotonic() >= deadline:
                stop_process(process)
                valid, detail = validate(args.validator, package) if not missing else (False, "Missing required files: " + ", ".join(missing))
                write_result(args.job, "timeout", valid, detail)
                if valid:
                    return
                raise SystemExit(f"Agent timed out before the package became reviewable: {detail}")
            time.sleep(args.check_interval)

        output_thread.join(timeout=5)
        valid, detail = validate(args.validator, package)
        write_result(args.job, "agent_exited", valid, detail)
        if not valid:
            raise SystemExit(f"Claude Code exited with {process.returncode}; package validation failed: {detail}")
    except BaseException:
        stop_process(process)
        raise


if __name__ == "__main__":
    main()
