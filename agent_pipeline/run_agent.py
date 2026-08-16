#!/usr/bin/env python3
import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


REQUIRED = (
    "index.json",
    "study.json",
    "source/specification.json",
    "source/metadata.json",
    "source/ground_truth.json",
    "source/evidence.json",
    "audit/missing_information.json",
)


class ProgressPublisher:
    def __init__(self, token: str, repo: str, branch: str, path: str) -> None:
        self.token = token
        self.repo = repo
        self.branch = branch
        self.path = path
        self.sha: str | None = None
        self._load_sha()

    def _request(self, method: str, body: dict | None = None) -> dict:
        encoded_path = urllib.parse.quote(self.path, safe="/")
        url = f"https://api.github.com/repos/{self.repo}/contents/{encoded_path}"
        if method == "GET":
            url += "?ref=" + urllib.parse.quote(self.branch, safe="")
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, method=method, headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "HumanStudy-Hub-Agent",
        })
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read())

    def _load_sha(self) -> None:
        try:
            self.sha = self._request("GET").get("sha")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise

    def publish(self, payload: dict) -> None:
        body = {
            "message": f"agent: progress {payload['completedRequired']}/{payload['totalRequired']}",
            "branch": self.branch,
            "content": base64.b64encode((json.dumps(payload, indent=2) + "\n").encode()).decode(),
        }
        if self.sha:
            body["sha"] = self.sha
        response = self._request("PUT", body)
        self.sha = response["content"]["sha"]


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


def ensure_readme(package: Path) -> None:
    root = package_root(package)
    if root is None or (root / "README.md").exists():
        return
    title = root.name.replace("-", " ").replace("_", " ").title()
    try:
        study = json.loads((root / "study.json").read_text())
        paper = study.get("paper", {}) if isinstance(study, dict) else {}
        title = paper.get("title") or study.get("title") or title
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    (root / "README.md").write_text(
        f"# {title}\n\n"
        "This HumanStudy-Hub package was reconstructed from a published paper.\n\n"
        "- Review `study.json` for the study overview and readiness status.\n"
        "- Review `audit/missing_information.json` before running the study.\n"
        "- Run `python task/adapter.py --smoke-test` to check the package entry point.\n"
    )


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


def progress_payload(phase: str, completed: int, total: int, missing: list[str]) -> dict:
    return {
        "phase": phase,
        "completedRequired": completed,
        "totalRequired": len(REQUIRED),
        "totalFiles": total,
        "missing": missing,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--validator", required=True, type=Path)
    parser.add_argument("--timeout-minutes", type=float, default=25)
    parser.add_argument("--check-interval", type=float, default=10)
    parser.add_argument("--progress-repo")
    parser.add_argument("--progress-branch")
    parser.add_argument("--progress-path")
    args = parser.parse_args()

    args.job = args.job.resolve()
    package = args.job / "package"
    logs = args.job / "logs"
    package.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    progress_token = os.environ.get("PIPELINE_PROGRESS_TOKEN", "")
    publisher = None
    if progress_token and args.progress_repo and args.progress_branch and args.progress_path:
        publisher = ProgressPublisher(progress_token, args.progress_repo, args.progress_branch, args.progress_path)
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
    claude_env = os.environ.copy()
    claude_env.pop("PIPELINE_PROGRESS_TOKEN", None)
    claude_env.pop("HUMANSTUDY_PIPELINE_TOKEN", None)
    claude_env.pop("GITHUB_TOKEN", None)
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=claude_env,
    )
    output_thread = threading.Thread(target=stream_output, args=(process, logs / "agent.log"), daemon=True)
    output_thread.start()
    deadline = time.monotonic() + args.timeout_minutes * 60

    try:
        while process.poll() is None:
            completed, total, missing = package_progress(package)
            print(f"[package-progress] required={completed}/{len(REQUIRED)} total={total}", flush=True)
            phase = "building_package" if missing else "validating_package"
            if publisher:
                try:
                    publisher.publish(progress_payload(phase, completed, total, missing))
                except Exception as exc:
                    print(f"[package-progress] could not publish frontend progress: {exc}", flush=True)
            if missing:
                print(f"[package-progress] missing: {' '.join(missing)}", flush=True)
            else:
                ensure_readme(package)
                try:
                    valid, detail = validate(args.validator, package)
                except subprocess.TimeoutExpired:
                    valid, detail = False, "Validator timed out; retrying."
                print(f"[package-progress] validator={'passed' if valid else 'not-ready'} {detail}", flush=True)
                if valid:
                    if publisher:
                        try:
                            publisher.publish(progress_payload("ready_for_review", completed, total, []))
                        except Exception as exc:
                            print(f"[package-progress] could not publish ready state: {exc}", flush=True)
                    write_result(args.job, "validator_passed", True, detail)
                    stop_process(process)
                    output_thread.join(timeout=5)
                    return
            if time.monotonic() >= deadline:
                stop_process(process)
                valid, detail = validate(args.validator, package) if not missing else (False, "Missing required files: " + ", ".join(missing))
                write_result(args.job, "timeout", valid, detail)
                if publisher:
                    try:
                        publisher.publish(progress_payload("ready_for_review" if valid else "timed_out", completed, total, missing))
                    except Exception as exc:
                        print(f"[package-progress] could not publish timeout state: {exc}", flush=True)
                if valid:
                    return
                raise SystemExit(f"Agent timed out before the package became reviewable: {detail}")
            time.sleep(args.check_interval)

        output_thread.join(timeout=5)
        ensure_readme(package)
        valid, detail = validate(args.validator, package)
        write_result(args.job, "agent_exited", valid, detail)
        completed, total, missing = package_progress(package)
        if publisher:
            try:
                publisher.publish(progress_payload("ready_for_review" if valid else "failed", completed, total, missing))
            except Exception as exc:
                print(f"[package-progress] could not publish final state: {exc}", flush=True)
        if not valid:
            raise SystemExit(f"Claude Code exited with {process.returncode}; package validation failed: {detail}")
    except BaseException:
        stop_process(process)
        raise


if __name__ == "__main__":
    main()
