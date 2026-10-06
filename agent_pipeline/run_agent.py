#!/usr/bin/env python3
import argparse
import base64
import json
import os
import shutil
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

try:
    from agent_pipeline.studio_output import studio_mode, validate_build_sidecars, validate_discussion
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from studio_output import studio_mode, validate_build_sidecars, validate_discussion


REQUIRED = (
    "study.json",
    "source/paper_metadata.json",
    "source/evidence.json",
    "materials/materials.json",
    "task/task.json",
    "task/adapter.py",
    "evaluation/evaluation.py",
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


def studio_complete(job: Path, request_id: str | None) -> bool:
    if request_id is None:
        return True
    try:
        marker = json.loads((job / "studio-complete.json").read_text())
        return marker.get("requestId") == request_id and marker.get("status") == "complete"
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


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


def cleanup_materials(job: Path) -> None:
    """Drop raw open materials once the agent has finished with them, so large
    archives are not committed back to the jobs repository."""
    for rel in ("input/open_materials.zip", "input/open_materials"):
        path = job / rel
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink(missing_ok=True)


def progress_payload(phase: str, completed: int, total: int, missing: list[str], mode: str = "build") -> dict:
    return {
        "phase": phase,
        "completedRequired": completed,
        "totalRequired": 0 if mode == "discuss" else len(REQUIRED),
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
    logs.mkdir(parents=True, exist_ok=True)
    studio_request = args.job / "studio_request.json"
    request_id = None
    mode = "build"
    accepted_model = None
    if studio_request.is_file():
        request = json.loads(studio_request.read_text())
        request_id = request["requestId"]
        mode = studio_mode(request)
        if request.get("purpose") == "accepted-model-sync":
            document = request.get("document")
            if not isinstance(document, dict) or document.get("model") is None:
                raise ValueError("accepted-model-sync requires document.model")
            accepted_model = document["model"]
        # A copied package may already validate. Only this run's completion
        # marker can release the watchdog after refinement.
        (args.job / "studio-complete.json").unlink(missing_ok=True)
        (args.job / "studio-turn.json").unlink(missing_ok=True)
    if mode == "build":
        package.mkdir(parents=True, exist_ok=True)
        if request_id is not None:
            root = package_root(package)
            if root is not None:
                # Copied packages may carry the previous turn's sidecars.
                # This build must produce its own validated model and reply.
                (root / "studio-model.json").unlink(missing_ok=True)
                (root / "studio-reply.md").unlink(missing_ok=True)

    def check_output() -> tuple[bool, str]:
        if not studio_complete(args.job, request_id):
            return False, "Studio completion marker is missing or does not match this request."
        if mode == "discuss":
            assert request_id is not None
            return validate_discussion(args.job, request_id)
        valid, detail = validate(args.validator, package)
        if valid and request_id is not None:
            valid, detail = validate_build_sidecars(package, accepted_model)
        return valid, detail
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
            if mode == "discuss":
                complete = studio_complete(args.job, request_id)
                valid, detail = validate_discussion(args.job, request_id) if complete else (False, "Awaiting Studio discussion response")
                if publisher:
                    try:
                        publisher.publish(progress_payload("complete" if valid else "discussing_study", 0, 0, [], mode))
                    except Exception as exc:
                        print(f"[studio-progress] could not publish frontend progress: {exc}", flush=True)
                if valid:
                    write_result(args.job, "studio_turn_complete", True, detail)
                    stop_process(process)
                    output_thread.join(timeout=5)
                    return
                if time.monotonic() >= deadline:
                    stop_process(process)
                    write_result(args.job, "timeout", False, detail)
                    raise SystemExit(f"Agent timed out before the Studio turn completed: {detail}")
                time.sleep(args.check_interval)
                continue
            completed, total, missing = package_progress(package)
            print(f"[package-progress] required={completed}/{len(REQUIRED)} total={total}", flush=True)
            refinement_pending = not studio_complete(args.job, request_id)
            phase = "building_package" if missing or refinement_pending else "validating_package"
            if publisher:
                try:
                    publisher.publish(progress_payload(phase, completed, total, missing, mode))
                except Exception as exc:
                    print(f"[package-progress] could not publish frontend progress: {exc}", flush=True)
            if missing:
                print(f"[package-progress] missing: {' '.join(missing)}", flush=True)
            elif refinement_pending:
                print("[package-progress] waiting for Studio refinement completion", flush=True)
            else:
                ensure_readme(package)
                try:
                    valid, detail = check_output()
                except subprocess.TimeoutExpired:
                    valid, detail = False, "Validator timed out; retrying."
                print(f"[package-progress] validator={'passed' if valid else 'not-ready'} {detail}", flush=True)
                if valid:
                    if publisher:
                        try:
                            publisher.publish(progress_payload("ready_for_review", completed, total, [], mode))
                        except Exception as exc:
                            print(f"[package-progress] could not publish ready state: {exc}", flush=True)
                    write_result(args.job, "validator_passed", True, detail)
                    stop_process(process)
                    output_thread.join(timeout=5)
                    return
            if time.monotonic() >= deadline:
                stop_process(process)
                if missing:
                    valid, detail = False, "Missing required files: " + ", ".join(missing)
                elif not studio_complete(args.job, request_id):
                    valid, detail = False, "Studio refinement did not complete."
                else:
                    valid, detail = check_output()
                write_result(args.job, "timeout", valid, detail)
                if publisher:
                    try:
                        publisher.publish(progress_payload("ready_for_review" if valid else "timed_out", completed, total, missing, mode))
                    except Exception as exc:
                        print(f"[package-progress] could not publish timeout state: {exc}", flush=True)
                if valid:
                    return
                raise SystemExit(f"Agent timed out before the package became reviewable: {detail}")
            time.sleep(args.check_interval)

        output_thread.join(timeout=5)
        if mode == "build":
            ensure_readme(package)
        valid, detail = check_output()
        write_result(args.job, "agent_exited", valid, detail)
        completed, total, missing = package_progress(package) if mode == "build" else (0, 0, [])
        if publisher:
            try:
                publisher.publish(progress_payload(("complete" if mode == "discuss" else "ready_for_review") if valid else "failed", completed, total, missing, mode))
            except Exception as exc:
                print(f"[package-progress] could not publish final state: {exc}", flush=True)
        if not valid:
            raise SystemExit(f"Claude Code exited with {process.returncode}; Studio output validation failed: {detail}")
    except BaseException:
        stop_process(process)
        raise
    finally:
        cleanup_materials(args.job)


if __name__ == "__main__":
    main()
