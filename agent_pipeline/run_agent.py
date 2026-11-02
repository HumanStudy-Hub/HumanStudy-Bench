#!/usr/bin/env python3
import argparse
import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    from agent_pipeline.repair_guard import guard_repair, package_diff
    from agent_pipeline.studio_output import studio_mode, validate_build_sidecars, validate_discussion
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from repair_guard import guard_repair, package_diff
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


MAX_REPAIR_ROUNDS = 3
# Minutes of the agent budget held back for the repair loop. The first build pass
# is stopped early enough to leave this much time; at most half the budget.
DEFAULT_REPAIR_RESERVE_MINUTES = 8.0
PRE_REPAIR_DIR = "pre_repair"


def repair_reserve_minutes(timeout_minutes: float, configured: float | None = None) -> float:
    if configured is None:
        raw = os.environ.get("AGENT_REPAIR_RESERVE_MINUTES", "").strip()
        try:
            configured = float(raw) if raw else DEFAULT_REPAIR_RESERVE_MINUTES
        except ValueError:
            configured = DEFAULT_REPAIR_RESERVE_MINUTES
    return max(0.0, min(configured, timeout_minutes / 2))


def snapshot_package(package: Path) -> Path:
    """Copy the package as it stood before repair to a private directory outside
    the job folder. The repair agent can write anywhere in the job folder, so the
    copy the guard trusts must not live there."""
    snapshot = Path(tempfile.mkdtemp(prefix="hs-pre-repair-")) / "package"
    shutil.copytree(package, snapshot)
    return snapshot


def publish_snapshot(snapshot: Path, job: Path) -> None:
    """Write the trusted pre-repair copy to <job>/pre_repair/, replacing anything
    the agent put there, so it is committed with the job branch for reviewers."""
    target = job / PRE_REPAIR_DIR
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(snapshot, target)


def restore_package(snapshot: Path, package: Path) -> None:
    shutil.rmtree(package, ignore_errors=True)
    shutil.copytree(snapshot, package)


def update_job_state(job: Path, **fields) -> None:
    """Merge fields into job.json, the state the web review UI reads.
    A value of None removes the key."""
    path = job / "job.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(data, dict):
        return
    for key, value in fields.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    path.write_text(json.dumps(data, indent=2) + "\n")

# Validator messages mapped to the rule in agent_pipeline/CLAUDE.md they enforce
# and the only acceptable repairs. A repair never fills a gap with new content.
REPAIR_RULES = (
    ("derived value requires derivation",
     "Evidence labels: `derived` means transformed from reported information with the "
     "derivation stated. Add the actual derivation (which reported values, what "
     "transformation, which evidence ids). If you cannot state one from the sources, "
     "relabel the field: `verbatim`/`reported` only if the source states the value as is, "
     "otherwise `missing` with a review issue. Do not invent a derivation."),
    ("requires a review issue",
     "Evidence labels: `missing` content requires researcher input and is never silently "
     "filled. Keep the field or rule unresolved and add a review issue to "
     "study.json program.issues that points at it (nodeId + fieldId, or stepId), and the "
     "matching entry in audit/missing_information.json (study, field, reason, impact, "
     "suggested_action). Do not replace the gap with plausible text."),
    ("disagree",
     "studio-model.json is the Studio copy of study.json program; after a repair write the "
     "same program to both files."),
    ("programBindings",
     "task/task.json programBindings must be derived from the canonical program in "
     "study.json: programId, stepIds, recordNodeIds (record and variable nodes) and "
     "analysisNodeIds, in program order. Change the bindings, not the program."),
)


def program_findings(package: Path) -> list[str]:
    """List every field and rule that breaks the two label rules the validator
    stops on, so one repair turn can fix all of them, not just the first."""
    root = package_root(package)
    try:
        program = json.loads((root / "study.json").read_text())["program"]  # type: ignore[operator]
        nodes, steps, issues = program["nodes"], program["steps"], program["issues"]
    except Exception:
        return []
    findings: list[str] = []

    def derived(field: object, where: str) -> None:
        if isinstance(field, dict) and field.get("origin") == "derived" and not str(field.get("derivation", "")).strip():
            findings.append(f"{where}: derived value requires derivation")

    for node in nodes if isinstance(nodes, list) else []:
        if not isinstance(node, dict):
            continue
        for field in node.get("fields", []) if isinstance(node.get("fields"), list) else []:
            if not isinstance(field, dict):
                continue
            where = f"node {node.get('id')} field {field.get('id')}"
            derived(field, where)
            if field.get("state") in ("missing", "decision") and not any(
                isinstance(i, dict) and i.get("nodeId") == node.get("id")
                and (not i.get("fieldId") or i.get("fieldId") == field.get("id")) for i in issues
            ):
                findings.append(f"{where}: unresolved field requires a review issue")
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, dict) or not isinstance(step.get("rule"), dict):
            continue
        where = f"step {step.get('id')} rule"
        derived(step["rule"], where)
        if step["rule"].get("state") in ("missing", "decision") and not any(
            isinstance(i, dict) and i.get("stepId") == step.get("id") for i in issues
        ):
            findings.append(f"{where}: unresolved rule requires a review issue")
    return findings


def error_summary(detail: str, limit: int = 4000) -> str:
    """Keep the validator's own message; drop traceback frames."""
    lines = [line for line in detail.strip().splitlines() if line.strip()]
    if any(line.startswith("Traceback") for line in lines):
        lines = [line for line in lines if not line.startswith((" ", "Traceback"))] or lines[-1:]
    text = "\n".join(lines)
    return text if len(text) <= limit else text[:limit] + "\n[truncated]"


def repair_prompt(job: Path, detail: str, findings: list[str], round_number: int, max_rounds: int) -> str:
    errors = error_summary(detail)
    text = errors + "\n" + "\n".join(findings)
    rules = [rule for key, rule in REPAIR_RULES if key in text]
    lines = [
        f"Repair round {round_number} of {max_rounds}. The study package in {job / 'package'} is "
        "almost done but failed the pipeline validator. Fix only what the validator reports; "
        "do not rebuild or restructure the package.",
        "",
        "Validator error:",
        "```",
        errors,
        "```",
    ]
    if findings:
        lines += ["", "Every place in study.json program with the same problem:"] + [f"- {f}" for f in findings]
    if rules:
        lines += ["", "Rules from agent_pipeline/CLAUDE.md that apply:"] + [f"- {rule}" for rule in rules]
    lines += [
        "",
        "The scientific contract in agent_pipeline/CLAUDE.md still holds: never fill a gap with "
        "invented content. A missing value stays `missing` with a review issue; a `derived` value "
        "gets its stated derivation or is relabelled. Keep study.json program, "
        "audit/missing_information.json, source/evidence.json and task/task.json programBindings "
        "consistent with each other.",
        "A deterministic check compares the package before and after this repair and discards the "
        "repair if it: resolves, removes or edits the value, extensions, derivation or note of any "
        "missing/decision item; changes or relabels any verbatim text; adds, edits or removes an "
        "evidence record; edits or removes an audit entry; or adds or changes a value that is not "
        "verbatim/reported/derived with existing evidence ids.",
        f"Re-run `python agent_pipeline/validate_package.py {job / 'package'}` until it prints "
        "\"Validated agent package\".",
    ]
    if (job / "studio_request.json").is_file():
        lines += [
            "This is a Studio build: studio-model.json in the paper folder must carry the same program as "
            "study.json program (write the repaired program to both), studio-reply.md in the paper "
            "folder must be non-empty, and studio-complete.json must stay in place. Check the "
            "sidecars with `python -c \"import sys; from pathlib import Path; "
            "from agent_pipeline.studio_output import validate_build_sidecars as v; "
            f"ok, d = v(Path(sys.argv[1])); print(d); sys.exit(0 if ok else 1)\" {job / 'package'}`. "
            "Stop as soon as both checks pass.",
        ]
    else:
        lines += ["Stop as soon as it passes."]
    return "\n".join(lines) + "\n"


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


def progress_payload(phase: str, completed: int, total: int, missing: list[str], mode: str = "build",
                     repair: dict | None = None) -> dict:
    payload = {
        "phase": phase,
        "completedRequired": completed,
        "totalRequired": 0 if mode == "discuss" else len(REQUIRED),
        "totalFiles": total,
        "missing": missing,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }
    if repair is not None:
        payload["repair"] = repair
    return payload


def repair_package(
    job: Path,
    package: Path,
    check_output,
    run_turn,
    deadline: float,
    detail: str,
    max_rounds: int = MAX_REPAIR_ROUNDS,
    publish=None,
) -> tuple[bool, str, list[dict]]:
    """Send validator errors back to the agent for at most ``max_rounds`` turns.

    ``run_turn(prompt, round_number, deadline)`` runs one agent turn and returns
    once it exits, the package validates, or the deadline passes. Every round is
    appended to logs/repair.json.
    """
    rounds: list[dict] = []
    valid = False
    for number in range(1, max_rounds + 1):
        if time.monotonic() >= deadline:
            print("[package-repair] no time left for another repair round", flush=True)
            break
        errors = error_summary(detail)
        findings = program_findings(package)
        print(f"[package-repair] round {number}/{max_rounds}; validator error:\n{errors}", flush=True)
        for finding in findings:
            print(f"[package-repair]   {finding}", flush=True)
        if publish:
            publish(number, max_rounds, errors)
        started = datetime.now(timezone.utc).isoformat()
        returncode = run_turn(repair_prompt(job, detail, findings, number, max_rounds), number, deadline)
        try:
            valid, detail = check_output()
        except subprocess.TimeoutExpired:
            valid, detail = False, "Validator timed out."
        rounds.append({
            "round": number,
            "startedAt": started,
            "finishedAt": datetime.now(timezone.utc).isoformat(),
            "agentExitCode": returncode,
            "errorsBefore": errors,
            "findingsBefore": findings,
            "valid": valid,
            "errorsAfter": "" if valid else error_summary(detail),
        })
        (job / "logs").mkdir(parents=True, exist_ok=True)
        (job / "logs/repair.json").write_text(json.dumps({"maxRounds": max_rounds, "rounds": rounds}, indent=2) + "\n")
        print(f"[package-repair] round {number} validator={'passed' if valid else 'not-ready'}", flush=True)
        if valid:
            break
    return valid, detail, rounds


def run_guarded_repair(job: Path, package: Path, run_repair) -> tuple[bool, str, str]:
    """Snapshot the package, run ``run_repair()`` (which returns repair_package's
    result), then guard the result. The package under review is either the
    accepted repair or exactly the pre-repair package: if the guard rejects the
    repair, the repaired package still fails validation, or the repair is
    interrupted by an error or cancellation, the snapshot is restored. Returns (valid, detail, reason)."""
    snapshot = snapshot_package(package)
    update_job_state(job, repaired=False, repair={"attempted": True, "inProgress": True, "snapshot": PRE_REPAIR_DIR})
    rounds: list[dict] = []
    guard: dict = {"passed": False, "violations": [], "checked": {}, "changedFiles": []}
    interrupted: BaseException | None = None
    main_thread = threading.current_thread() is threading.main_thread()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)} if main_thread else {}

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    try:
        if main_thread:
            # A cancelled Actions run sends SIGINT then SIGTERM; both must unwind
            # through the restore below instead of killing the process mid-repair.
            signal.signal(signal.SIGTERM, terminate)
        try:
            valid, detail, rounds = run_repair()
            reason = "repaired" if valid else "repair_failed"
            guard = guard_repair(package_root(snapshot), package_root(package))
        except BaseException as exc:
            interrupted = exc
            try:
                rounds = json.loads((job / "logs/repair.json").read_text())["rounds"]
            except (OSError, ValueError, KeyError, TypeError):
                pass
            valid, reason = False, "repair_interrupted"
            detail = f"Repair was interrupted ({type(exc).__name__}: {exc}); the pre-repair package was kept."
            guard["violations"] = [detail]
        if main_thread:
            # Bookkeeping is short; finish it so the package is never half restored.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        for violation in guard["violations"]:
            print(f"[package-repair] guard: {violation}", flush=True)
        if not guard["passed"] and reason != "repair_interrupted":
            valid, reason = False, "repair_rejected"
            detail = ("Repair rejected by the fidelity guard; the pre-repair package was kept: "
                      + "; ".join(guard["violations"][:20]))
        (job / "logs").mkdir(parents=True, exist_ok=True)
        if root := package_root(snapshot):
            post = package_root(package)
            if post is not None and post.name == root.name:
                (job / "logs/repair.diff").write_text(package_diff(root, post))
        if not valid:
            restore_package(snapshot, package)
            ensure_readme(package)
        publish_snapshot(snapshot, job)
        (job / "logs/repair.json").write_text(json.dumps(
            {"maxRounds": MAX_REPAIR_ROUNDS, "rounds": rounds, "outcome": reason, "guard": guard}, indent=2) + "\n")
        update_job_state(job, repaired=valid, error=None if valid else error_summary(detail, 1000), repair={
            "attempted": bool(rounds),
            "accepted": valid,
            "outcome": reason,
            "rounds": len(rounds),
            "snapshot": PRE_REPAIR_DIR,
            "guard": {**guard, "violations": guard["violations"][:50], "violationCount": len(guard["violations"])},
        })
        if interrupted is not None:
            raise interrupted
        return valid, detail, reason
    finally:
        shutil.rmtree(snapshot.parent, ignore_errors=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--prompt", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--validator", required=True, type=Path)
    parser.add_argument("--timeout-minutes", type=float, default=25)
    parser.add_argument("--repair-reserve-minutes", type=float, default=None,
                        help="Budget held back for repair (default: $AGENT_REPAIR_RESERVE_MINUTES or 8).")
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
        # A re-run must not inherit an earlier run's repair snapshot or verdict.
        shutil.rmtree(args.job / PRE_REPAIR_DIR, ignore_errors=True)
        for stale in ("logs/repair.json", "logs/repair.diff"):
            (args.job / stale).unlink(missing_ok=True)
        update_job_state(args.job, repaired=None, repair=None, error=None)

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
    session_id = str(uuid.uuid4())
    base_command = [
        "claude",
        "--print",
        "--model",
        args.model,
        "--add-dir",
        str(args.job),
        "--dangerously-skip-permissions",
    ]
    command = base_command + ["--session-id", session_id, args.prompt.read_text()]
    claude_env = os.environ.copy()
    claude_env.pop("PIPELINE_PROGRESS_TOKEN", None)
    claude_env.pop("HUMANSTUDY_PIPELINE_TOKEN", None)
    claude_env.pop("GITHUB_TOKEN", None)
    repo_root = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        command,
        cwd=repo_root,
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
    reserve = repair_reserve_minutes(args.timeout_minutes, args.repair_reserve_minutes) if mode == "build" else 0.0
    build_deadline = deadline - reserve * 60

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
            now = time.monotonic()
            if build_deadline <= now < deadline and not missing and not refinement_pending:
                # The package is complete but not valid and the first pass has used
                # its share: stop it so the reserved time goes to repair. An
                # incomplete package keeps building until the full deadline.
                stop_process(process)
                output_thread.join(timeout=5)
                print(f"[package-progress] build pass reached its cap; {reserve:g} min reserved for repair", flush=True)
                break
            if now >= deadline:
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
        reason = "agent_exited"
        if not valid and mode == "build" and not studio_complete(args.job, request_id):
            # The marker is only the agent's own claim that the Studio turn is
            # done; a repair turn could write it without doing the work. Never
            # repair on it alone.
            reason = "studio_incomplete"
            update_job_state(args.job, error="The agent stopped before completing the Studio request; no repair was attempted.")
        elif not valid and mode == "build":
            def publish_repair(number: int, max_rounds: int, errors: str) -> None:
                if not publisher:
                    return
                completed, total, missing = package_progress(package)
                try:
                    publisher.publish(progress_payload("validating_package", completed, total, missing, mode,
                                                       {"round": number, "maxRounds": max_rounds, "error": errors[:500]}))
                except Exception as exc:
                    print(f"[package-repair] could not publish frontend progress: {exc}", flush=True)

            def run_turn(prompt: str, number: int, turn_deadline: float) -> int | None:
                log_path = logs / f"agent-repair-{number}.log"
                attempts = [base_command + ["--resume", session_id, prompt],
                            base_command + ["--session-id", str(uuid.uuid4()), prompt]]
                returncode = None
                for attempt, turn_command in enumerate(attempts):
                    started = time.monotonic()
                    turn = subprocess.Popen(turn_command, cwd=repo_root, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                                            start_new_session=True, env=claude_env)
                    turn_log = log_path.with_name(f"{log_path.stem}-fresh.log") if attempt else log_path
                    reader = threading.Thread(target=stream_output, args=(turn, turn_log), daemon=True)
                    reader.start()
                    try:
                        while turn.poll() is None:
                            if time.monotonic() >= turn_deadline:
                                print("[package-repair] time budget reached during repair", flush=True)
                                break
                            time.sleep(args.check_interval)
                            if turn.poll() is None and package_progress(package)[2] == []:
                                try:
                                    if check_output()[0]:
                                        print("[package-repair] validator passed; stopping repair turn", flush=True)
                                        break
                                except subprocess.TimeoutExpired:
                                    pass
                    finally:
                        stop_process(turn)
                        reader.join(timeout=5)
                    returncode = turn.returncode
                    # A resume that fails at once (no stored session) falls back to a fresh turn
                    # with the same self-contained repair prompt.
                    if attempt == 0 and returncode not in (0, None) and time.monotonic() - started < 30 \
                            and time.monotonic() < turn_deadline:
                        print(f"[package-repair] resume exited with {returncode}; starting a fresh repair turn", flush=True)
                        continue
                    break
                ensure_readme(package)
                return returncode

            valid, detail, reason = run_guarded_repair(
                args.job, package, lambda: repair_package(args.job, package, check_output, run_turn, deadline,
                                                          detail, publish=publish_repair))
        write_result(args.job, reason, valid, detail)
        completed, total, missing = package_progress(package) if mode == "build" else (0, 0, [])
        if publisher:
            try:
                publisher.publish(progress_payload(("complete" if mode == "discuss" else "ready_for_review") if valid else "failed", completed, total, missing, mode))
            except Exception as exc:
                print(f"[package-progress] could not publish final state: {exc}", flush=True)
        if not valid:
            suffix = {"repair_failed": " after repair rounds", "repair_rejected": " (repair rejected by fidelity guard)",
                      "repair_interrupted": " (repair interrupted)",
                      "studio_incomplete": " (Studio request not completed; repair not attempted)"}.get(reason, "")
            raise SystemExit(f"Claude Code exited with {process.returncode}; Studio output validation failed{suffix}: {detail}")
    except BaseException:
        stop_process(process)
        raise
    finally:
        cleanup_materials(args.job)


if __name__ == "__main__":
    main()
