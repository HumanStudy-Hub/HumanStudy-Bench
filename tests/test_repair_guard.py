"""Fidelity guard for the repair loop, tested on real fixture packages.

Nothing here is mocked: each test copies tests/fixtures/repair_guard/paper to a
pre- and a post-repair folder, edits the post copy the way a repair turn could,
and runs the guard (or the whole run_agent watchdog with a fake `claude` on
PATH) over the files on disk.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent_pipeline.repair_guard import compare_packages, guard_repair, load_package
from agent_pipeline.run_agent import repair_reserve_minutes

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/repair_guard/paper"


def _copies(tmp_path: Path) -> tuple[Path, Path]:
    pre = tmp_path / "pre" / "paper"
    post = tmp_path / "post" / "paper"
    shutil.copytree(FIXTURE, pre)
    shutil.copytree(FIXTURE, post)
    return pre, post


def _edit(path: Path, change) -> None:
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data, indent=2))


def _field(study: dict, node: str, field: str) -> dict:
    nodes = {n["id"]: n for n in study["program"]["nodes"]}
    return next(f for f in nodes[node]["fields"] if f["id"] == field)


def _violations(pre: Path, post: Path) -> list[str]:
    result = guard_repair(pre, post)
    assert result["passed"] is (not result["violations"])
    return result["violations"]


def test_unchanged_package_passes(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    result = guard_repair(pre, post)
    assert result["passed"] is True
    assert result["checked"]["unresolvedItems"] == 4
    assert result["checked"]["verbatimItems"] == 2
    assert result["checked"]["evidenceQuotes"] == 4


def test_allowed_repair_passes(tmp_path: Path) -> None:
    """What the repair prompt asks for: state a derivation, add a review issue,
    add an audit entry, move a confirmed field into missing."""
    pre, post = _copies(tmp_path)

    def repair(study: dict) -> None:
        _field(study, "dv", "scale_mean")["derivation"] = "Mean of the three items (ev_item1)."
        n = _field(study, "participants", "n")
        n.update(state="missing", value=None)
        study["program"]["issues"].append({"id": "i_n", "nodeId": "participants", "fieldId": "n"})
        study["program"]["nodes"].reverse()  # reordering is not a change

    _edit(post / "study.json", repair)
    _edit(post / "audit/missing_information.json", lambda d: d["entries"].append(
        {"study": "study1", "field": "participants.n", "reason": "r", "impact": "i", "suggested_action": "s"}))
    assert _violations(pre, post) == []


def test_missing_field_filled_as_reported_without_evidence_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "study.json", lambda s: _field(s, "participants", "age").update(
        value="18-65", origin="reported", state="confirmed"))
    found = _violations(pre, post)
    assert any("participants.fields.#age: state moved out of missing" in v for v in found)
    assert any("#age: content of an unresolved item was changed" in v for v in found)
    assert any("#age: reported item added or changed by repair cites no evidence ids" in v for v in found)


def test_missing_field_filled_even_with_evidence_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "study.json", lambda s: _field(s, "participants", "age").update(
        value="18-65", origin="reported", state="confirmed", evidenceIds=["ev_n"]))
    assert any("state moved out of missing" in v for v in _violations(pre, post))


def test_decision_field_resolved_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "study.json", lambda s: _field(s, "dv", "items23").update(state="check"))
    assert _violations(pre, post) == ["study.json:program.nodes.#dv.fields.#items23: state moved out of decision (-> check)"]


def test_missing_rule_given_any_derivation_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "study.json", lambda s: s["program"]["steps"][0]["rule"].update(
        value="random", origin="derived", state="confirmed", derivation="standard practice"))
    found = _violations(pre, post)
    assert any("#assign.rule: state moved out of missing" in v for v in found)
    assert any("#assign.rule: derived item added or changed by repair cites no evidence ids" in v for v in found)


def test_missing_field_kept_missing_but_filled_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "study.json", lambda s: _field(s, "participants", "age").update(value="18-65"))
    assert _violations(pre, post) == [
        "study.json:program.nodes.#participants.fields.#age: content of an unresolved item was changed"]


def test_missing_field_deleted_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)

    def drop(study: dict) -> None:
        node = study["program"]["nodes"][0]
        node["fields"] = [f for f in node["fields"] if f["id"] != "age"]
        study["program"]["issues"] = [i for i in study["program"]["issues"] if i.get("fieldId") != "age"]

    _edit(post / "study.json", drop)
    assert "unresolved item was removed" in _violations(pre, post)[0]


def test_verbatim_program_value_rewritten_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "study.json", lambda s: _field(s, "dv", "item1").update(
        value="How likely would you be to delete the app?"))
    assert _violations(pre, post) == ["study.json:program.nodes.#dv.fields.#item1: verbatim text was changed"]


def test_verbatim_material_rewritten_or_relabelled_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "materials/materials.json",
          lambda m: m["shared"]["dependentMeasure"].update(text="How likely would you uninstall the app?"))
    assert _violations(pre, post) == ["materials/materials.json:shared.dependentMeasure: verbatim text was changed"]

    _edit(post / "materials/materials.json", lambda m: m["shared"]["dependentMeasure"].update(
        text="How likely is it that you would uninstall the app?", origin="reported"))
    assert _violations(pre, post) == [
        "materials/materials.json:shared.dependentMeasure: verbatim item was relabelled reported"]


def test_verbatim_metadata_change_passes(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "materials/materials.json", lambda m: m["shared"]["dependentMeasure"].update(
        note="See issue i_items23.", evidenceIds=["ev_item1", "ev_n"]))
    assert _violations(pre, post) == []


def test_missing_material_relabelled_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "materials/materials.json", lambda m: m["shared"]["otherItems"].update(
        evidence_label="reported", text="Would you recommend the app?", evidenceIds=["ev_item1"]))
    found = _violations(pre, post)
    assert "materials/materials.json:shared.otherItems: label moved out of missing (missing -> reported)" in found
    assert "materials/materials.json:shared.otherItems: content of an unresolved item was changed" in found


def test_evidence_quote_changed_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "source/evidence.json", lambda e: e["evidence"][0].update(quote="Participants were 18-65."))
    assert _violations(pre, post) == ["source/evidence.json:evidence.#ev_n: evidence quote was changed"]


def test_audit_entry_removed_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "audit/missing_information.json", lambda a: a["entries"].pop())
    assert _violations(pre, post) == ["audit/missing_information.json: entry for study1 / dv.items23 was removed"]


@pytest.mark.parametrize("ids, expected", [
    ([], "cites no evidence ids"),
    (["ev_invented"], "cites unknown evidence ids ev_invented"),
    (["ev_n"], None),
])
def test_new_reported_field_needs_resolvable_evidence(tmp_path: Path, ids, expected) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "study.json", lambda s: s["program"]["nodes"][0]["fields"].append(
        {"id": "n_female", "label": "Women", "value": 60, "origin": "reported", "state": "confirmed", "evidenceIds": ids}))
    found = _violations(pre, post)
    if expected is None:
        assert found == []
    else:
        assert len(found) == 1 and expected in found[0]


def test_renamed_paper_folder_or_missing_snapshot_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    renamed = post.with_name("other-paper")
    post.rename(renamed)
    assert guard_repair(pre, renamed)["violations"] == ["repair removed or renamed the paper folder"]
    assert guard_repair(pre, None)["passed"] is False
    assert guard_repair(None, renamed)["passed"] is False


def test_compare_packages_is_pure() -> None:
    pre = load_package(FIXTURE)
    before = json.dumps(pre, sort_keys=True)
    assert compare_packages(pre, json.loads(before))["passed"] is True
    assert json.dumps(pre, sort_keys=True) == before


def test_unparseable_json_is_compared_as_text(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    (post / "materials/materials.json").write_text("{not json")
    found = _violations(pre, post)
    assert "materials/materials.json:shared.dependentMeasure: verbatim item was removed" in found


@pytest.mark.parametrize("timeout, configured, env, expected", [
    (25, None, None, 8.0),
    (25, None, "5", 5.0),
    (25, None, "junk", 8.0),
    (10, None, None, 5.0),     # never more than half the budget
    (25, 0.0, "5", 0.0),       # the CLI flag wins over the environment
    (25, -3.0, None, 0.0),
])
def test_repair_reserve(monkeypatch, timeout, configured, env, expected) -> None:
    if env is None:
        monkeypatch.delenv("AGENT_REPAIR_RESERVE_MINUTES", raising=False)
    else:
        monkeypatch.setenv("AGENT_REPAIR_RESERVE_MINUTES", env)
    assert repair_reserve_minutes(timeout, configured) == expected


# --- Whole watchdog, real files, fake `claude` executable --------------------

FAKE_CLAUDE = r'''#!/usr/bin/env python3
import json, os, pathlib, shutil, sys, time
job = pathlib.Path(sys.argv[sys.argv.index('--add-dir') + 1])
root = job / 'package' / 'paper'
(job / 'calls.jsonl').open('a').write(json.dumps(sys.argv[1:-1]) + '\n')

def edit(rel, change):
    path = root / rel
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data, indent=2))

if '--resume' in sys.argv:
    mutation = os.environ['FAKE_REPAIR']
    if mutation == 'fill':
        def fill(study):
            field = study['program']['nodes'][0]['fields'][1]
            field.update(value='18-65', origin='reported', state='confirmed')
        edit('study.json', fill)
        # Tampering with the committed snapshot must not help.
        snap = job / 'pre_repair' / 'paper' / 'study.json'
        if snap.exists():
            snap.write_text((root / 'study.json').read_text())
    elif mutation == 'derive':
        edit('study.json', lambda s: s['program']['nodes'][1]['fields'][1].update(derivation='Mean of the items.'))
    (root / 'repaired.txt').write_text('1')
    raise SystemExit(0)

shutil.copytree(os.environ['FAKE_FIXTURE'], root)
for rel in ('source/paper_metadata.json', 'task/task.json'):
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text('{}\n')
for rel in ('task/adapter.py', 'evaluation/evaluation.py'):
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text('x\n')
if os.environ.get('FAKE_MARKER'):
    (job / 'studio-complete.json').write_text(json.dumps({'requestId': 'req-1', 'status': 'complete'}))
time.sleep(float(os.environ.get('FAKE_BUILD_SLEEP', '0')))
'''

FAKE_VALIDATOR = r'''import pathlib, sys
if (pathlib.Path(sys.argv[1]) / 'paper' / 'repaired.txt').exists():
    print('Validated agent package: paper'); raise SystemExit(0)
raise SystemExit('ValueError: scale_mean: derived value requires derivation')
'''


def _job(tmp_path: Path, repair: str, **env_extra: str) -> tuple[Path, Path, dict]:
    job = tmp_path / "job"
    (job / "logs").mkdir(parents=True)
    (job / "prompt.md").write_text("Build the package")
    (job / "job.json").write_text(json.dumps({"id": "job-1", "status": "running", "repaired": True}))
    validator = tmp_path / "validator.py"
    validator.write_text(FAKE_VALIDATOR)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "claude").write_text(FAKE_CLAUDE)
    (bin_dir / "claude").chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_REPAIR": repair,
           "FAKE_FIXTURE": str(FIXTURE), **env_extra}
    return job, validator, env


def _run(job: Path, validator: Path, env: dict, timeout_minutes: str = "0.5") -> subprocess.CompletedProcess:
    return subprocess.run([
        sys.executable, str(ROOT / "agent_pipeline/run_agent.py"),
        "--job", str(job), "--prompt", str(job / "prompt.md"), "--model", "test-model",
        "--validator", str(validator), "--timeout-minutes", timeout_minutes, "--check-interval", "0.05",
    ], capture_output=True, text=True, env=env, timeout=90)


def test_watchdog_keeps_accepted_repair_and_records_it(tmp_path: Path) -> None:
    job, validator, env = _job(tmp_path, "derive")
    result = _run(job, validator, env)
    assert result.returncode == 0, result.stdout + result.stderr
    state = json.loads((job / "job.json").read_text())
    assert state["repaired"] is True
    assert state["repair"]["accepted"] is True and state["repair"]["guard"]["passed"] is True
    assert state["repair"]["snapshot"] == "pre_repair"
    assert "error" not in state
    post = json.loads((job / "package/paper/study.json").read_text())
    pre = json.loads((job / "pre_repair/paper/study.json").read_text())
    assert _field(post, "dv", "scale_mean")["derivation"] == "Mean of the items."
    assert "derivation" not in _field(pre, "dv", "scale_mean")
    assert json.loads((job / "logs/repair.json").read_text())["outcome"] == "repaired"


def test_watchdog_rejects_repair_that_fills_a_gap(tmp_path: Path) -> None:
    job, validator, env = _job(tmp_path, "fill")
    result = _run(job, validator, env)
    assert result.returncode != 0
    assert "repair rejected by fidelity guard" in result.stderr
    watchdog = json.loads((job / "logs/watchdog.json").read_text())
    assert watchdog["reason"] == "repair_rejected" and watchdog["package_valid"] is False
    state = json.loads((job / "job.json").read_text())
    assert state["repaired"] is False and state["repair"]["accepted"] is False
    assert any("state moved out of missing" in v for v in state["repair"]["guard"]["violations"])
    assert "fidelity guard" in state["error"]
    # The package under review and the committed snapshot are both the pre-repair package,
    # even though the repair turn tried to overwrite the snapshot.
    for path in (job / "package/paper/study.json", job / "pre_repair/paper/study.json"):
        assert _field(json.loads(path.read_text()), "participants", "age")["state"] == "missing"
    assert not (job / "package/paper/repaired.txt").exists()


def test_watchdog_does_not_repair_missing_studio_marker(tmp_path: Path) -> None:
    job, validator, env = _job(tmp_path, "derive")
    (job / "studio_request.json").write_text(json.dumps({"version": 1, "requestId": "req-1"}))
    result = _run(job, validator, env)
    assert result.returncode != 0
    assert "repair not attempted" in result.stderr
    assert len((job / "calls.jsonl").read_text().splitlines()) == 1
    assert json.loads((job / "logs/watchdog.json").read_text())["reason"] == "studio_incomplete"
    assert not (job / "pre_repair").exists()
    state = json.loads((job / "job.json").read_text())
    assert "repaired" not in state and "no repair was attempted" in state["error"]


def test_build_pass_is_capped_to_leave_repair_budget(tmp_path: Path) -> None:
    # 0.3 min total with 0.15 min reserved: the build agent would sleep for 60 s,
    # so without the cap it would use the whole budget and never be repaired.
    job, validator, env = _job(tmp_path, "derive", FAKE_BUILD_SLEEP="60", AGENT_REPAIR_RESERVE_MINUTES="0.15")
    result = _run(job, validator, env, timeout_minutes="0.3")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "build pass reached its cap" in result.stdout
    assert json.loads((job / "logs/watchdog.json").read_text())["reason"] == "repaired"
    assert json.loads((job / "job.json").read_text())["repaired"] is True
