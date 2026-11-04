"""Fidelity guard for the repair loop, tested on real fixture packages.

Nothing here is mocked: each test copies tests/fixtures/repair_guard/paper to a
pre- and a post-repair folder, edits the post copy the way a repair turn could,
and runs the guard (or the whole run_agent watchdog with a fake `claude` on
PATH) over the files on disk. The fixture's program is schema-valid Human
Program v2 except for the one validator error a repair is meant to fix: a
`derived` field with no derivation.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent_pipeline.human_program import validate_human_program
from agent_pipeline.repair_guard import compare_packages, guard_repair, load_package, package_diff
from agent_pipeline.run_agent import repair_reserve_minutes, run_guarded_repair

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/repair_guard/paper"
AGE = "study.json:program.nodes.#participants.fields.#age"


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


def _derive(study: dict) -> None:
    _field(study, "dv", "scale_mean")["derivation"] = "Mean of the three uninstall items."


# --- What a repair is allowed to do -----------------------------------------

def test_fixture_fails_validation_only_on_the_derivation() -> None:
    study = json.loads((FIXTURE / "study.json").read_text())
    with pytest.raises(ValueError, match="derived value requires derivation"):
        validate_human_program(study["program"])
    _derive(study)
    validate_human_program(study["program"])


def test_unchanged_package_passes(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    result = guard_repair(pre, post)
    assert result["passed"] is True and result["changedFiles"] == []
    assert result["checked"]["unresolvedItems"] == 4
    assert result["checked"]["verbatimItems"] == 5
    assert result["checked"]["evidenceRecords"] == 4
    assert result["checked"]["auditEntries"] == 2


def test_stating_a_derivation_passes_even_without_evidence_ids(tmp_path: Path) -> None:
    """The repair the validator asks for most often; the field has no evidence ids."""
    pre, post = _copies(tmp_path)
    _edit(post / "study.json", _derive)
    assert _field(json.loads((post / "study.json").read_text()), "dv", "scale_mean")["evidenceIds"] == []
    assert _violations(pre, post) == []


def test_allowed_repair_passes(tmp_path: Path) -> None:
    """Add a review issue and audit entry, move a confirmed field into missing,
    reorder nodes, and annotate a verbatim item."""
    pre, post = _copies(tmp_path)

    def repair(study: dict) -> None:
        _derive(study)
        _field(study, "participants", "n").update(state="missing", value=None)
        study["program"]["issues"].append({"id": "i_n", "nodeId": "participants", "fieldId": "n", "title": "n",
                                           "severity": "check", "reason": "r", "impact": "i",
                                           "suggestedAction": "s", "evidenceIds": []})
        study["program"]["nodes"].reverse()

    _edit(post / "study.json", repair)
    validate_human_program(json.loads((post / "study.json").read_text())["program"])
    _edit(post / "audit/missing_information.json", lambda d: d["entries"].append(
        {"study": "study1", "field": "participants.n", "reason": "r", "impact": "i", "suggested_action": "s"}))
    _edit(post / "materials/materials.json", lambda m: m["shared"]["dependentMeasure"].update(
        note="See issue i_items23.", evidenceIds=["ev_item1", "ev_n"]))
    assert _violations(pre, post) == []


def test_fixing_origin_missing_to_schema_value_passes(tmp_path: Path) -> None:
    """`origin: missing` is not a v2 origin; the fix is origin unknown, state missing."""
    pre, post = _copies(tmp_path)
    _edit(pre / "study.json", lambda s: _field(s, "participants", "age").update(origin="missing"))
    _edit(post / "study.json", lambda s: _field(s, "participants", "age").update(origin="unknown"))
    assert _violations(pre, post) == []


def test_studio_model_sync_is_not_compared_field_by_field(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    stale = json.loads((pre / "study.json").read_text())["program"]
    _field({"program": stale}, "participants", "n").update(state="missing")
    (pre / "studio-model.json").write_text(json.dumps({"program": stale}))
    shutil.copy(post / "study.json", post / "studio-model.json")
    assert _violations(pre, post) == []


def test_inserting_into_an_id_less_list_does_not_misalign(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "materials/materials.json", lambda m: m["instructions"].insert(
        0, {"origin": "reported", "text": "Participants were told the study takes ten minutes.",
            "evidenceIds": ["ev_n"]}))
    assert _violations(pre, post) == []


# --- Gaps must stay gaps ----------------------------------------------------

def test_missing_field_filled_as_reported_without_evidence_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "study.json", lambda s: _field(s, "participants", "age").update(
        value="18-65", origin="reported", state="confirmed"))
    found = _violations(pre, post)
    assert f"{AGE}: moved out of missing/decision (state missing -> confirmed, origin unknown -> reported)" in found
    assert f"{AGE}: value of an unresolved item was changed" in found
    assert f"{AGE}: reported item added or changed by repair cites no evidence ids" in found


def test_missing_field_filled_even_with_evidence_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "study.json", lambda s: _field(s, "participants", "age").update(
        value="18-65", origin="reported", state="confirmed", evidenceIds=["ev_n"]))
    assert any("moved out of missing/decision" in v for v in _violations(pre, post))


def test_decision_field_resolved_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "study.json", lambda s: _field(s, "dv", "items23").update(state="check"))
    assert _violations(pre, post) == [
        "study.json:program.nodes.#dv.fields.#items23: moved out of missing/decision "
        "(state decision -> check, origin unknown -> unknown)",
    ]


def test_missing_rule_given_any_derivation_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "study.json", lambda s: s["program"]["steps"][0]["rule"].update(
        value="random", origin="derived", state="confirmed", derivation="standard practice", evidenceIds=["ev_n"]))
    found = _violations(pre, post)
    assert any("#assign.rule: moved out of missing/decision" in v for v in found)
    assert any("#assign.rule: derivation was added to an unresolved item" in v for v in found)


@pytest.mark.parametrize("smuggle, message", [
    ({"value": "18-65"}, "value of an unresolved item was changed"),
    ({"extensions": {"assumed": "18-65"}}, "value of an unresolved item was changed"),
    ({"derivation": "Typical student sample, 18-65."}, "derivation was added to an unresolved item"),
    ({"note": "Probably 18-65."}, "note of an unresolved item was changed"),
])
def test_missing_field_kept_missing_but_filled_fails(tmp_path: Path, smuggle, message) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "study.json", lambda s: _field(s, "participants", "age").update(smuggle))
    assert _violations(pre, post) == [f"{AGE}: {message}"]


def test_missing_field_deleted_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)

    def drop(study: dict) -> None:
        node = study["program"]["nodes"][0]
        node["fields"] = [f for f in node["fields"] if f["id"] != "age"]

    _edit(post / "study.json", drop)
    assert _violations(pre, post) == [f"{AGE}: unresolved item was removed or changed"]


def test_missing_material_relabelled_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "materials/materials.json", lambda m: m["shared"]["otherItems"].update(
        evidence_label="reported", text="Would you recommend the app?", evidenceIds=["ev_item1"]))
    found = _violations(pre, post)
    assert any("shared.otherItems: moved out of missing/decision" in v for v in found)
    assert "materials/materials.json:shared.otherItems: value of an unresolved item was changed" in found


@pytest.mark.parametrize("origin", ["researcher", "implementation", "unknown"])
def test_new_value_without_evidence_origin_fails(tmp_path: Path, origin) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "study.json", lambda s: s["program"]["nodes"][0]["fields"].append(
        {"id": "age2", "label": "Age", "value": "18-65", "origin": origin, "state": "confirmed", "evidenceIds": []}))
    assert _violations(pre, post) == [
        f"study.json:program.nodes.#participants.fields.#age2: repair introduced a resolved value with origin "
        f"{origin}; only evidence-backed values may be added"]


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


def test_evidence_ids_removed_from_reported_field_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "study.json", lambda s: _field(s, "participants", "n").update(evidenceIds=[]))
    assert _violations(pre, post) == [
        "study.json:program.nodes.#participants.fields.#n: evidence ids were removed from a reported item"]


# --- Transcribed text must not change ---------------------------------------

def test_verbatim_program_value_rewritten_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "study.json", lambda s: _field(s, "dv", "item1").update(
        value="How likely would you be to delete the app?"))
    found = _violations(pre, post)
    assert "study.json:program.nodes.#dv.fields.#item1: verbatim text was changed" in found


def test_verbatim_material_rewritten_or_relabelled_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "materials/materials.json",
          lambda m: m["shared"]["dependentMeasure"].update(text="How likely would you uninstall the app?"))
    assert "materials/materials.json:shared.dependentMeasure: verbatim text was changed" in _violations(pre, post)

    shutil.copy(pre / "materials/materials.json", post / "materials/materials.json")
    _edit(post / "materials/materials.json", lambda m: m["shared"]["dependentMeasure"].update(origin="reported"))
    assert _violations(pre, post) == [
        "materials/materials.json:shared.dependentMeasure: verbatim item was relabelled reported"]


def test_nested_verbatim_text_rewritten_fails(tmp_path: Path) -> None:
    """Scale anchors live in nested `label` keys; those are text, not metadata."""
    pre, post = _copies(tmp_path)
    _edit(post / "materials/materials.json",
          lambda m: m["shared"]["responseScale"]["options"][1].update(label="Very likely"))
    assert _violations(pre, post) == ["materials/materials.json:shared.responseScale: verbatim text was changed"]


def test_id_less_verbatim_item_rewritten_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "materials/materials.json",
          lambda m: m["instructions"][1].update(text="Please answer all questions."))
    found = _violations(pre, post)
    assert any(v.startswith("materials/materials.json:instructions.~") and "verbatim item was removed or changed" in v
               for v in found)


# --- Evidence and the audit checklist ----------------------------------------

def test_evidence_quote_changed_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "source/evidence.json", lambda e: e["evidence"][0].update(quote="Participants were 18-65."))
    assert _violations(pre, post) == ["source/evidence.json:evidence.#ev_n: evidence record was changed (quote)"]


def test_fabricated_evidence_cited_by_new_field_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    record = {"id": "ev_age", "path": "source/paper.pdf", "locator": {"page": 2}, "quote": "Ages ranged 18-65."}
    _edit(post / "source/evidence.json", lambda e: e["evidence"].append(record))
    _edit(post / "study.json", lambda s: s["program"]["evidence"].append(record))
    assert _violations(pre, post) == [
        "source/evidence.json:evidence.#ev_age: repair added a new evidence record",
        "study.json:program.evidence.#ev_age: repair added a new evidence record",
    ]


def test_copying_an_existing_evidence_record_passes(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(pre / "source/evidence.json", lambda e: e["evidence"].pop())
    assert _violations(pre, post) == []


@pytest.mark.parametrize("change, message", [
    (lambda a: a["entries"].pop(), "entry for study1 / dv.items23 was removed"),
    (lambda a: a["entries"][1].update(reason="Resolved: items supplied."), "entry for study1 / dv.items23 was changed"),
])
def test_audit_entry_removed_or_rewritten_fails(tmp_path: Path, change, message) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "audit/missing_information.json", change)
    assert _violations(pre, post) == [f"audit/missing_information.json: {message}"]


@pytest.mark.parametrize("container", ["records", None])
def test_evidence_records_outside_an_evidence_list_are_guarded(tmp_path: Path, container) -> None:
    pre, post = _copies(tmp_path)
    for root in (pre, post):
        records = json.loads((root / "source/evidence.json").read_text())["evidence"]
        (root / "source/evidence.json").write_text(json.dumps({container: records} if container else records))
    data = json.loads((post / "source/evidence.json").read_text())
    records = data[container] if container else data
    records[0]["quote"] = "Participants were 18-65."
    records.append({"id": "ev_new", "quote": "Invented."})
    (post / "source/evidence.json").write_text(json.dumps(data))
    found = _violations(pre, post)
    assert any("#ev_n: evidence record was changed (quote)" in v for v in found)
    assert any("#ev_new: repair added a new evidence record" in v for v in found)


def test_new_unresolved_item_carrying_a_value_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    _edit(post / "study.json", lambda s: s["program"]["nodes"][0]["fields"].append(
        {"id": "age2", "label": "Age", "value": "18-65", "origin": "unknown", "state": "decision", "evidenceIds": []}))
    assert _violations(pre, post) == [
        "study.json:program.nodes.#participants.fields.#age2: repair added an unresolved item that already "
        "carries a value"]


def test_verbatim_option_label_rewritten_fails(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    for root in (pre, post):
        _edit(root / "materials/materials.json", lambda m: m.update(options=[
            {"id": "o1", "origin": "verbatim", "label": "Strongly agree", "evidenceIds": ["ev_item1"]}]))
    _edit(post / "materials/materials.json", lambda m: m["options"][0].update(label="Agree"))
    assert _violations(pre, post) == ["materials/materials.json:options.#o1: verbatim text was changed"]


def test_legacy_studio_sidecar_is_guarded(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    legacy = {"id": "m", "title": "t", "fields": [{"id": "age", "status": "unresolved", "value": None}]}
    (pre / "studio-model.json").write_text(json.dumps(legacy))
    legacy["fields"][0].update(status="reported", value="18-65")
    (post / "studio-model.json").write_text(json.dumps(legacy))
    assert any("studio-model.json:fields.#age: moved out of missing/decision" in v for v in _violations(pre, post))


# --- Files the guard cannot read -------------------------------------------

def test_unparseable_file_changed_by_repair_fails_closed(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    (pre / "study.json").write_text((pre / "study.json").read_text()[:200])
    _edit(post / "study.json", lambda s: _field(s, "participants", "age").update(
        value="18-65", origin="reported", state="confirmed", evidenceIds=["ev_n"]))
    assert _violations(pre, post) == [
        "study.json: was not valid JSON before repair and was changed; the repair cannot be verified"]


def test_unparseable_file_left_alone_passes(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    (pre / "extra.json").write_text("{not json")
    (post / "extra.json").write_text("{not json")
    assert _violations(pre, post) == []


def test_non_json_changes_are_listed_and_diffed(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    (pre / "task").mkdir()
    (pre / "task/adapter.py").write_text("AGE = None\n")
    (post / "task").mkdir()
    (post / "task/adapter.py").write_text("AGE = '18-65'\n")
    result = guard_repair(pre, post)
    assert result["passed"] is True and result["changedFiles"] == ["task/adapter.py"]
    diff = package_diff(pre, post)
    assert "-AGE = None" in diff and "+AGE = '18-65'" in diff


def test_paper_folder_problems_fail(tmp_path: Path) -> None:
    pre, post = _copies(tmp_path)
    renamed = post.with_name("other-paper")
    post.rename(renamed)
    assert guard_repair(pre, renamed)["violations"] == ["repair renamed the paper folder paper to other-paper"]
    assert "after repair" in guard_repair(pre, None)["violations"][0]
    assert "before repair" in guard_repair(None, renamed)["violations"][0]


def test_compare_packages_is_pure() -> None:
    pre = load_package(FIXTURE)
    before = json.dumps(pre, sort_keys=True)
    assert compare_packages(pre, json.loads(before))["passed"] is True
    assert json.dumps(pre, sort_keys=True) == before


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


# --- run_guarded_repair on real folders --------------------------------------

def _job_dir(tmp_path: Path) -> tuple[Path, Path]:
    job = tmp_path / "job"
    package = job / "package"
    shutil.copytree(FIXTURE, package / "paper")
    (job / "logs").mkdir()
    (job / "job.json").write_text(json.dumps({"id": "job-1", "status": "running"}))
    return job, package


def test_interrupted_repair_restores_the_snapshot(tmp_path: Path) -> None:
    job, package = _job_dir(tmp_path)

    def run_repair():
        _edit(package / "paper/study.json", lambda s: _field(s, "participants", "age").update(state="confirmed"))
        raise KeyboardInterrupt("cancelled")

    with pytest.raises(KeyboardInterrupt):
        run_guarded_repair(job, package, run_repair)
    assert (package / "paper/study.json").read_text() == (FIXTURE / "study.json").read_text()
    assert (job / "pre_repair/paper/study.json").read_text() == (FIXTURE / "study.json").read_text()
    state = json.loads((job / "job.json").read_text())
    assert state["repaired"] is False and state["repair"]["outcome"] == "repair_interrupted"
    assert "inProgress" not in state["repair"]


def test_interrupt_kills_running_turn_and_counts_it(tmp_path: Path) -> None:
    job, package = _job_dir(tmp_path)
    turns: list = []

    def run_repair():
        turns.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                      start_new_session=True))
        _edit(package / "paper/study.json", _derive)
        (job / "logs/repair.json").write_text(json.dumps({"rounds": "not a list"}))
        raise RuntimeError("runner lost")

    with pytest.raises(RuntimeError):
        run_guarded_repair(job, package, run_repair, turns)
    assert turns[0].poll() is not None
    assert (package / "paper/study.json").read_text() == (FIXTURE / "study.json").read_text()
    repair = json.loads((job / "job.json").read_text())["repair"]
    assert repair["attempted"] is True and repair["turnsStarted"] == 1 and repair["rounds"] == 0
    assert repair["outcome"] == "repair_interrupted"


def test_guard_passing_but_still_invalid_repair_is_rolled_back(tmp_path: Path) -> None:
    job, package = _job_dir(tmp_path)

    def run_repair():
        _edit(package / "paper/study.json", _derive)
        return False, "ValueError: something else", [{"round": 1}]

    valid, detail, reason = run_guarded_repair(job, package, run_repair)
    assert (valid, reason) == (False, "repair_failed")
    assert (package / "paper/study.json").read_text() == (FIXTURE / "study.json").read_text()
    assert "+" in (job / "logs/repair.diff").read_text()
    state = json.loads((job / "job.json").read_text())
    assert state["repair"]["attempted"] is True and state["repair"]["guard"]["passed"] is True


# --- Whole watchdog, real files, fake `claude` executable --------------------

FAKE_CLAUDE = r'''#!/usr/bin/env python3
import json, os, pathlib, shutil, sys, time
job = pathlib.Path(sys.argv[sys.argv.index('--add-dir') + 1])
root = job / 'package' / 'paper'
(job / 'calls.jsonl').open('a').write(json.dumps(sys.argv[1:-1]) + '\n')

def field(study, node, ident):
    return next(f for n in study['program']['nodes'] if n['id'] == node for f in n['fields'] if f['id'] == ident)

def edit(rel, change):
    path = root / rel
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data, indent=2))

if '--resume' in sys.argv:
    mutation = os.environ['FAKE_REPAIR']
    if mutation == 'fill':
        edit('study.json', lambda s: field(s, 'participants', 'age').update(
            value='18-65', origin='reported', state='confirmed'))
        # Tampering with the committed snapshot must not help.
        snap = job / 'pre_repair' / 'paper' / 'study.json'
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text((root / 'study.json').read_text())
    elif mutation == 'derive':
        edit('study.json', lambda s: field(s, 'dv', 'scale_mean').update(derivation='Mean of the items.'))
    raise SystemExit(0)

shutil.copytree(os.environ['FAKE_FIXTURE'], root)
if os.environ.get('FAKE_REPLY'):
    (root / 'studio-reply.md').write_text(os.environ['FAKE_REPLY'])
for rel in ('source/paper_metadata.json', 'task/task.json'):
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text('{}\n')
for rel in ('task/adapter.py', 'evaluation/evaluation.py'):
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text('x\n')
time.sleep(float(os.environ.get('FAKE_BUILD_SLEEP', '0')))
'''

# Runs the real Human Program validator over study.json, so the fixture's
# missing derivation is the actual failure the repair has to fix.
FAKE_VALIDATOR = r'''import json, os, pathlib, sys
sys.path.insert(0, os.environ['REPO_ROOT'])
from agent_pipeline.human_program import validate_human_program
if os.environ.get('FAKE_NEVER_VALID'):
    raise SystemExit('ValueError: never valid')
study = json.loads((pathlib.Path(sys.argv[1]) / 'paper' / 'study.json').read_text())
validate_human_program(study['program'])
print('Validated agent package: paper')
'''


def _job(tmp_path: Path, repair: str, **env_extra: str) -> tuple[Path, Path, dict]:
    job = tmp_path / "job"
    (job / "logs").mkdir(parents=True)
    (job / "prompt.md").write_text("Build the package")
    (job / "job.json").write_text(json.dumps(
        {"id": "job-1", "status": "running", "repaired": True, "error": "stale error from an earlier run"}))
    validator = tmp_path / "validator.py"
    validator.write_text(FAKE_VALIDATOR)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "claude").write_text(FAKE_CLAUDE)
    (bin_dir / "claude").chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_REPAIR": repair,
           "FAKE_FIXTURE": str(FIXTURE), "REPO_ROOT": str(ROOT), **env_extra}
    return job, validator, env


def _run(job: Path, validator: Path, env: dict, timeout_minutes: str = "0.5") -> subprocess.CompletedProcess:
    return subprocess.run([
        sys.executable, str(ROOT / "agent_pipeline/run_agent.py"),
        "--job", str(job), "--prompt", str(job / "prompt.md"), "--model", "test-model",
        "--validator", str(validator), "--timeout-minutes", timeout_minutes, "--check-interval", "0.05",
    ], capture_output=True, text=True, env=env, timeout=120)


def test_watchdog_keeps_accepted_repair_and_records_it(tmp_path: Path) -> None:
    job, validator, env = _job(tmp_path, "derive")
    result = _run(job, validator, env)
    assert result.returncode == 0, result.stdout + result.stderr
    state = json.loads((job / "job.json").read_text())
    assert state["repaired"] is True and "error" not in state
    assert state["repair"]["accepted"] is True and state["repair"]["guard"]["passed"] is True
    assert state["repair"]["snapshot"] == "pre_repair" and state["repair"]["rounds"] == 1
    assert state["repair"]["guard"]["changedFiles"] == ["study.json"]
    post = json.loads((job / "package/paper/study.json").read_text())
    pre = json.loads((job / "pre_repair/paper/study.json").read_text())
    assert _field(post, "dv", "scale_mean")["derivation"] == "Mean of the items."
    assert "derivation" not in _field(pre, "dv", "scale_mean")
    assert "Mean of the items." in (job / "logs/repair.diff").read_text()
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
    assert any("moved out of missing/decision" in v for v in state["repair"]["guard"]["violations"])
    assert "fidelity guard" in state["error"]
    # The package under review and the committed snapshot are both the pre-repair
    # package, even though the repair turn tried to overwrite the snapshot.
    for path in (job / "package/paper/study.json", job / "pre_repair/paper/study.json"):
        assert path.read_text() == (FIXTURE / "study.json").read_text()


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
    assert "researcherMessage" not in state  # the fixture writes no researcher reply


def test_stopped_build_reports_the_agents_reply(tmp_path: Path) -> None:
    from agent_pipeline.run_agent import stopped_reply_message
    package = tmp_path / "package"
    assert stopped_reply_message(package) is None
    root = package / "paper"
    root.mkdir(parents=True)
    assert stopped_reply_message(package) is None
    (root / "studio-reply.md").write_text(
        "**Ready to run:** Nothing yet.\nneeds you: Please upload the supplement.\n"
        "Licence: CC BY 3.0 (printed on page 1)\n\n**Questions**\n")
    assert stopped_reply_message(package) == (
        "The study build ended without a finished package. "
        "Ready to run: Nothing yet. needs you: Please upload the supplement.")
    (root / "studio-reply.md").write_text("# Reply\nI could not finish.\n")
    assert stopped_reply_message(package) is None
    (root / "studio-reply.md").write_text("Needs you: " + "x" * 5000)
    assert len(stopped_reply_message(package)) == 1000


def test_watchdog_records_reply_of_a_stopped_build_for_this_run(tmp_path: Path) -> None:
    job, validator, env = _job(tmp_path, "derive", FAKE_REPLY="Ready to run: Nothing yet.\nNeeds you: Upload S1.\n",
                               GITHUB_RUN_ID="777")
    state = json.loads((job / "job.json").read_text())
    (job / "job.json").write_text(json.dumps({**state, "researcherMessage": {"text": "old", "runId": "1"}}))
    (job / "studio_request.json").write_text(json.dumps({"version": 1, "requestId": "req-1"}))
    result = _run(job, validator, env)
    assert "repair not attempted" in result.stderr
    note = json.loads((job / "job.json").read_text())["researcherMessage"]
    assert note == {"text": "The study build ended without a finished package. "
                            "Ready to run: Nothing yet. Needs you: Upload S1.", "runId": "777"}


def _workflow_step(name: str) -> str:
    import yaml
    workflow = yaml.safe_load((ROOT / ".github/workflows/run-humanstudy-pipeline.yml").read_text())
    steps = workflow["jobs"]["build-study"]["steps"]
    script = next(step["run"] for step in steps if step.get("name") == name)
    return script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


@pytest.mark.parametrize("note, run_id, expected", [
    ({"text": "Needs you: Upload S1.", "runId": "9"}, "9", "Needs you: Upload S1."),
    ({"text": "Needs you: Upload S1.", "runId": "8"}, "9", "The study build failed or was cancelled"),
    ("Needs you: Upload S1.", "9", "The study build failed or was cancelled"),
    (None, "9", "The study build failed or was cancelled"),
])
def test_failure_step_shows_only_this_runs_reply(tmp_path: Path, note, run_id: str, expected: str) -> None:
    job = tmp_path / "job-storage/jobs/job-1"
    job.mkdir(parents=True)
    (job / "job.json").write_text(json.dumps({"id": "job-1", **({"researcherMessage": note} if note else {})}))
    subprocess.run([sys.executable, "-c", _workflow_step("Mark failed or cancelled")], cwd=tmp_path, check=True,
                   env={**os.environ, "JOB_ID": "job-1", "GITHUB_RUN_ID": run_id})
    assert json.loads((job / "job.json").read_text())["message"] == expected


def test_watchdog_clears_stale_state_when_first_pass_is_valid(tmp_path: Path) -> None:
    job, validator, env = _job(tmp_path, "derive")
    (job / "pre_repair").mkdir()
    shutil.copytree(FIXTURE, job / "first")
    _edit(job / "first/study.json", _derive)
    env["FAKE_FIXTURE"] = str(job / "first")
    result = _run(job, validator, env)
    assert result.returncode == 0, result.stdout + result.stderr
    state = json.loads((job / "job.json").read_text())
    assert "repaired" not in state and "repair" not in state and "error" not in state
    assert not (job / "pre_repair").exists()


def test_build_pass_is_capped_to_leave_repair_budget(tmp_path: Path) -> None:
    # 0.6 min total with 0.3 min reserved: the build agent would sleep for 120 s,
    # so without the cap it would use the whole budget and never be repaired.
    job, validator, env = _job(tmp_path, "derive", FAKE_BUILD_SLEEP="120", AGENT_REPAIR_RESERVE_MINUTES="0.3")
    result = _run(job, validator, env, timeout_minutes="0.6")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "build pass reached its cap" in result.stdout
    assert json.loads((job / "logs/watchdog.json").read_text())["reason"] == "repaired"
    assert json.loads((job / "job.json").read_text())["repaired"] is True
