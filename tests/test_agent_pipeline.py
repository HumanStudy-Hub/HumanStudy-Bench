import json
import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from agent_pipeline.run_agent import ProgressPublisher, ensure_readme, package_progress, studio_complete
from agent_pipeline.build_prompt import conversation_context, fetch_materials, referenced_messages


ROOT = Path(__file__).resolve().parents[1]
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


def make_zip(entries: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return stream.getvalue()


def test_studio_materials_expand_one_nested_zip(tmp_path: Path) -> None:
    nested = make_zip({"instrument.csv": b"question,answer\nA,B\n"})
    source = tmp_path / "uploaded.zip"
    source.write_bytes(make_zip({"resources/source-1/questionnaire.zip": nested,
                                 "resources/source-2/notes.txt": b"Notes"}))
    job = tmp_path / "job"
    fetch_materials(job, {"openMaterialsUrl": source.as_uri(),
                          "openMaterialsSourceIds": ["source-1", "source-2"]})

    root = job / "input/open_materials/resources"
    assert (root / "source-1/questionnaire.contents/instrument.csv").read_text() == "question,answer\nA,B\n"
    assert (root / "source-2/notes.txt").read_text() == "Notes"


@pytest.mark.parametrize("entry", ["../escape.txt", "/absolute.txt", "C:/drive.txt", "a\\escape.txt"])
def test_studio_materials_reject_unsafe_paths(tmp_path: Path, entry: str) -> None:
    source = tmp_path / "uploaded.zip"
    source.write_bytes(make_zip({"resources/source-1/valid.txt": b"OK", entry: b"unsafe"}))
    job = tmp_path / "job"
    with pytest.raises(RuntimeError, match="Required Studio open materials"):
        fetch_materials(job, {"openMaterialsUrl": source.as_uri(), "openMaterialsSourceIds": ["source-1"]})
    assert not (job / "input/open_materials").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_studio_materials_reject_symlink_and_collision(tmp_path: Path) -> None:
    source = tmp_path / "uploaded.zip"
    with zipfile.ZipFile(source, "w") as archive:
        link = zipfile.ZipInfo("resources/source-1/link")
        link.create_system = 3
        link.external_attr = (0o120777 << 16)
        archive.writestr(link, "target")
    with pytest.raises(RuntimeError, match="link or special file"):
        fetch_materials(tmp_path / "job", {"openMaterialsUrl": source.as_uri(),
                                           "openMaterialsSourceIds": ["source-1"]})

    source.write_bytes(make_zip({"resources/source-1/A.txt": b"A",
                                 "resources/source-1/a.txt": b"a"}))
    with pytest.raises(RuntimeError, match="colliding paths"):
        fetch_materials(tmp_path / "job", {"openMaterialsUrl": source.as_uri(),
                                           "openMaterialsSourceIds": ["source-1"]})


def test_studio_materials_enforce_expanded_budget_and_depth(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "uploaded.zip"
    source.write_bytes(make_zip({"resources/source-1/big.txt": b"0123456789"}))
    monkeypatch.setattr("agent_pipeline.build_prompt.MAX_MATERIAL_EXPANDED", 8)
    with pytest.raises(RuntimeError, match="expanded limit"):
        fetch_materials(tmp_path / "job", {"openMaterialsUrl": source.as_uri(),
                                           "openMaterialsSourceIds": ["source-1"]})
    monkeypatch.setattr("agent_pipeline.build_prompt.MAX_MATERIAL_EXPANDED", 100 * 1024 * 1024)
    deepest = make_zip({"data.txt": b"data"})
    middle = make_zip({"second.zip": deepest})
    source.write_bytes(make_zip({"resources/source-1/first.zip": middle}))
    with pytest.raises(RuntimeError, match="nesting exceeds"):
        fetch_materials(tmp_path / "job", {"openMaterialsUrl": source.as_uri(),
                                           "openMaterialsSourceIds": ["source-1"]})


def test_studio_materials_bound_total_entries_and_download(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "uploaded.zip"
    with zipfile.ZipFile(source, "w") as archive:
        for index in range(3):
            archive.writestr(f"resources/source-1/folder-{index}/", b"")
    monkeypatch.setattr("agent_pipeline.build_prompt.MAX_MATERIAL_ENTRIES", 2)
    with pytest.raises(RuntimeError, match="entry limit"):
        fetch_materials(tmp_path / "job", {"openMaterialsUrl": source.as_uri(),
                                           "openMaterialsSourceIds": ["source-1"]})
    monkeypatch.setattr("agent_pipeline.build_prompt.MAX_MATERIAL_ENTRIES", 4000)
    monkeypatch.setattr("agent_pipeline.build_prompt.MAX_MATERIAL_DOWNLOAD", 10)
    with pytest.raises(RuntimeError, match="download exceeds"):
        fetch_materials(tmp_path / "job", {"openMaterialsUrl": source.as_uri(),
                                           "openMaterialsSourceIds": ["source-1"]})


def test_studio_materials_bound_path_components(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "uploaded.zip"
    source.write_bytes(make_zip({"resources/source-1/a/b/c/file.txt": b"data"}))
    monkeypatch.setattr("agent_pipeline.build_prompt.MAX_MATERIAL_PATH_PARTS", 5)
    with pytest.raises(RuntimeError, match="component limit"):
        fetch_materials(tmp_path / "job", {"openMaterialsUrl": source.as_uri(),
                                           "openMaterialsSourceIds": ["source-1"]})


def test_legacy_materials_failure_remains_optional(tmp_path: Path, capsys) -> None:
    source = tmp_path / "invalid.zip"
    source.write_bytes(b"not a ZIP")
    fetch_materials(tmp_path / "job", {"openMaterialsUrl": source.as_uri()})
    assert "continuing without them" in capsys.readouterr().out
    assert not (tmp_path / "job/input/open_materials").exists()


def test_studio_materials_fail_when_required_source_is_omitted(tmp_path: Path) -> None:
    source = tmp_path / "uploaded.zip"
    source.write_bytes(make_zip({"resources/source-1/notes.txt": b"Notes"}))
    with pytest.raises(RuntimeError, match="missing from the archive"):
        fetch_materials(tmp_path / "job", {"openMaterialsUrl": source.as_uri(),
                                           "openMaterialsSourceIds": ["source-1", "source-2"]})


def test_build_prompt_includes_job_inputs(tmp_path: Path) -> None:
    job = tmp_path / "jobs" / "example"
    (job / "input").mkdir(parents=True)
    (job / "input" / "paper.pdf").write_bytes(b"%PDF-test")
    (job / "job.json").write_text(json.dumps({
        "paperName": "paper.pdf",
        "contributorName": "Researcher",
        "osfUrl": "https://osf.io/example/",
    }))
    output = job / "agent-prompt.md"

    subprocess.run([
        sys.executable,
        str(ROOT / "agent_pipeline/build_prompt.py"),
        "--contract",
        str(ROOT / "agent_pipeline/CLAUDE.md"),
        "--job",
        str(job),
        "--output",
        str(output),
    ], check=True)

    prompt = output.read_text()
    assert "https://osf.io/example/" in prompt
    assert str((job / "input/paper.pdf").resolve()) in prompt
    assert "Do not invent study facts" in prompt


def test_build_prompt_disables_search_without_user_url(tmp_path: Path) -> None:
    job = tmp_path / "jobs" / "paper-only"
    (job / "input").mkdir(parents=True)
    (job / "input" / "paper.pdf").write_bytes(b"%PDF-test")
    (job / "job.json").write_text(json.dumps({
        "paperName": "paper.pdf",
        "contributorName": "Researcher",
    }))
    output = job / "agent-prompt.md"

    subprocess.run([
        sys.executable,
        str(ROOT / "agent_pipeline/build_prompt.py"),
        "--contract",
        str(ROOT / "agent_pipeline/CLAUDE.md"),
        "--job",
        str(job),
        "--output",
        str(output),
    ], check=True)

    prompt = output.read_text()
    assert "Network research is not authorized" in prompt
    assert "Use only the uploaded PDF" in prompt


def test_build_prompt_includes_bounded_studio_feedback(tmp_path: Path) -> None:
    job = tmp_path / "job"
    (job / "input").mkdir(parents=True)
    (job / "job.json").write_text(json.dumps({"paperName": "paper.pdf"}))
    (job / "studio_request.json").write_text(json.dumps({
        "version": 1, "workspaceId": "workspace-1", "conversationId": "conversation-1",
        "requestId": "request-1", "message": "Clarify the analysis unit for this outcome.",
        "modelAnchor": {"kind": "objects", "entityIds": ["analysis"]},
        "document": {"model": {"id": "study-1"}, "sources": [{"id": "source-1", "name": "paper.pdf",
                     "pages": [{"page": 3, "text": "Source passage."}]}],
                     "annotations": [], "reviewResponses": {}, "conversations": [], "artifacts": []},
    }))
    output = job / "agent-prompt.md"

    subprocess.run([sys.executable, str(ROOT / "agent_pipeline/build_prompt.py"),
                    "--contract", str(ROOT / "agent_pipeline/CLAUDE.md"),
                    "--job", str(job), "--output", str(output)], check=True)

    prompt = output.read_text()
    assert "Clarify the analysis unit" in prompt
    assert '"entityIds": ["analysis"]' in prompt
    assert "Source passage." in prompt
    assert "If `package/` has no paper folder, build a new complete package" in prompt
    assert "alongside `study.json` (never at the top level of `package/`)" in prompt
    assert "kind (`participants`, `material`, `procedure`, `record`," in prompt
    assert "Procedure step: id, name, input, actor, output, evidence" in prompt
    assert "Variable: id, name, role, type, unit, producedBy, usedBy, definition" in prompt
    assert "the physical PDF page number, starting at 1" in prompt
    assert "Use `rects: []` for new citations" in prompt
    assert "sourceId when an attached source ID is present" in prompt
    assert "studio-model.json" in prompt
    assert "Optional root `reviewIssues`" in prompt
    assert "genuinely missing rule that prevents execution" in prompt
    assert "future run-time observation or derived statistic is not missing input" in prompt
    assert "studio-complete.json" in prompt
    assert "untrusted research context" in prompt


def test_studio_branch_context_stops_at_fork_and_omits_siblings() -> None:
    def message(id: str) -> dict:
        return {"id": id, "role": "user", "text": id}

    conversations = [
        {"id": "main", "messages": [message("m1"), message("m2"), message("later")]},
        {"id": "side", "parent": {"conversationId": "main", "messageId": "m2"}, "messages": [message("s1"), message("current")]},
        {"id": "sibling", "parent": {"conversationId": "main", "messageId": "m2"}, "messages": [message("other")]},
    ]
    context = conversation_context(conversations, "side", "current")
    assert [item["id"] for item in context] == ["m1", "m2", "s1"]
    assert conversation_context(conversations, "missing", "current") == []


def test_studio_branch_context_rejects_broken_links_and_cycles() -> None:
    conversations = [
        {"id": "one", "parent": {"conversationId": "two", "messageId": "b"}, "messages": [{"id": "a"}]},
        {"id": "two", "parent": {"conversationId": "one", "messageId": "a"}, "messages": [{"id": "b"}]},
    ]
    try:
        conversation_context(conversations, "one", "new")
        assert False, "cycle should fail"
    except ValueError as error:
        assert "cycle" in str(error)
    conversations[1].pop("parent")
    conversations[0]["parent"]["messageId"] = "missing"
    try:
        conversation_context(conversations, "one", "new")
        assert False, "missing parent message should fail"
    except ValueError as error:
        assert "message is missing" in str(error)


def test_explicit_refs_include_old_turn_and_one_sibling_message_only() -> None:
    old = {"conversationId": "main", "messageId": "m0"}
    selected = {"conversationId": "side", "messageId": "s1"}
    conversations = [
        {"id": "main", "messages": [{"id": f"m{i}", "role": "user", "text": f"turn-{i}"} for i in range(30)]},
        {"id": "side", "parent": old, "messages": [{"id": "s1", "role": "agent", "text": "Selected insight"}, {"id": "s2", "text": "private later turn"}]},
    ]
    recent = conversation_context(conversations, "main", "new")[-24:]
    assert "m0" not in [item["id"] for item in recent]
    assert "s1" not in [item["id"] for item in recent]
    request = {"replyTo": old, "mergedFrom": selected, "referencedMessages": [
        {"relation": "replyTo", "ref": old, "message": {"id": "m0", "role": "user", "text": "turn-0", "modelAnchor": {"kind": "objects", "entityIds": ["analysis"]}}},
        {"relation": "mergedFrom", "ref": selected, "message": {"id": "s1", "role": "agent", "text": "Selected insight", "sourceSelection": {"page": 3, "text": "Source quote"}}},
    ]}
    refs = referenced_messages(request)
    assert [item["message"]["text"] for item in refs] == ["turn-0", "Selected insight"]
    assert refs[0]["message"]["modelAnchor"]["entityIds"] == ["analysis"]
    assert refs[1]["message"]["sourceSelection"]["text"] == "Source quote"
    assert "private later turn" not in json.dumps(refs)


def test_validate_complete_agent_package(tmp_path: Path) -> None:
    package = tmp_path / "package"
    study = package / "paper-name"
    for relative in REQUIRED:
        path = study / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text("{}\n")
        elif relative == "task/adapter.py":
            path.write_text("import sys\ndef run_sessions(llm, seed, n, arms=None):\n    return []\nif __name__ == '__main__':\n    raise SystemExit(0 if '--smoke-test' in sys.argv else 1)\n")
        elif relative == "evaluation/evaluation.py":
            path.write_text("def evaluate(sessions):\n    return {}\n")
        else:
            path.write_text("Generated test file\n")

    result = subprocess.run([
        sys.executable,
        str(ROOT / "agent_pipeline/validate_package.py"),
        str(package),
    ], capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Validated agent package" in result.stdout


def test_validate_rejects_missing_file(tmp_path: Path) -> None:
    study = tmp_path / "package" / "paper-name"
    study.mkdir(parents=True)

    result = subprocess.run([
        sys.executable,
        str(ROOT / "agent_pipeline/validate_package.py"),
        str(tmp_path / "package"),
    ], capture_output=True, text=True)

    assert result.returncode != 0
    assert "missing required package files" in result.stderr


def test_watchdog_progress_counts_required_files(tmp_path: Path) -> None:
    package = tmp_path / "package"
    study = package / "paper-name"
    (study / "source").mkdir(parents=True)
    (study / "study.json").write_text("{}\n")
    (study / "source/paper_metadata.json").write_text("{}\n")
    (study / "extra.txt").write_text("Additional material\n")

    completed, total, missing = package_progress(package)

    assert completed == 2
    assert total == 3
    assert "task/adapter.py" in missing


def test_studio_completion_requires_matching_request(tmp_path: Path) -> None:
    assert studio_complete(tmp_path, None)
    assert not studio_complete(tmp_path, "new-request")
    (tmp_path / "studio-complete.json").write_text(json.dumps({"requestId": "old-request", "status": "complete"}))
    assert not studio_complete(tmp_path, "new-request")
    (tmp_path / "studio-complete.json").write_text(json.dumps({"requestId": "new-request", "status": "complete"}))
    assert studio_complete(tmp_path, "new-request")


def test_readme_is_generated_from_study_title(tmp_path: Path) -> None:
    root = tmp_path / "package" / "paper-name"
    root.mkdir(parents=True)
    (root / "study.json").write_text(json.dumps({"paper": {"title": "Paper title"}}))

    ensure_readme(tmp_path / "package")

    readme = (root / "README.md").read_text()
    assert readme.startswith("# Paper title")
    assert "missing_information.json" in readme


def test_progress_publisher_updates_existing_progress(monkeypatch) -> None:
    calls = []

    def fake_request(self, method, body=None):
        calls.append((method, body))
        return {"sha": "old-sha"} if method == "GET" else {"content": {"sha": "new-sha"}}

    monkeypatch.setattr(ProgressPublisher, "_request", fake_request)
    publisher = ProgressPublisher("secret", "owner/jobs", "jobs/example", "jobs/example/progress.json")
    publisher.publish({"completedRequired": 4, "totalRequired": 8})

    assert publisher.sha == "new-sha"
    assert calls[1][1]["sha"] == "old-sha"
    assert calls[1][1]["branch"] == "jobs/example"


def test_watchdog_stops_agent_after_validation(tmp_path: Path) -> None:
    job = tmp_path / "job"
    (job / "logs").mkdir(parents=True)
    prompt = job / "prompt.md"
    prompt.write_text("Build the package")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_claude = bin_dir / "claude"
    fake_claude.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys, time\n"
        "job = pathlib.Path(sys.argv[sys.argv.index('--add-dir') + 1])\n"
        "(job / 'claude-env.json').write_text(json.dumps({'pipeline_token_visible': 'PIPELINE_PROGRESS_TOKEN' in os.environ}))\n"
        "root = job / 'package' / 'paper'\n"
        f"required = {REQUIRED!r}\n"
        "for relative in required:\n"
        "    path = root / relative\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    if relative == 'task/adapter.py':\n"
        "        path.write_text(\"import sys\\ndef run_sessions(llm, seed, n, arms=None):\\n    return []\\nif __name__ == '__main__':\\n    raise SystemExit(0 if '--smoke-test' in sys.argv else 1)\\n\")\n"
        "    elif relative == 'evaluation/evaluation.py':\n"
        "        path.write_text(\"def evaluate(sessions):\\n    return {}\\n\")\n"
        "    elif path.suffix == '.json':\n"
        "        path.write_text(json.dumps({}) + '\\n')\n"
        "    else:\n"
        "        path.write_text('Generated\\n')\n"
        "print('package written; waiting forever', flush=True)\n"
        "time.sleep(60)\n"
    )
    fake_claude.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "PIPELINE_PROGRESS_TOKEN": "private-test-token"}

    result = subprocess.run([
        sys.executable,
        str(ROOT / "agent_pipeline/run_agent.py"),
        "--job",
        str(job),
        "--prompt",
        str(prompt),
        "--model",
        "test-model",
        "--validator",
        str(ROOT / "agent_pipeline/validate_package.py"),
        "--timeout-minutes",
        "0.1",
        "--check-interval",
        "0.05",
    ], capture_output=True, text=True, env=env, timeout=10)

    assert result.returncode == 0, result.stdout + result.stderr
    watchdog = json.loads((job / "logs/watchdog.json").read_text())
    assert watchdog["reason"] == "validator_passed"
    assert watchdog["package_valid"] is True
    assert json.loads((job / "claude-env.json").read_text())["pipeline_token_visible"] is False


def test_watchdog_waits_for_studio_refinement_of_copied_package(tmp_path: Path) -> None:
    job = tmp_path / "job"
    root = job / "package" / "paper"
    (job / "logs").mkdir(parents=True)
    for relative in REQUIRED:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "existing\n")
    (job / "studio_request.json").write_text(json.dumps({"version": 1, "requestId": "current-request"}))
    (job / "studio-complete.json").write_text(json.dumps({"requestId": "current-request", "status": "complete"}))
    prompt = job / "prompt.md"
    prompt.write_text("Refine the package")
    validator = tmp_path / "validator.py"
    validator.write_text("import sys\nraise SystemExit(0)\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_claude = bin_dir / "claude"
    fake_claude.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys, time\n"
        "job = pathlib.Path(sys.argv[sys.argv.index('--add-dir') + 1])\n"
        "time.sleep(0.3)\n"
        "(job / 'package/paper/studio-reply.md').write_text('Updated analysis unit.\\n')\n"
        "(job / 'studio-complete.json').write_text(json.dumps({'requestId': 'current-request', 'status': 'complete'}))\n"
        "time.sleep(60)\n"
    )
    fake_claude.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}

    result = subprocess.run([
        sys.executable, str(ROOT / "agent_pipeline/run_agent.py"),
        "--job", str(job), "--prompt", str(prompt), "--model", "test-model",
        "--validator", str(validator), "--timeout-minutes", "0.1", "--check-interval", "0.05",
    ], capture_output=True, text=True, env=env, timeout=10)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "waiting for Studio refinement completion" in result.stdout
    assert (root / "studio-reply.md").read_text() == "Updated analysis unit.\n"
    assert json.loads((job / "logs/watchdog.json").read_text())["reason"] == "validator_passed"
