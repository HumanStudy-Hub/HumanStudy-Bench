"""Invented CC0 Desktop export: structural intake never grants publication or execution."""
import hashlib
import json
from pathlib import Path

import pytest
from scripts.research.revisions import integrate, validate_assets, revision_id

ROOT = Path(__file__).resolve().parent / "fixtures/paired-draft"


def files():
    return {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in ROOT.rglob("*") if p.is_file()}


def put(f, name, value):
    f[name] = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def edit(f, name, change):
    value = json.loads(f[name]); change(value); put(f, name, value)


def meta():
    return {"kind": "protocol", "paperKey": "title:lamp-study", "paper": {"title": "Lamp study", "year": 2026}}


def test_actual_desktop_draft_waits_for_review_even_with_fake_acceptance(tmp_path):
    f = files()
    result = integrate(tmp_path, meta(), f, review={"decision": "accept", "reviewer": "uploader",
                                                  "candidateRevision": revision_id(f)})
    assert result["status"] == "needs_review"
    assert result["requiresMaintainer"] is True
    assert result["validation"]["checks"]["codeExecuted"] is False
    assert result["validation"]["checks"]["scientificReview"] is False
    assert not result["validation"]["capabilities"]["replayable"]
    assert not (tmp_path / "current.json").exists()
    assert not (tmp_path / "revisions").exists()
    assert validate_assets(f)["status"] == "failed"  # still not a formal release


@pytest.mark.parametrize("mutation", [
    lambda f: edit(f, "program.json", lambda p: p.pop("steps")),
    lambda f: edit(f, "program.json", lambda p: p["extensions"]["scholarly"]["protocol"].update(sha256="0"*64)),
    lambda f: edit(f, "study.json", lambda s: s["protocol"].update(sha256="0"*64)),
    lambda f: edit(f, "study.json", lambda s: s.update(title="Different paper")),
    lambda f: edit(f, "study.json", lambda s: s.update(doi="10.1234/another")),
    lambda f: edit(f, "protocol.json", lambda p: p.update(arms="bad schema")),
    lambda f: edit(f, "protocol.json", lambda p: p["distribution"].update(held={"x": {}})),
    lambda f: edit(f, "sources/SOURCE.json", lambda s: s["files"][0].update(license="All rights reserved")),
    lambda f: f.update({"sources/private/secret.txt": b"private"}),
    lambda f: f.update({"artifacts/run/records.jsonl": b"raw"}),
    lambda f: f.update({"sources/open/../private/secret.txt": b"private"}),
    lambda f: f.update({"review/checks-latest.json": b'{"PASS":true,"protocol_sha256":"wrong"}'}),
])
def test_invalid_draft_remains_failed(tmp_path, mutation):
    f = files(); mutation(f)
    result = integrate(tmp_path, meta(), f)
    assert result["status"] == "checks_failed", result
    assert not (tmp_path / "current.json").exists()


def test_doi_mismatch_rejected_after_structural_checks(tmp_path):
    f = files()
    edit(f, "program.json", lambda p: p["source"].update(doi="10.1234/original"))
    edit(f, "study.json", lambda s: s.update(doi="10.1234/original"))
    m = meta(); m.update(paperKey="doi:10.1234/wrong", paper={"title":"Lamp study", "doi":"10.1234/wrong"})
    assert integrate(tmp_path, m, f)["status"] == "checks_failed"


def test_protocol_checks_never_execute_uploads(tmp_path):
    f = files()
    f["artifacts/probe.py"] = b"raise RuntimeError('must not execute')"
    assert integrate(tmp_path, meta(), f)["status"] == "needs_review"


def test_bad_program_checked_even_with_other_publication_errors():
    f = files()
    edit(f, "program.json", lambda p: p.pop("steps"))
    edit(f, "study.json", lambda s: s.update(release_hold="not cleared"))
    result = validate_assets(f, draft=True)
    assert result["status"] == "failed"
    assert any("program:" in e and "steps" in e for e in result["errors"])
    assert any("release_hold" in e for e in result["errors"])


def test_no_doi_draft_still_checks_submission_title(tmp_path):
    m = meta(); m["paper"]["title"] = "Another paper"
    assert integrate(tmp_path, m, files())["status"] == "checks_failed"


def test_program_only_proposal_cannot_remove_a_recommended_protocol(tmp_path):
    paired = files()
    current = revision_id(paired)
    folder = tmp_path / "revisions" / current
    manifest = []
    for name, raw in paired.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        manifest.append({"path": name, "sha256": hashlib.sha256(raw).hexdigest()})
    (folder / "manifest.json").write_text(json.dumps({"files": manifest}))
    pointer = tmp_path / "current.json"
    pointer.write_text(json.dumps({"revision": current}))
    proposal = {"program.json": paired["program.json"]}
    def program_only(p):
        p["extensions"]["scholarly"].pop("protocol")
        p["evidence"].append({"id": "withheld", "path": "handout.txt", "sourceSha256": "b" * 64,
                              "locator": {"page": 1}})
    edit(proposal, "program.json", program_only)
    standalone = {"paperKey": "title:lamp-study", "paper": {"title": "Lamp study", "year": 2026},
                  "baseRevision": current}
    result = integrate(tmp_path, standalone, proposal,
                       review={"decision": "accept", "reviewer": "curator", "candidateRevision": revision_id(proposal)})
    assert result["status"] == "needs_review"
    assert result["requiresMaintainer"] is True
    assert result["validation"]["status"] == "passed"
    assert not result["validation"]["capabilities"]["replayable"]
    assert json.loads(pointer.read_text())["revision"] == current
    assert not (tmp_path / "revisions" / revision_id(proposal)).exists()
