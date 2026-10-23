import json
import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

import scripts.maintain_community as worker
from scripts.research.revisions import revision_id

ROOT = Path(__file__).resolve().parents[1]


def test_worker_reads_untrusted_git_blobs_without_checkout_or_execution(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path).decode().strip()
    git("init", "-b", "main")
    git("config", "user.name", "Fixture author")
    git("config", "user.email", "fixture@example.invalid")
    (tmp_path / "README.md").write_text("Trusted baseline\n")
    git("add", "."); git("commit", "-m", "baseline")
    base = git("rev-parse", "HEAD")
    git("switch", "-c", "community/fixture")
    mailbox = "paper_" + hashlib.sha256(b"doi:10.1234/test").hexdigest()[:24]
    proposal = tmp_path / f"community/{mailbox}/proposals/abcdef01"
    proposal.mkdir(parents=True)
    program = (ROOT / "tests/fixtures/research-program.json").read_bytes()
    (proposal / "program.json").write_bytes(program)
    artifact = proposal / "artifacts/never-execute.py"
    artifact.parent.mkdir()
    artifact.write_text("raise RuntimeError('Untrusted code must never execute')\n")
    metadata = {"paperKey": "doi:10.1234/test", "paper": {"title": "Fixture", "doi": "10.1234/test", "authors": [], "year": 2026},
                "contributor": {"name": "Researcher", "github": "researcher"}, "intent": "reconstruct", "baseRevision": None}
    (proposal / "CONTRIBUTION.json").write_text(json.dumps(metadata))
    git("add", "."); git("commit", "-m", "proposal")
    head = git("rev-parse", "HEAD")
    git("switch", "main")
    assert not proposal.exists()
    (tmp_path / "catalog").mkdir()
    shutil.copy(ROOT / "catalog/taxonomy.json", tmp_path / "catalog/taxonomy.json")
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "build_index", lambda: [])
    monkeypatch.setattr(worker, "review_candidate", lambda files, rid, paper: {"candidateRevision": rid, "decision": "accept", "reviewer": "independent-test-reviewer"})
    report = worker.maintain(head, base, "researcher", agent_review=True)
    assert report["status"] == "accepted"
    assert git("rev-parse", "HEAD") == base
    catalog = json.loads((tmp_path / "co_website/data/research_index.json").read_text())
    assert len(catalog["papers"]) == 1
    assert catalog["papers"][0]["currentRevision"] == report["revision"]
    assert not catalog["papers"][0]["capabilities"]["replayable"]
    with pytest.raises(ValueError, match="authenticated GitHub"):
        worker.proposal_from_ref(head, base, "someone-else")

