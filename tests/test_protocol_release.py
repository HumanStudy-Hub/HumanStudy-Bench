"""Protocol releases in the community store (V080-PLATFORM §1.2, §2.1, B2).

The fixture (tests/fixtures/protocol_release/example2026lamps) is invented, CC0-1.0 material: no real
paper, protocol or participant. Its review/checks-latest.json is a real hsenv checks report on it.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("hsenv")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsenv.schema_check import canonical_sha256  # noqa: E402
from scripts.research import protocols  # noqa: E402
from scripts.research.revisions import revision_id, validate_assets  # noqa: E402

FIXTURE = ROOT / "tests/fixtures/protocol_release/example2026lamps"


def fixture_files() -> dict:
    return {p.relative_to(FIXTURE).as_posix(): p.read_bytes() for p in sorted(FIXTURE.rglob("*")) if p.is_file()}


def put(files, name, data):
    files[name] = (json.dumps(data, indent=1) + "\n").encode()


def edit(files, name, change):
    data = json.loads(files[name])
    change(data)
    put(files, name, data)
    return data


def rebind(files):
    """Keep study.json and the checks report bound to an edited protocol (as hsenv study sync would)."""
    sha = canonical_sha256(json.loads(files["protocol.json"]))
    edit(files, "study.json", lambda s: s["protocol"].update(sha256=sha))
    edit(files, "review/checks-latest.json", lambda r: r.update(protocol_sha256=sha))


def store(root: Path, files: dict, paper="paper_" + "0" * 24, name=None) -> Path:
    rid = revision_id(files)
    folder = root / "community" / paper / "revisions" / (name or rid)
    for rel, raw in files.items():
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(raw)
    entries = [{"path": n, "sha256": hashlib.sha256(r).hexdigest(), "bytes": len(r)} for n, r in sorted(files.items())]
    (folder / "manifest.json").write_text(json.dumps({"schemaVersion": 1, "revision": rid, "files": entries}))
    return folder


def errors(files, **kw):
    return protocol_errors(protocols.validate_protocol_files(files, **kw))


def protocol_errors(result):
    return "\n".join(result["errors"])


def test_fixture_release_passes_with_engine_replay(tmp_path):
    result = protocols.validate_revision_dir(store(tmp_path, fixture_files()), sessions=50)
    assert result["status"] == "passed", result["errors"]
    assert result["replay"]["sessions"] == 50
    assert result["summary"]["arms"] == [{"id": "main", "runnable": True, "nPaper": 40}]


def test_directory_name_must_be_the_manifest_digest(tmp_path):
    folder = store(tmp_path, fixture_files(), name="f" * 64)
    assert "digest" in protocol_errors(protocols.validate_revision_dir(folder))


@pytest.mark.parametrize("where", ["arm", "study"])
def test_release_hold_is_refused(where):
    files = fixture_files()
    if where == "arm":
        edit(files, "protocol.json", lambda p: p["arms"][0].update(release_hold="licence not stated"))
        rebind(files)
    else:
        edit(files, "study.json", lambda s: s.update(release_hold="licence not stated"))
    assert "release_hold" in errors(files)


@pytest.mark.parametrize("name", ["sources/private/notes.txt", "sources/researcher/a.pdf",
                                  "sources/quarantine/x.txt", "runs/2026-11-01/records.jsonl",
                                  "artifacts/runs/1/records.jsonl", "paper/paper.pdf", "bundle.json",
                                  "sourcetext/closedsource.txt", "review/notes.md"])
def test_non_public_paths_are_refused(name):
    files = fixture_files()
    files[name] = b"x"
    assert name in errors(files)


def test_open_source_text_and_assets_are_public():
    files = fixture_files()
    files["sourcetext/survey#second.txt"] = b"x"
    files["variants/short.json"] = b"{}"
    assert errors(files) == ""


def test_protocol_hash_and_stale_report_fail():
    files = fixture_files()
    edit(files, "protocol.json", lambda p: p["meta"].update(citation=p["meta"]["citation"] + " Edited."))
    text = errors(files)
    assert "protocol.sha256 does not match" in text and "stale" in text


def test_failed_checks_report_fails():
    files = fixture_files()
    edit(files, "review/checks-latest.json", lambda r: r.update(PASS=False))
    assert "did not PASS" in errors(files)
    del files["review/checks-latest.json"]
    assert "checks-latest.json: missing" in errors(files)


@pytest.mark.parametrize("change", [lambda s: s.update(visibility="private"),
                                    lambda s: s["fidelity"].update(signed_by=""),
                                    lambda s: s.update(quality="checked"),
                                    lambda s: s["fidelity"].update(review="review/fidelity-review-9.md")])
def test_release_needs_public_visibility_and_signed_fidelity(change):
    files = fixture_files()
    edit(files, "study.json", change)
    assert errors(files)


def test_arms_in_study_json_follow_the_protocol():
    files = fixture_files()
    edit(files, "study.json", lambda s: s.update(arms=[{"id": "main", "runnable": False}]))
    assert "arms disagree" in errors(files)


def test_schema_errors_and_legacy_files_fail():
    files = fixture_files()
    edit(files, "protocol.json", lambda p: p.pop("schema_version"))
    rebind(files)
    assert "schema_version is required" in errors(files)
    files = fixture_files()
    edit(files, "study.json", lambda s: s.update(colour="blue"))
    assert "study.json" in errors(files)


@pytest.mark.parametrize("src,ok", [("https://example.org/a.png", True), ("lamp.png", False),
                                    ("asset:0123456789abcdef", False), ("OPEN", True)])
def test_image_sources(src, ok):
    files = fixture_files()
    files["sources/open/lamp.png"] = b"\x89PNG invented"
    if src == "OPEN":
        src = "asset:" + hashlib.sha256(files["sources/open/lamp.png"]).hexdigest()[:16]
    edit(files, "protocol.json", lambda p: p["stages"]["q"]["content"].append(
        {"id": "img", "value": {"src": src, "alt": "a lamp", "cite": {"label": "verbatim", "source": "survey", "page": 1}}}))
    rebind(files)
    assert ("image source" not in errors(files)) is ok


def test_engine_range_is_checked(monkeypatch):
    files = fixture_files()
    monkeypatch.setattr(protocols, "engine_version", lambda: "0.9.1")
    assert "needs hsenv" in errors(files)
    monkeypatch.setattr(protocols, "engine_version", lambda: "0.8.3")
    assert errors(files) == ""


def test_validate_assets_takes_the_protocol_branch():
    result = validate_assets(fixture_files())
    assert result["status"] == "passed", result["errors"]
    assert result["checks"]["codeExecuted"] is False
    assert result["capabilities"]["replayable"] and result["capabilities"]["humanBenchmark"]
    assert result["projection"]["protocol"]["slug"] == "example2026lamps"


def test_validate_assets_without_protocol_still_needs_program():
    files = fixture_files()
    del files["protocol.json"]
    assert validate_assets(files)["status"] == "failed"


def test_allowlist_is_the_engine_list_when_hsenv_ships_one():
    release = pytest.importorskip("hsenv.release")
    assert tuple(release.PUBLIC_PATHS) == tuple(protocols.PUBLIC_PATHS)


def test_fixture_is_invented_cc0():
    source = json.loads((FIXTURE / "sources/SOURCE.json").read_text())
    assert all(s["licence"] == "CC0-1.0" for s in source.values())
    assert "not a real" in (FIXTURE / "sources/open/survey.txt").read_text()


# ---- B3: research_index protocol block (V080 §3.1) -------------------------------------------------------

def catalog_root(tmp_path):
    (tmp_path / "catalog").mkdir()
    (tmp_path / "catalog/taxonomy.json").write_text(json.dumps({"version": "test", "concepts": [], "assignments": []}))
    return tmp_path


PROPOSAL_META = {"paperKey": "title:example2026lamps", "intent": "reconstruct", "baseRevision": None,
                 "paper": {"title": "An invented lamp-rating study (public test fixture)",
                           "authors": ["A. Example", "B. Sample"], "year": 2026},
                 "contributor": {"name": "Fixture Author", "github": "fixture-author"}}


@pytest.mark.parametrize("reviewer", ["independent-agent:some/model", "test-curator", "fixture-maintainer"])
def test_proposed_protocol_release_always_needs_a_maintainer(tmp_path, reviewer):
    """On the community proposal path nothing accepts a protocol release: not an agent's "accept", not any
    review dict, and not the release's own signed fidelity block and PASS checks report."""
    from scripts.research.revisions import integrate
    root = catalog_root(tmp_path)
    files = fixture_files()
    study = json.loads(files["study.json"])
    assert study["fidelity"]["signed_by"] and json.loads(files["review/checks-latest.json"])["PASS"] is True
    folder = root / "community/paper_test"
    review = {"candidateRevision": revision_id(files), "reviewer": reviewer, "decision": "accept",
              "verifiedCapabilities": {"traceable": True}}
    for kwargs in ({}, {"review": review}):
        result = integrate(folder, dict(PROPOSAL_META), files, **kwargs)
        assert result["status"] == "needs_review", result
        assert result["requiresMaintainer"] is True
        assert result["candidateRevision"] == revision_id(files)
    assert not (folder / "current.json").exists()
    assert not (folder / "revisions").exists()


def test_maintain_skips_agent_review_for_protocol_releases(tmp_path, monkeypatch):
    import scripts.maintain_community as mc
    files = fixture_files()
    folder = catalog_root(tmp_path) / "community/paper_test"
    monkeypatch.setattr(mc, "ROOT", tmp_path)
    monkeypatch.setattr(mc, "proposal_from_ref", lambda *a: (folder, {**PROPOSAL_META, "proposalId": "p1"}, files))
    monkeypatch.setattr(mc, "build_index", lambda: [])
    monkeypatch.setattr(mc, "write_catalog", lambda *a: None)
    called = []
    monkeypatch.setattr(mc, "review_candidate", lambda *a: called.append(a) or {"decision": "accept"})
    report = mc.maintain("head", "base", "fixture-author", agent_review=True)
    assert report["status"] == "needs_review" and report["requiresMaintainer"] is True
    assert not called
    assert not (folder / "current.json").exists()


def test_released_protocol_gets_a_protocol_block(tmp_path):
    """A maintainer release PR commits the revision directly; the catalog then indexes it."""
    from scripts.research.catalog import build_catalog
    root = catalog_root(tmp_path)
    files = fixture_files()
    folder = store(root, files, paper="paper_test")
    (folder.parents[1] / "current.json").write_text(json.dumps({
        "schemaVersion": 1, "revision": folder.name, "paper": {"id": "paper_test", "title": "t", "aliases": []},
        "capabilities": {"discoverable": True}}))
    paper = build_catalog(root, [])["papers"][0]
    block = paper["protocol"]
    assert block == {"slug": "example2026lamps", "schemaVersion": "1", "hsenv": ">=0.8,<0.9",
                     "fidelity": {"verdict": "HIGH FIDELITY", "signed_by": "fixture-maintainer"},
                     "arms": [{"id": "main", "runnable": True, "nPaper": 40}], "runExcluded": [],
                     "effects": 1, "revision": folder.name}
    assert paper["capabilities"]["replayable"] and paper["capabilities"]["humanBenchmark"]
    assert paper["coverage"] == "protocol-release"
    # counts and ids only: no on-screen text of the protocol reaches the index
    text = json.dumps(paper)
    assert "warm, soft light" not in text and "like this lamp" not in text


def test_hand_set_capabilities_are_recomputed_from_the_protocol(tmp_path):
    from scripts.research.catalog import build_catalog
    root = catalog_root(tmp_path)
    files = fixture_files()
    edit(files, "protocol.json", lambda p: p["arms"][0].update(runnable=False, why="fixture"))
    rebind(files)
    edit(files, "study.json", lambda s: s.update(arms=[{"id": "main", "runnable": False}]))
    folder = store(root, files, paper="paper_test")
    (folder.parents[1] / "current.json").write_text(json.dumps({
        "schemaVersion": 1, "revision": folder.name, "paper": {"id": "paper_test", "title": "t", "aliases": []},
        "capabilities": {"discoverable": True, "replayable": True, "humanBenchmark": False}}))
    paper = build_catalog(root, [])["papers"][0]
    assert paper["capabilities"]["replayable"] is False
    assert paper["protocol"]["arms"][0]["runnable"] is False


def test_legacy_indexes_are_byte_identical():
    """studies_index.json keeps its legacy shape; research_index.json changes only for protocol releases."""
    from scripts.build_studies_index import build_index
    from scripts.research.catalog import build_catalog
    studies = build_index()
    assert (json.dumps({"studies": studies}, ensure_ascii=False, indent=2)
            == (ROOT / "co_website/data/studies_index.json").read_text(encoding="utf-8"))
    value = build_catalog(ROOT, studies)
    assert (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            == (ROOT / "co_website/data/research_index.json").read_text(encoding="utf-8"))


SHARED = ROOT / "tests/fixtures/protocol_release/research-index-protocol.json"


def indexed_fixture_paper(root):
    from scripts.research.catalog import build_catalog
    files = fixture_files()
    folder = store(root, files, paper="paper_test")
    (folder.parents[1] / "current.json").write_text(json.dumps({
        "schemaVersion": 1, "revision": folder.name, "paper": {"id": "paper_test", "title": "t", "aliases": []},
        "capabilities": {"discoverable": True, "traceable": False, "reanalyzable": False,
                         "replayable": False, "humanBenchmark": False}}))
    return build_catalog(root, [])["papers"][0]


def test_shared_contract_fixture_matches_what_bench_writes(tmp_path):
    """The web repo tests its research_index reader against this same file (protocol schemaVersion "1")."""
    paper = indexed_fixture_paper(catalog_root(tmp_path))
    got = {"protocol": paper["protocol"], "capabilities": paper["capabilities"]}
    assert got["protocol"]["schemaVersion"] == "1"
    assert got == json.loads(SHARED.read_text())
