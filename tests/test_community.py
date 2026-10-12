"""Community layout: paper keys, contribution validation, PR rules and the index."""

import copy
import json
from pathlib import Path

import pytest

from scripts import build_studies_index as bsi
from scripts.check_community_pr import check_changes, parse_name_status
from scripts.community import (
    make_paper_key,
    normalise_title,
    paper_key_to_dir,
    parse_contribution_dir,
)
from scripts.validate_contribution import main as validate_main
from scripts.validate_contribution import validate_contribution

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = json.loads((ROOT / "tests/fixtures/human-program-v2.json").read_text())
KEY = "doi:10.1111/1468-0017.00215"
PAPER_DIR = "doi_10.1111_1468-0017.00215"
FOLDER = "knobe-2003--gh-ada"


def knobe_program() -> dict:
    return copy.deepcopy(next(c["program"] for c in FIXTURES["cases"] if c["id"] == "study_005"))


def contribution_meta(**overrides) -> dict:
    meta = {
        "schemaVersion": 1,
        "paperKey": KEY,
        "slug": "knobe-2003",
        "contributor": {"name": "Ada Lovelace", "github": "Ada"},
        "license": {"content": "CC-BY-4.0", "code": "MIT"},
        "includesPdf": False,
        "appVersion": "0.9.0",
        "createdAt": "2026-10-12T09:30:00Z",
        "checks": {"schema": True, "package": None, "simulation": None},
    }
    meta.update(overrides)
    return meta


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")


def make_package(folder: Path, program: dict) -> None:
    root = folder / "package" / "knobe-2003"
    write(root / "study.json", {"program": program})
    for name in ("source/paper_metadata.json", "source/evidence.json", "materials/materials.json"):
        write(root / name, {})
    write(root / "audit/missing_information.json", [])
    write(
        root / "task/task.json",
        {
            "programBindings": {
                "programId": program["id"],
                "stepIds": [s["id"] for s in program["steps"]],
                "recordNodeIds": [
                    n["id"] for n in program["nodes"] if n["kind"] in ("record", "variable")
                ],
                "analysisNodeIds": [n["id"] for n in program["nodes"] if n["kind"] == "analysis"],
            }
        },
    )
    write(
        root / "task/adapter.py",
        "import sys\n\ndef run_sessions(agent_fn, seed):\n    return []\n\n"
        "if __name__ == '__main__' and '--smoke-test' in sys.argv:\n    print('ok')\n",
    )
    write(root / "evaluation/evaluation.py", "def evaluate(sessions):\n    return {}\n")


@pytest.fixture
def community(tmp_path: Path) -> Path:
    """community/ with one paper and one valid contribution (no package)."""
    root = tmp_path / "community"
    paper = root / PAPER_DIR
    write(
        paper / "paper.json",
        {
            "schemaVersion": 1,
            "paperKey": KEY,
            "title": "Intentional Action and Side-Effects in Ordinary Language",
            "authors": ["Joshua Knobe"],
            "year": 2003,
            "doi": "10.1111/1468-0017.00215",
        },
    )
    write(paper / FOLDER / "program.json", knobe_program())
    write(paper / FOLDER / "CONTRIBUTION.json", contribution_meta())
    return root


def folder_of(community: Path, name: str = FOLDER) -> Path:
    return community / PAPER_DIR / name


# ---- paper keys ----------------------------------------------------------------------------


def test_doi_key_is_lowercase_without_prefix():
    assert make_paper_key("https://doi.org/10.1111/ABC.1", "t", 2000, "A B") == "doi:10.1111/abc.1"
    assert make_paper_key("doi: 10.1/x", "t", 2000, "A B") == "doi:10.1/x"


def test_title_key_without_doi():
    key = make_paper_key(None, "Intentional Action & Side-Effects!", 2003, "Joshua Knobe")
    assert key == "title:intentional-action-side-effects-2003-knobe"
    assert (
        make_paper_key("", "Café Müller", 1999, "Müller, Hans") == "title:cafe-muller-1999-muller"
    )
    assert make_paper_key("not-a-doi", "T", None, "Ada Lovelace") == "title:t-nd-lovelace"


def test_long_titles_are_cut_on_a_word_boundary():
    title = "word " * 30
    result = normalise_title(title)
    assert len(result) <= 60 and not result.endswith("-") and result.startswith("word-word")


def test_paper_key_to_dir_replaces_unsafe_characters():
    assert paper_key_to_dir(KEY) == PAPER_DIR
    assert paper_key_to_dir("doi:10.1002/(sici)1097<x>") == "doi_10.1002__sici_1097_x_"
    assert paper_key_to_dir("title:a-b-2000-c") == "title_a-b-2000-c"


def test_parse_contribution_dir():
    assert parse_contribution_dir("knobe-2003--gh-ada") == {
        "slug": "knobe-2003",
        "contributor": "gh-ada",
        "version": 1,
    }
    assert parse_contribution_dir("knobe-2003--gh-ada-v12")["version"] == 12
    assert parse_contribution_dir("knobe-2003") is None
    assert parse_contribution_dir("Knobe--x") is None


# ---- validator -----------------------------------------------------------------------------


def test_valid_contribution_passes(community, capsys):
    assert validate_contribution(folder_of(community)) == []
    assert validate_main([str(folder_of(community))]) == 0


def test_valid_contribution_with_package(community):
    folder = folder_of(community)
    make_package(folder, knobe_program())
    meta = contribution_meta(checks={"schema": True, "package": True, "simulation": None})
    write(folder / "CONTRIBUTION.json", meta)
    assert validate_contribution(folder) == []


def test_package_with_different_program_is_rejected(community):
    folder = folder_of(community)
    other = knobe_program()
    other["title"] = "Something else"
    make_package(folder, other)
    errors = validate_contribution(folder)
    assert any("different Study document" in e for e in errors)


def test_broken_package_reports_plain_message(community):
    folder = folder_of(community)
    make_package(folder, knobe_program())
    (folder / "package/knobe-2003/task/adapter.py").unlink()
    errors = validate_contribution(folder)
    assert any("runnable package did not pass" in e and "adapter.py" in e for e in errors)


def test_invalid_program_and_schema_messages(community, capsys):
    folder = folder_of(community)
    program = knobe_program()
    program["steps"] = [{"id": "x"}]
    write(folder / "program.json", program)
    meta = contribution_meta()
    del meta["appVersion"]
    meta["includesPdf"] = "no"
    write(folder / "CONTRIBUTION.json", meta)
    errors = validate_contribution(folder)
    assert any("program.json is not a valid Study document" in e for e in errors)
    assert any("'appVersion' is missing" in e for e in errors)
    assert any("includesPdf" in e for e in errors)
    assert validate_main([str(folder)]) == 1
    assert "cannot be accepted" in capsys.readouterr().err


def test_identity_mismatches(community):
    folder = folder_of(community)
    write(folder / "CONTRIBUTION.json", contribution_meta(slug="other", paperKey="doi:10.9/zzz"))
    errors = validate_contribution(folder)
    assert any("slug 'other'" in e for e in errors)
    assert any("but the folder is under" in e for e in errors)
    assert any("different papers" in e for e in errors)


def test_contributor_needs_github_or_hub(community):
    folder = folder_of(community)
    write(folder / "CONTRIBUTION.json", contribution_meta(contributor={"name": "Nobody"}))
    assert any("GitHub username or a HumanStudy Hub id" in e for e in validate_contribution(folder))
    hub = {"name": "Hub User", "hub": {"id": "hs-1", "signature": "abc"}}
    write(folder / "CONTRIBUTION.json", contribution_meta(contributor=hub))
    assert validate_contribution(folder) == []


def test_missing_paper_json_and_bad_folder_name(community):
    (community / PAPER_DIR / "paper.json").unlink()
    errors = validate_contribution(folder_of(community))
    assert any("paper.json" in e and "missing" in e for e in errors)
    bad = community / PAPER_DIR / "Bad Name"
    bad.mkdir()
    assert any("must look like" in e for e in validate_contribution(bad))


def test_stray_files_links_pdfs_and_size(community, monkeypatch):
    folder = folder_of(community)
    write(folder / "sources/notes.txt", "x")
    write(folder / "artifacts/paper.PDF", "%PDF")
    (folder / "artifacts/link").symlink_to("/etc/hosts")
    errors = validate_contribution(folder)
    assert any("'sources' does not belong" in e for e in errors)
    assert any("paper.PDF" in e and "PDF" in e for e in errors)
    assert any("is a link" in e for e in errors)

    write(folder / "CONTRIBUTION.json", contribution_meta(includesPdf=True))
    errors = validate_contribution(folder)
    assert not any("paper.PDF" in e for e in errors)

    monkeypatch.setattr("scripts.validate_contribution.MAX_FILE_BYTES", 3)
    assert any("limit is 25 MB" in e for e in validate_contribution(folder))


def test_checks_cannot_claim_missing_parts(community):
    folder = folder_of(community)
    meta = contribution_meta(checks={"schema": True, "package": True, "simulation": True})
    write(folder / "CONTRIBUTION.json", meta)
    errors = validate_contribution(folder)
    assert any("no package/ folder" in e for e in errors)
    assert any("simulations/ is empty" in e for e in errors)


def test_supersedes_must_exist(community):
    folder = folder_of(community)
    write(folder / "CONTRIBUTION.json", contribution_meta(supersedes="knobe-2003--gh-old"))
    assert any("supersedes" in e for e in validate_contribution(folder))
    (community / PAPER_DIR / "knobe-2003--gh-old").mkdir()
    assert not any("supersedes" in e for e in validate_contribution(folder))


# ---- PR rules ------------------------------------------------------------------------------


def run_rules(changes, author, head=None, base=None):
    head = head or {}
    base = base or {}
    return check_changes(changes, author, head.get, base.get)


NEW_PATH = f"community/{PAPER_DIR}/{FOLDER}"
OWNED = json.dumps(contribution_meta())
HUB_OWNED = json.dumps(
    contribution_meta(contributor={"name": "H", "hub": {"id": "hs-1", "signature": "sig"}})
)
HUB_UNSIGNED = json.dumps(contribution_meta(contributor={"name": "H", "hub": {"id": "hs-1"}}))
ADD = [("A", f"{NEW_PATH}/CONTRIBUTION.json"), ("A", f"{NEW_PATH}/program.json")]


def test_parse_name_status():
    text = "A\ta/b\nM\tc\nR100\told\tnew\n"
    assert parse_name_status(text) == [("A", "a/b"), ("M", "c"), ("D", "old"), ("A", "new")]


def test_author_matches_contributor_case_insensitively():
    head = {f"{NEW_PATH}/CONTRIBUTION.json": OWNED}
    errors, folders, add_only = run_rules(ADD, "ada", head)
    assert errors == [] and folders == [NEW_PATH] and add_only


def test_wrong_author_is_told_what_to_write():
    head = {f"{NEW_PATH}/CONTRIBUTION.json": OWNED}
    errors, _, add_only = run_rules(ADD, "mallory", head)
    assert len(errors) == 1 and "'mallory'" in errors[0] and not add_only


def test_bot_needs_signature():
    path = f"{NEW_PATH}/CONTRIBUTION.json"
    assert run_rules(ADD, "humanstudy-bot", {path: HUB_OWNED})[0] == []
    errors = run_rules(ADD, "humanstudy-bot", {path: HUB_UNSIGNED})[0]
    assert errors and "signed Hub id" in errors[0]


def test_editing_someone_elses_folder_is_refused_but_owner_and_maintainers_may():
    changes = [("M", f"{NEW_PATH}/program.json")]
    base = {f"{NEW_PATH}/CONTRIBUTION.json": OWNED}
    errors, _, add_only = run_rules(changes, "mallory", base=base)
    assert errors and "reviewed by its owner" in errors[0] and not add_only
    for author in ("Ada", "XuanL17", "humanstudy-bot"):
        errors, folders, add_only = run_rules(changes, author, base=base)
        assert errors == [] and folders == [NEW_PATH] and not add_only


def test_rewriting_owner_in_the_pr_does_not_grant_ownership():
    changes = [("M", f"{NEW_PATH}/CONTRIBUTION.json")]
    base = {f"{NEW_PATH}/CONTRIBUTION.json": OWNED}
    head = {
        f"{NEW_PATH}/CONTRIBUTION.json": json.dumps(
            contribution_meta(contributor={"name": "M", "github": "mallory"})
        )
    }
    assert run_rules(changes, "mallory", head, base)[0]


def test_new_file_in_existing_folder_counts_as_editing():
    base = {f"{NEW_PATH}/CONTRIBUTION.json": OWNED}
    changes = [("A", f"{NEW_PATH}/artifacts/x.md")]
    assert run_rules(changes, "mallory", base=base)[0]
    assert run_rules(changes, "ada", base=base)[1:] == ([NEW_PATH], False)


def test_paper_json_and_other_files():
    paper = f"community/{PAPER_DIR}/paper.json"
    head = {f"{NEW_PATH}/CONTRIBUTION.json": OWNED}
    errors, _, add_only = run_rules([("A", paper)] + ADD, "ada", head)
    assert errors == [] and add_only
    errors, _, add_only = run_rules([("M", paper)], "ada")
    assert errors and not add_only
    assert run_rules([("M", "community/README.md")], "ada")[0]
    assert run_rules([("M", "community/README.md")], "diana3135")[0] == []


def test_deleting_a_folder_needs_owner_and_validates_nothing():
    changes = [("D", f"{NEW_PATH}/program.json"), ("D", f"{NEW_PATH}/CONTRIBUTION.json")]
    base = {f"{NEW_PATH}/CONTRIBUTION.json": OWNED}
    assert run_rules(changes, "mallory", base=base)[0]
    errors, folders, add_only = run_rules(changes, "ada", base=base)
    assert errors == [] and folders == [] and not add_only


# ---- index ---------------------------------------------------------------------------------


def test_community_index_summarises_and_orders(community):
    second = community / PAPER_DIR / "knobe-2003--gh-bob"
    write(second / "program.json", knobe_program())
    write(
        second / "CONTRIBUTION.json",
        contribution_meta(
            contributor={"name": "Bob", "github": "bob"}, createdAt="2026-01-01T00:00:00Z"
        ),
    )
    write(second / "artifacts/simulations/2026-01-02/report.md", "r")
    make_package(second, knobe_program())
    other = community / "title_zzz-2000-a"
    write(other / "paper.json", {"paperKey": "title:zzz-2000-a", "title": "Z"})
    (community / "empty-paper").mkdir()

    index = bsi.build_community_index(community)

    assert [p["paperKey"] for p in index] == [KEY, "title:zzz-2000-a"]
    assert index[1]["contributions"] == []
    first = index[0]
    assert first["paper"]["title"].startswith("Intentional")
    assert [c["folder"] for c in first["contributions"]] == ["knobe-2003--gh-bob", FOLDER]
    bob, ada = first["contributions"]
    assert bob["hasPackage"] and bob["hasSimulation"] and not ada["hasPackage"]
    assert ada["contributor"] == {"name": "Ada Lovelace", "github": "Ada", "hubId": None}
    assert ada["counts"] == {
        "conditions": 0,
        "steps": 0,
        "openIssues": {"blocking": 0, "decision": 0, "check": 0},
    }
    assert bsi.build_community_index(community) == index


def test_open_issues_counted_by_severity(community):
    program = knobe_program()
    program["issues"] = [
        {
            "id": f"i{n}",
            "title": "t",
            "severity": sev,
            "reason": "",
            "impact": "",
            "suggestedAction": "",
            "evidenceIds": [],
        }
        for n, sev in enumerate(["blocking", "check", "check"])
    ]
    write(folder_of(community) / "program.json", program)
    entry = bsi.read_contribution_entry(folder_of(community))
    assert entry["counts"]["openIssues"] == {"blocking": 1, "decision": 0, "check": 2}


def test_write_index_keeps_studies_and_adds_community(tmp_path, monkeypatch):
    out = tmp_path / "idx.json"
    monkeypatch.setattr(bsi, "INDEX_OUTPUT", out)
    bsi.write_index([{"study_id": "study_001"}], [])
    assert json.loads(out.read_text()) == {"studies": [{"study_id": "study_001"}], "community": []}
    bsi.write_index([{"study_id": "study_001"}])
    assert list(json.loads(out.read_text())) == ["studies"]


def test_missing_community_dir_gives_empty_list(tmp_path):
    assert bsi.build_community_index(tmp_path / "nope") == []
