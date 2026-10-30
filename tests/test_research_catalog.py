"""Regression cases for scientific identity, stale edits, complete content and source coverage."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.research.catalog import build_catalog, community_identity, digest, paper_identity, project_program, taxonomy_errors
from scripts.research.imports import psych201_records, reanalysis_records
from scripts.research.revisions import integrate, merge_assets, merge_json, revision_id, safe_path, validate_assets

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def program():
    return json.loads((ROOT / "tests/fixtures/research-program.json").read_text())


def assets(program):
    return {"program.json": json.dumps(program, ensure_ascii=False).encode()}


def approve(files):
    return {"candidateRevision": revision_id(files), "reviewer": "test-independent-curator",
            "decision": "accept", "verifiedCapabilities": {"traceable": False}}


def metadata(base=None):
    return {"paperKey": "doi:10.1234/test", "paper": {"doi": "10.1234/test", "title": "Fixture, not an empirical result", "authors": [], "year": 2026},
            "baseRevision": base, "intent": "reconstruct", "contributor": {"name": "Test curator"}}


def test_identity_links_doi_but_never_collapses_title_only_records():
    a = paper_identity({"doi": "https://doi.org/10.1234/ABC"}, "source:a")
    b = paper_identity({"doi": "doi:10.1234/abc"}, "source:b")
    assert a[0] == b[0]
    assert paper_identity({"title": "Same title"}, "source:a")[0] != paper_identity({"title": "Same title"}, "source:b")[0]


def test_legacy_slug_collisions_cannot_reassign_canonical_paper_identity():
    key = "title:a-legacy-slug-that-collides"
    a = community_identity({"title": "完整标题 A", "authors": ["Author"], "year": 2026}, key)
    b = community_identity({"title": "完整标题 B", "authors": ["Author"], "year": 2026}, key)
    assert a[0] != b[0]


def test_three_way_merge_preserves_simultaneous_independent_improvements():
    base = {"fields": [{"id": "sample", "value": 30}, {"id": "outcome", "value": "choice"}]}
    current = deepcopy(base); current["fields"][0]["value"] = 40
    proposal = deepcopy(base); proposal["fields"][1]["value"] = "risky choice"
    result, conflicts = merge_json(base, current, proposal)
    assert not conflicts
    assert result["fields"] == [{"id": "sample", "value": 40}, {"id": "outcome", "value": "risky choice"}]


def test_conflicting_interpretations_and_delete_modify_do_not_silently_win():
    base = {"id": "f", "value": "original"}
    result, conflicts = merge_json(base, {"id": "f", "value": "first"}, {"id": "f", "value": "second"})
    assert conflicts == ["/value"] and result["value"] == "first"
    result, conflicts = merge_json({"f": base}, {"f": {**base, "value": "changed"}}, {})
    assert conflicts and result["f"]["value"] == "changed"
    assert merge_json({"a": 1}, {"a": 1}, {}) == ({}, [])


def test_ordered_visibility_lists_are_not_treated_as_unordered_sets():
    result, conflicts = merge_json(["first"], ["first", "second"], ["first", "third"])
    assert conflicts and result == ["first", "second"]


def test_no_content_truncation_and_missing_information_is_allowed(program):
    program["nodes"][0]["fields"][0]["value"] = "完整材料 " * 120000 + "END-OF-SOURCE"
    report = validate_assets(assets(program))
    assert report["status"] == "passed", report
    assert "END-OF-SOURCE" in report["projection"]["searchText"]
    assert program["issues"] and not report["capabilities"]["replayable"]


def test_schema_and_reference_validation_rejects_missing_node(program):
    program["steps"][0]["actorIds"] = ["not-an-actor"]
    report = validate_assets(assets(program))
    assert report["status"] == "failed" and any("unknown reference" in e for e in report["errors"])


def test_pending_review_never_replaces_current_and_review_is_content_bound(tmp_path, program):
    files = assets(program); folder = tmp_path / "paper"
    assert integrate(folder, metadata(), files)["status"] == "needs_review"
    assert not (folder / "current.json").exists()
    first = integrate(folder, metadata(), files, review=approve(files))
    assert first["status"] == "accepted"
    changed = deepcopy(program); changed["title"] += " corrected"
    newer = assets(changed)
    denied = integrate(folder, metadata(first["revision"]), newer, review=approve(files))
    assert denied["status"] == "needs_review"
    assert json.loads((folder / "current.json").read_text())["revision"] == first["revision"]


def test_stale_contribution_preserves_newer_independent_change(tmp_path, program):
    folder = tmp_path / "paper"; original = assets(program)
    first = integrate(folder, metadata(), original, review=approve(original))
    p1 = deepcopy(program); p1["title"] += " revised title"
    second_files = assets(p1)
    second = integrate(folder, metadata(first["revision"]), second_files, review=approve(second_files))
    assert second["status"] == "accepted"
    p2 = deepcopy(program); p2["source"]["version"] += " checked"
    candidate, conflicts = merge_assets(original, second_files, assets(p2))
    assert not conflicts
    third = integrate(folder, metadata(first["revision"]), assets(p2), review=approve(candidate))
    assert third["status"] == "accepted"
    final = json.loads((folder / "revisions" / third["revision"] / "program.json").read_text())
    assert final["title"] == p1["title"] and final["source"]["version"] == p2["source"]["version"]
    assert integrate(folder, metadata(), assets(p2))["status"] == "needs_review"


def test_resource_path_cannot_publish_credentials_or_escape():
    for name in ["../program.json", "package/../secret", "sources/.env", "artifacts/chat/a.json", "/program.json", "sources/a\\b", "sources/.humanstudy/token"]:
        with pytest.raises(ValueError): safe_path(name)


def test_curation_requires_accountability_and_rejects_cycles():
    taxonomy = {"concepts": [{"id": "a", "label": "A", "definition": "Definition", "status": "reviewed", "broader": ["b"]},
                             {"id": "b", "label": "B", "definition": "Definition", "broader": ["a"]}]}
    errors = taxonomy_errors(taxonomy)
    assert any("reviewer" in e for e in errors) and any("cycle" in e for e in errors)


def test_public_import_does_not_claim_full_paper_or_run_transcripts():
    tree = {"sha": "a" * 40, "tree": [{"path": "demo/README.md", "type": "blob", "sha": "b" * 40},
                                      {"path": "demo/generate_prompts.py", "type": "blob", "sha": "c" * 40}]}
    calls = []
    def read(path):
        calls.append(path)
        return b"### Reference:\nAuthor (2022). Example.\n### Data source:\nhttps://osf.io/example\n"
    papers = psych201_records(tree, read)
    assert calls == ["demo/README.md"]
    assert papers[0]["sourceMetadata"]["substudyCoverage"] == "not-yet-verified"
    assert not papers[0]["capabilities"]["replayable"]
    assert next(r for r in papers[0]["resources"] if r["kind"] == "original-data-link")["access"] == "unverified"


def test_source_citation_is_not_the_whole_readme_and_full_content_is_preserved():
    tree = {"sha": "a" * 40, "tree": [{"path": "demo/README.md", "type": "blob", "sha": "b" * 40}]}
    raw = "## Reference:\nAuthor (2025). Complete citation.\n\n## Methods\n" + "Complete methods " * 2000 + "FINAL-METHOD\n## Data source:\nhttps://osf.io/example"
    paper = psych201_records(tree, lambda path: raw.encode())[0]
    assert paper["title"] == "Author (2025). Complete citation."
    assert "FINAL-METHOD" in paper["searchText"]


def test_catalog_hides_proposals_and_keeps_human_and_ai_results_distinct(program):
    projection = project_program(program)
    assert projection["effectEstimates"] == []
    assert len(projection["studies"]) == len(program["studies"])
    from scripts.build_studies_index import build_index
    catalog = build_catalog(ROOT, build_index())
    assert catalog["coverage"]["papers"] >= 15
    assert len({p["id"] for p in catalog["papers"]}) == len(catalog["papers"])
    framing = next(p for p in catalog["papers"] if "bench:study_003" in p["aliases"])
    assert "phenomenon:framing" in framing["conceptIds"]
    assert catalog["snapshot"] == build_catalog(ROOT, build_index())["snapshot"]


def test_recommended_program_replaces_its_original_legacy_card(tmp_path, program):
    (tmp_path / "catalog").mkdir()
    (tmp_path / "catalog/taxonomy.json").write_text(json.dumps({"version": "test", "concepts": [], "assignments": []}))
    legacy = [{"study_id": "study_005", "title": program["source"]["title"], "authors": ["Researcher"], "year": 2003}]
    original = build_catalog(tmp_path, legacy)["papers"][0]
    meta = metadata()
    meta["_resolvedPaper"] = {"id": original["id"], "aliases": original["aliases"], "legacyStudyId": "study_005"}
    files = assets(program)
    result = integrate(tmp_path / "community/doi_10.1234_test", meta, files, review=approve(files))
    catalog = build_catalog(tmp_path, legacy)
    assert len(catalog["papers"]) == 1
    assert catalog["papers"][0]["id"] == original["id"]
    assert catalog["papers"][0]["currentRevision"] == result["revision"]


def package_files(program, slim=True):
    """A minimal package in either layout (desktop #45)."""
    bindings = {"programId": program["id"], "stepIds": [s["id"] for s in program["steps"]],
                "recordNodeIds": [n["id"] for n in program["nodes"] if n["kind"] in {"record", "variable"}],
                "analysisNodeIds": [n["id"] for n in program["nodes"] if n["kind"] == "analysis"]}
    study = {"title": "Fixture", **({"programRef": {"path": "../../program.json", "id": program["id"]}} if slim else {"program": program})}
    files = {"study.json": study, "source/paper_metadata.json": {}, "materials/materials.json": {},
             "task/task.json": {"programBindings": bindings}}
    out = {"package/p/" + k: json.dumps(v).encode() for k, v in files.items()}
    out["package/p/task/adapter.py"] = b"def run_sessions(llm, seed, n, arms=None):\n    return []\n"
    out["package/p/evaluation/evaluation.py"] = b"def evaluate(sessions):\n    return {}\n"
    if not slim:
        out["package/p/source/evidence.json"] = b"[]"
        out["package/p/audit/missing_information.json"] = b"[]"
    return {**assets(program), **out}


def test_slim_and_standalone_packages_both_validate(program):
    assert validate_assets(package_files(program))["status"] == "passed"
    assert validate_assets(package_files(program, slim=False))["status"] == "passed"
    copy_without_views = {k: v for k, v in package_files(program, slim=False).items() if not k.endswith("evidence.json")}
    report = validate_assets(copy_without_views)
    assert report["status"] == "failed" and any("source/evidence.json" in e for e in report["errors"])
    wrong = package_files(program)
    wrong["package/p/study.json"] = json.dumps({"programRef": {"path": "../../program.json", "id": "other"}}).encode()
    assert validate_assets(wrong)["status"] == "failed"


def test_merge_keeps_a_slim_study_json_slim(program):
    files = package_files(program)
    merged, conflicts = merge_assets(files, files, files)
    assert not conflicts and "program" not in json.loads(merged["package/p/study.json"])


def test_capabilities_and_failure_cases_are_checked(program):
    files = package_files(program)
    caps = {"schemaVersion": 1, "read": {"status": "yes"}, "run": {"status": "partial", "reason": "pictures", "excluded": [{"what": "x", "reason": "picture"}]},
            "humanData": {"status": "no", "reason": "none"}, "authorCode": {"status": "no", "reason": "none"},
            "profiles": {"status": "no", "reason": "not yet"}, "reproduced": {"package": None, "author": None}}
    files["capabilities.json"] = json.dumps(caps).encode()
    files["artifacts/failures.json"] = json.dumps([{"pattern": "non-text-stimuli", "where": "e1", "what": "picture", "fallback": "not run"}]).encode()
    assert validate_assets(files)["status"] == "passed"
    caps["run"]["status"] = "maybe"
    files["capabilities.json"] = json.dumps(caps).encode()
    assert validate_assets(files)["status"] == "failed"
    files["capabilities.json"] = json.dumps({**caps, "run": {"status": "no"}}).encode()
    assert any("without a reason" in e for e in validate_assets(files)["errors"])


QUOTE = ("In the harm condition 82% said the chairman harmed the environment intentionally (n = 39); "
         "in the help condition 23% said he helped intentionally (n = 39).")


def effects_files(program):
    program = deepcopy(program)
    program["evidence"].append({"id": "e-result", "sourceId": "paper", "locator": {"page": 2}, "quote": QUOTE})
    effects = {"schemaVersion": 1, "programId": program["id"], "topics": ["moral judgment"],
               "hypotheses": [{"id": "h1", "nodeId": "hypothesis", "status": "tested", "phenomena": ["Side-effect effect"]}],
               "effects": [{"id": "harm_vs_help", "label": "Harm vs help", "role": "test", "hypothesisIds": ["h1"],
                            "studyId": "exp-1", "design": "between", "unit": "participant",
                            "outcome": {"variableId": "intentionality", "type": "binary", "higherMeans": "more say intentional"},
                            "contrast": {"kind": "two_groups", "treatment": "condition-harm", "reference": "condition-help",
                                         "conditionIds": ["condition-harm", "condition-help"]},
                            "prediction": {"direction": "positive"}, "analysisIds": ["analysis"], "resultIds": ["result-1"],
                            "compute": {"op": "proportion_difference", "field": "answer", "success": ["yes"]},
                            "reported": {"cells": {"condition-harm": {"n": 39, "percent": 82},
                                                   "condition-help": {"n": 39, "percent": 23}},
                                         "missing": [], "evidenceIds": ["e-result"]}}]}
    files = package_files(program)
    files["effects.json"] = json.dumps(effects).encode()
    return files, effects


def test_effects_are_checked_and_indexed_for_cross_paper_queries(tmp_path, program):
    files, effects = effects_files(program)
    assert validate_assets(files)["status"] == "passed", validate_assets(files)["errors"]
    bad = deepcopy(effects)
    bad["effects"][0]["reported"]["cells"]["condition-harm"]["n"] = 40  # not printed in the paper
    report = validate_assets({**files, "effects.json": json.dumps(bad).encode()})
    assert report["status"] == "failed" and any("is not written in its cited quotes" in e for e in report["errors"])

    (tmp_path / "catalog").mkdir()
    (tmp_path / "catalog/taxonomy.json").write_text(json.dumps({"version": "test", "concepts": [
        {"id": "phenomenon:side-effect", "facet": "phenomenon", "label": "Side-effect effect", "aliases": ["Knobe effect"], "definition": "test",
         "status": "reviewed", "reviewedBy": "test"}], "assignments": []}))
    folder = tmp_path / "community/doi_10.1234_test"
    candidate = integrate(folder, metadata(), files)["candidateRevision"]
    review = {**approve(files), "candidateRevision": candidate}
    assert integrate(folder, metadata(), files, review=review)["status"] == "accepted"
    paper = build_catalog(tmp_path, [])["papers"][0]
    [record] = paper["effectEstimates"]
    assert record["id"].endswith("#harm_vs_help") and record["computeOp"] == "proportion_difference"
    assert record["conceptSuggestions"] == ["phenomenon:side-effect"] and paper["conceptIds"] == []
    assert record["reported"]["cells"]["condition-harm"]["percent"] == 82 and record["reportedPages"] == [2]
