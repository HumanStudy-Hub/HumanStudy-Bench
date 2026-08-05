import json
from pathlib import Path

from scripts.build_studies_index import read_study_entry


def test_reads_agent_study_json_from_nested_package(tmp_path: Path) -> None:
    study = tmp_path / "study_016"
    package = study / "paper-slug"
    package.mkdir(parents=True)
    (package / "study.json").write_text(json.dumps({
        "paper": {
            "title": "A contributed paper",
            "authors": [{"name": "First Author"}, "Second Author"],
            "publication_year": 2026,
            "abstract": "Paper abstract",
        },
        "contributors": [{"name": "Researcher", "github": "researcher"}],
    }))

    entry = read_study_entry(study)

    assert entry == {
        "study_id": "study_016",
        "title": "A contributed paper",
        "authors": ["First Author", "Second Author"],
        "year": 2026,
        "description": "Paper abstract",
        "contributors": [{"name": "Researcher", "github": "https://github.com/researcher"}],
    }
