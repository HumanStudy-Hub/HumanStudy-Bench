"""One contributor rule for legacy index.json and protocol study.json (V080-PLATFORM §2.1, B2)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_contributors import main, verdict, with_contributor  # noqa: E402


def study(tmp_path, name, filename, contributors):
    d = tmp_path / name
    d.mkdir()
    data = {"title": "t"} if contributors is None else {"title": "t", "contributors": contributors}
    (d / filename).write_text(json.dumps(data))
    return d


def test_legacy_index_json(tmp_path):
    d = study(tmp_path, "study_900", "index.json", [{"name": "A", "github": "https://github.com/Alice/"}])
    assert verdict(d / "index.json", "alice") == "MATCH"
    assert verdict(d / "index.json", "bob") == "MISMATCH"
    assert main(["--author", "alice", str(d)]) == 0
    assert main(["--author", "bob", str(d)]) == 1


def test_protocol_study_json(tmp_path):
    d = study(tmp_path, "rev", "study.json", [{"name": "A", "github": "alice"}])
    assert main(["--author", "Alice", str(d)]) == 0
    empty = study(tmp_path, "rev2", "study.json", [])
    assert verdict(empty / "study.json", "alice") == "MISSING"
    assert main(["--author", "alice", str(empty)]) == 1


def test_index_json_wins_and_dirs_without_either_are_skipped(tmp_path, capsys):
    d = study(tmp_path, "study_901", "index.json", [{"name": "A", "github": "alice"}])
    (d / "study.json").write_text(json.dumps({"contributors": [{"github": "bob"}]}))
    assert main(["--author", "alice", str(d)]) == 0
    bare = tmp_path / "study_902"
    bare.mkdir()
    assert main(["--author", "alice", str(bare)]) == 0
    assert "study_902" not in capsys.readouterr().out


def test_with_contributor_appends_once():
    raw = json.dumps({"id": "x", "contributors": [{"name": "A", "github": "alice"}]}).encode()
    assert with_contributor(raw, "alice") == raw
    added = json.loads(with_contributor(raw, "bob", "Bob"))
    assert added["contributors"][-1] == {"name": "Bob", "github": "bob"}


def test_with_contributor_leaves_malformed_study_json_to_the_checks():
    for raw in (b"[]", b'"x"', b'{"contributors": "alice"}'):
        assert with_contributor(raw, "bob") == raw
