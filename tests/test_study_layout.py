import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class StudyLayoutTests(unittest.TestCase):
    def test_extension_and_archive_packages_stay_out_of_default_registry(self):
        expected_extended = {"study_016", "study_017", "study_019"}
        expected_archived = {"study_018"}

        self.assertEqual(
            {
                path.name
                for path in (REPO_ROOT / "extended_study").glob("study_*")
                if path.is_dir()
            },
            expected_extended,
        )
        self.assertEqual(
            {
                path.name
                for path in (REPO_ROOT / "archived_study").glob("study_*")
                if path.is_dir()
            },
            expected_archived,
        )

        default_studies = {
            path.name
            for path in (REPO_ROOT / "studies").glob("study_*")
            if path.is_dir()
        }
        self.assertTrue(
            (expected_extended | expected_archived).isdisjoint(default_studies)
        )

        index = json.loads(
            (REPO_ROOT / "co_website" / "data" / "studies_index.json").read_text(
                encoding="utf-8"
            )
        )
        indexed = {entry["study_id"] for entry in index["studies"]}
        self.assertTrue((expected_extended | expected_archived).isdisjoint(indexed))


if __name__ == "__main__":
    unittest.main()
