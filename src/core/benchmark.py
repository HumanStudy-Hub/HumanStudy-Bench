"""
Main benchmark class for HumanStudyBench.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm

from src.core.study import Study
from src.core.exceptions import StudyNotFoundError, DataLoadError
from src.agents.base_agent import BaseAgent
# Delay Scorer import to avoid circular dependency
# from src.evaluation.scorer import Scorer


class HumanStudyBench:
    """Main benchmark class for loading and evaluating studies."""

    # Benchmark-level pass criteria
    BENCHMARK_PASS_THRESHOLD = 0.60  # Minimum average score across all studies
    MIN_PASS_RATE = 0.50  # Minimum percentage of studies that must pass
    GOOD_THRESHOLD = 0.75  # Good performance threshold
    EXCELLENT_THRESHOLD = 0.85  # Excellent performance threshold

    def __init__(self, data_dir: str | Path, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the benchmark.

        Args:
            data_dir: Path to data directory containing studies
            config: Optional configuration dictionary
        """
        self.data_dir = Path(data_dir)
        self.studies_dir = self.data_dir / "studies"
        self.schemas_dir = self.data_dir / "schemas"
        self.config = config or {}

        # Load registry
        self.registry = self._load_registry()

        # Cache for loaded studies
        self.studies: Dict[str, Study] = {}

    def _load_registry(self) -> Dict[str, Any]:
        """Load the study registry."""
        registry_path = self.data_dir / "registry.json"

        if not registry_path.exists():
            studies = [
                {
                    "id": path.name,
                    "status": "active",
                }
                for path in sorted(self.studies_dir.glob("study_*"))
                if path.is_dir()
            ]
            return {
                "version": "hub-directory-scan",
                "total_studies": len(studies),
                "studies": studies,
            }

        with open(registry_path, "r", encoding='utf-8', errors='replace') as f:
            return json.load(f)

    def load_study(self, study_id: str) -> Study:
        """
        Load a specific study by ID.

        Args:
            study_id: Study identifier (e.g., 'study_003')

        Returns:
            Study object

        Raises:
            StudyNotFoundError: If study doesn't exist
        """
        # Check cache first
        if study_id in self.studies:
            return self.studies[study_id]

        study_info = self._get_study_info(study_id)
        if study_info is None and not (self.studies_dir / study_id).exists():
            raise StudyNotFoundError(f"Study '{study_id}' not found")

        # Load study
        study_path = self.studies_dir / study_id
        if not study_path.exists():
            raise StudyNotFoundError(f"Study directory not found: {study_path}")

        study = Study.load(study_path)
        self.studies[study_id] = study

        return study

    def _get_study_info(self, study_id: str) -> Optional[Dict[str, Any]]:
        """Get study info from registry."""
        for study_info in self.registry["studies"]:
            if study_info["id"] == study_id:
                return study_info
        return None

    def get_studies(
        self,
        domain: Optional[str] = None,
        difficulty: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: str = "active"
    ) -> List[Study]:
        """
        Get filtered list of studies.

        Args:
            domain: Filter by domain (e.g., 'cognitive_psychology')
            difficulty: Filter by difficulty ('easy', 'medium', 'hard')
            tags: Filter by tags (studies must have at least one matching tag)
            status: Filter by status (default: 'active')

        Returns:
            List of Study objects matching filters
        """
        matching_studies = []

        for study_info in self.registry["studies"]:
            # Apply filters
            if status and study_info.get("status") != status:
                continue

            if domain and study_info.get("domain") != domain:
                continue

            if difficulty and study_info.get("difficulty") != difficulty:
                continue

            if tags:
                study_tags = study_info.get("tags", [])
                if not any(tag in study_tags for tag in tags):
                    continue

            # Load study
            try:
                study = self.load_study(study_info["id"])
                matching_studies.append(study)
            except Exception as e:
                print(f"Warning: Failed to load study {study_info['id']}: {e}")

        return matching_studies

    def get_all_study_ids(self, status: str = "active") -> List[str]:
        """
        Get list of all study IDs.

        Args:
            status: Filter by status (default: 'active')

        Returns:
            List of study IDs
        """
        return [
            study_info["id"]
            for study_info in self.registry["studies"]
            if not status or study_info.get("status", "active") == status
        ]

    def evaluate(
        self,
        agent: BaseAgent,
        study_ids: Optional[List[str]] = None,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate an agent on specified studies.

        Args:
            agent: Agent instance (must inherit from BaseAgent)
            study_ids: List of study IDs to evaluate (None = all active studies)
            verbose: Show progress bar
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary containing evaluation results
        """
        # Get studies to evaluate
        if study_ids is None:
            study_ids = self.get_all_study_ids(status="active")

        # Initialize scorer
        # Import here to avoid circular dependency
        from src.evaluation.scorer import Scorer
        scorer = Scorer(config=self.config.get("evaluation", {}))

        # Evaluate each study
        study_results = []
        passed_count = 0

        iterator = tqdm(study_ids, desc="Evaluating studies") if verbose else study_ids

        for study_id in iterator:
            try:
                # Load study
                study = self.load_study(study_id)

                # Run agent
                agent_results = agent.run_study(study.specification)

                # Score results
                score_report = scorer.score_study(study, agent_results)

                # Track results
                study_results.append({
                    "study_id": study_id,
                    "study_title": study.metadata["title"],
                    "domain": study.get_domain(),
                    "difficulty": study.get_difficulty(),
                    "score": score_report["total_score"],
                    "passed": score_report["passed"],
                    "total_tests": score_report["total_tests"],
                    "test_results": score_report["tests"]
                })

                if score_report["passed"]:
                    passed_count += 1

            except Exception as e:
                print(f"Error evaluating study {study_id}: {e}")
                study_results.append({
                    "study_id": study_id,
                    "error": str(e),
                    "score": 0.0,
                    "passed": False
                })

        # Aggregate results
        total_studies = len(study_results)
        overall_score = sum(r["score"] for r in study_results if "score" in r) / total_studies if total_studies > 0 else 0.0

        # Aggregate by domain and difficulty
        by_domain = self._aggregate_by_category(study_results, "domain")
        by_difficulty = self._aggregate_by_category(study_results, "difficulty")

        return {
            "overall_score": overall_score,
            "pass_rate": passed_count / total_studies if total_studies > 0 else 0.0,
            "studies_passed": passed_count,
            "studies_completed": total_studies,
            "total_studies": total_studies,
            "study_results": study_results,
            "by_domain": by_domain,
            "by_difficulty": by_difficulty,
            "agent_name": agent.__class__.__name__,
            "config": self.config
        }

    def _aggregate_by_category(self, study_results: List[Dict], category: str) -> Dict[str, Any]:
        """Aggregate results by a category (domain or difficulty)."""
        category_results = {}

        for result in study_results:
            if category not in result:
                continue

            cat_value = result[category]
            if cat_value not in category_results:
                category_results[cat_value] = {
                    "scores": [],
                    "passed": 0,
                    "total": 0
                }

            category_results[cat_value]["scores"].append(result.get("score", 0.0))
            category_results[cat_value]["total"] += 1
            if result.get("passed", False):
                category_results[cat_value]["passed"] += 1

        # Compute averages
        for cat_value, data in category_results.items():
            scores = data["scores"]
            data["average_score"] = sum(scores) / len(scores) if scores else 0.0
            data["pass_rate"] = data["passed"] / data["total"] if data["total"] > 0 else 0.0
            data["n_studies"] = data["total"]
            del data["scores"]  # Remove raw scores

        return category_results

    def evaluate_benchmark_pass(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate if evaluation results pass the benchmark.

        Uses dual criteria:
        1. Average score >= BENCHMARK_PASS_THRESHOLD (60%)
        2. Pass rate >= MIN_PASS_RATE (50%)

        Args:
            evaluation_results: Results from evaluate() method

        Returns:
            Dictionary with benchmark pass evaluation:
            {
                "passed": bool,
                "grade": str,  # "fail", "pass", "good", "excellent"
                "overall_score": float,
                "pass_rate": float,
                "score_passed": bool,
                "rate_passed": bool,
                "feedback": str
            }
        """
        overall_score = evaluation_results.get("overall_score", 0.0)
        pass_rate = evaluation_results.get("pass_rate", 0.0)

        # Check both criteria
        score_passed = overall_score >= self.BENCHMARK_PASS_THRESHOLD
        rate_passed = pass_rate >= self.MIN_PASS_RATE
        passed = score_passed and rate_passed

        # Determine grade
        if overall_score >= self.EXCELLENT_THRESHOLD and pass_rate >= 0.80:
            grade = "excellent"
        elif overall_score >= self.GOOD_THRESHOLD and pass_rate >= 0.65:
            grade = "good"
        elif passed:
            grade = "pass"
        else:
            grade = "fail"

        # Generate feedback
        feedback_parts = []
        if not score_passed:
            feedback_parts.append(
                f"Overall score {overall_score:.1%} below threshold {self.BENCHMARK_PASS_THRESHOLD:.1%}"
            )
        if not rate_passed:
            feedback_parts.append(
                f"Pass rate {pass_rate:.1%} below minimum {self.MIN_PASS_RATE:.1%}"
            )
        if passed:
            if grade == "excellent":
                feedback_parts.append("Excellent performance across the benchmark!")
            elif grade == "good":
                feedback_parts.append("Good performance across the benchmark")
            else:
                feedback_parts.append("Meets minimum benchmark requirements")

        feedback = "; ".join(feedback_parts) if feedback_parts else "No evaluation data"

        return {
            "passed": passed,
            "grade": grade,
            "overall_score": overall_score,
            "pass_rate": pass_rate,
            "score_passed": score_passed,
            "rate_passed": rate_passed,
            "feedback": feedback,
            "studies_passed": evaluation_results.get("studies_passed", 0),
            "total_studies": evaluation_results.get("total_studies", 0)
        }

    def get_registry(self) -> Dict[str, Any]:
        """Get the study registry."""
        return self.registry

    def __repr__(self) -> str:
        n_studies = self.registry.get("total_studies", len(self.registry.get("studies", [])))
        return f"HumanStudyBench(total_studies={n_studies}, version={self.registry['version']})"
