"""
Study class representing a single human study in the benchmark.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.core.exceptions import DataLoadError, ValidationError


@dataclass
class Study:
    """Represents a single human study in the benchmark."""
    
    id: str
    metadata: Dict[str, Any]
    specification: Dict[str, Any]
    ground_truth: Dict[str, Any]
    materials_path: Path
    
    # Pass thresholds
    PASS_THRESHOLD = 0.70  # 70% - minimum passing score for a study
    HIGH_QUALITY_THRESHOLD = 0.85  # 85% - high quality replication
    PERFECT_THRESHOLD = 1.00  # 100% - perfect replication
    
    @classmethod
    def load(cls, study_path: Path) -> "Study":
        """
        Load a study from disk.
        
        Args:
            study_path: Path to study directory
            
        Returns:
            Study object
            
        Raises:
            DataLoadError: If study files cannot be loaded
        """
        study_path = Path(study_path)
        if not study_path.exists():
            raise DataLoadError(f"Study directory not found: {study_path}")
        # study_path is the study root (e.g. data/studies/study_001); data lives under source/
        source_dir = study_path / "source"
        study_id = study_path.name
        if not source_dir.exists():
            raise DataLoadError(f"Study source directory not found: {source_dir}")

        try:
            # Load metadata
            with open(source_dir / "metadata.json", "r", encoding='utf-8', errors='replace') as f:
                metadata = json.load(f)
            
            # Load specification
            with open(source_dir / "specification.json", "r", encoding='utf-8', errors='replace') as f:
                specification = json.load(f)
            
            # Load ground truth
            with open(source_dir / "ground_truth.json", "r", encoding='utf-8', errors='replace') as f:
                ground_truth = json.load(f)
            
            materials_path = source_dir / "materials"
            
            return cls(
                id=study_id,
                metadata=metadata,
                specification=specification,
                ground_truth=ground_truth,
                materials_path=materials_path
            )
        
        except FileNotFoundError as e:
            raise DataLoadError(f"Required file missing in study {study_id}: {e}")
        except json.JSONDecodeError as e:
            raise DataLoadError(f"Invalid JSON in study {study_id}: {e}")
        except Exception as e:
            raise DataLoadError(f"Error loading study {study_id}: {e}")
    
    def get_validation_criteria(self) -> List[Dict[str, Any]]:
        """
        Get validation criteria for this study.
        
        Returns:
            List of validation test dictionaries
        """
        return self.ground_truth["validation_criteria"]["required_tests"]
    
    def get_materials(self, material_type: Optional[str] = None) -> Path:
        """
        Get path to study materials.
        
        Args:
            material_type: Type of material (e.g., 'stimuli', 'instructions')
                          If None, returns base materials path
        
        Returns:
            Path to requested materials
        """
        if material_type is None:
            return self.materials_path
        return self.materials_path / material_type
    
    def validate(self) -> bool:
        """
        Validate study data integrity.
        
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        # Check IDs match
        if self.metadata["id"] != self.id:
            raise ValidationError(f"Metadata ID mismatch: {self.metadata['id']} != {self.id}")
        
        if self.specification["study_id"] != self.id:
            raise ValidationError(f"Specification ID mismatch: {self.specification['study_id']} != {self.id}")
        
        if self.ground_truth["study_id"] != self.id:
            raise ValidationError(f"Ground truth ID mismatch: {self.ground_truth['study_id']} != {self.id}")
        
        # Check materials directory exists
        if not self.materials_path.exists():
            raise ValidationError(f"Materials directory not found: {self.materials_path}")
        
        return True
    
    def get_domain(self) -> str:
        """Get study domain."""
        return self.metadata["domain"]
    
    def get_difficulty(self) -> str:
        """Get study difficulty level."""
        return self.metadata["difficulty"]
    
    def get_tags(self) -> List[str]:
        """Get study tags."""
        return self.metadata.get("tags", [])
    
    def get_pass_threshold(self) -> float:
        """Get the pass threshold for this study (can be customized per study)."""
        return self.ground_truth.get("validation_criteria", {}).get(
            "pass_threshold", 
            self.PASS_THRESHOLD
        )
    
    def evaluate_pass_status(self, score: float) -> Dict[str, Any]:
        """
        Evaluate if a score passes this study.
        
        Args:
            score: Overall score for this study (0.0 to 1.0)
        
        Returns:
            Dictionary with pass evaluation:
            {
                "passed": bool,
                "grade": str,  # "fail", "pass", "high_quality", "perfect"
                "score": float,
                "threshold": float,
                "margin": float  # score - threshold
            }
        """
        threshold = self.get_pass_threshold()
        
        if score >= self.PERFECT_THRESHOLD:
            grade = "perfect"
            passed = True
        elif score >= self.HIGH_QUALITY_THRESHOLD:
            grade = "high_quality"
            passed = True
        elif score >= threshold:
            grade = "pass"
            passed = True
        else:
            grade = "fail"
            passed = False
        
        return {
            "passed": passed,
            "grade": grade,
            "score": score,
            "threshold": threshold,
            "margin": score - threshold
        }
    
    def __repr__(self) -> str:
        return f"Study(id='{self.id}', title='{self.metadata.get('title', 'Unknown')}')"
    
    def __str__(self) -> str:
        return f"{self.id}: {self.metadata.get('title', 'Unknown')}"
