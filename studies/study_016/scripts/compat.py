"""Standalone fallback. In HumanStudy-Bench use its actual BaseStudyConfig."""
class BaseStudyConfig:
    def __init__(self,study_path,specification):
        self.study_id=specification['study_id']
        self.specification=specification

