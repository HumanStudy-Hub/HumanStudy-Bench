"""
Custom exceptions for HumanStudyBench.
"""


class HumanStudyBenchError(Exception):
    """Base exception for HumanStudyBench."""
    pass


class StudyNotFoundError(HumanStudyBenchError):
    """Raised when a requested study cannot be found."""
    pass


class ValidationError(HumanStudyBenchError):
    """Raised when data validation fails."""
    pass


class SchemaError(HumanStudyBenchError):
    """Raised when schema validation fails."""
    pass


class AgentError(HumanStudyBenchError):
    """Raised when agent execution fails."""
    pass


class ConfigurationError(HumanStudyBenchError):
    """Raised when configuration is invalid."""
    pass


class DataLoadError(HumanStudyBenchError):
    """Raised when data cannot be loaded."""
    pass
