"""
Validator interface definitions for mlpstorage.

This module defines the abstract interface for benchmark validators,
which check that benchmark parameters and results meet the requirements
for OPEN or CLOSED submissions.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from mlpstorage.rules import Issue


class ValidationCategory(Enum):
    """Categories for validation results."""
    CLOSED = "closed"  # Meets all requirements for closed submission
    OPEN = "open"      # Valid but only for open submission
    INVALID = "invalid"  # Does not meet minimum requirements


@dataclass
class ValidationResult:
    """Result of a validation check.

    Attributes:
        category: The validation category (CLOSED, OPEN, or INVALID).
        issues: List of Issue objects describing validation findings.
        warnings: List of warning messages (non-blocking issues).
        is_valid: Whether the run is valid (CLOSED or OPEN).
        is_closed: Whether the run meets CLOSED requirements.
        summary: Human-readable summary of the validation result.
    """
    category: ValidationCategory
    issues: List['Issue'] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Check if the run is valid (can be submitted)."""
        return self.category != ValidationCategory.INVALID

    @property
    def is_closed(self) -> bool:
        """Check if the run meets CLOSED submission requirements."""
        return self.category == ValidationCategory.CLOSED

    def get_open_issues(self) -> List['Issue']:
        """Get issues that prevent CLOSED submission."""
        from mlpstorage.config import PARAM_VALIDATION
        return [i for i in self.issues if i.validation == PARAM_VALIDATION.OPEN]

    def get_invalid_issues(self) -> List['Issue']:
        """Get issues that make the run invalid."""
        from mlpstorage.config import PARAM_VALIDATION
        return [i for i in self.issues if i.validation == PARAM_VALIDATION.INVALID]


@dataclass
class ClosedRequirements:
    """Requirements for CLOSED submission.

    Attributes:
        min_runs: Minimum number of runs required.
        allowed_param_overrides: Parameters that can be overridden.
        required_checks: List of check names that must pass.
        description: Human-readable description of requirements.
    """
    min_runs: int = 1
    allowed_param_overrides: List[str] = field(default_factory=list)
    required_checks: List[str] = field(default_factory=list)
    description: str = ""


class ValidatorInterface(ABC):
    """Interface for benchmark parameter validators.

    Validators check benchmark configuration and results against
    MLPerf Storage rules to determine if a submission qualifies
    for CLOSED or OPEN division.

    Example:
        class TrainingValidator(ValidatorInterface):
            def validate_pre_run(self, benchmark, args) -> ValidationResult:
                issues = []
                # Check dataset size meets requirements
                if not self._check_dataset_size(args):
                    issues.append(Issue(...))
                return ValidationResult(
                    category=self._determine_category(issues),
                    issues=issues
                )
    """

    @abstractmethod
    def validate_pre_run(self, benchmark, args) -> ValidationResult:
        """Validate before benchmark execution.

        Checks configuration and parameters before running the benchmark.
        This allows early detection of configuration issues.

        Args:
            benchmark: The benchmark instance.
            args: Parsed command-line arguments.

        Returns:
            ValidationResult indicating if configuration is valid.
        """
        pass

    @abstractmethod
    def validate_post_run(self, benchmark, results: Dict[str, Any]) -> ValidationResult:
        """Validate after benchmark execution.

        Checks results and metrics after the benchmark completes.
        This validates runtime behavior and output.

        Args:
            benchmark: The benchmark instance.
            results: Dictionary of benchmark results and metrics.

        Returns:
            ValidationResult indicating if results are valid.
        """
        pass

    @abstractmethod
    def get_closed_requirements(self) -> ClosedRequirements:
        """Return requirements for closed submission.

        Returns:
            ClosedRequirements describing what's needed for CLOSED.
        """
        pass

    @abstractmethod
    def format_validation_message(self, result: ValidationResult) -> str:
        """Format validation result as user-friendly message.

        Args:
            result: The validation result to format.

        Returns:
            Human-readable string explaining the validation outcome.
        """
        pass


class SubmissionValidatorInterface(ABC):
    """Interface for multi-run submission validators.

    Validates groups of benchmark runs as a complete submission,
    checking requirements like minimum run count and consistency.
    """

    @abstractmethod
    def validate_submission(self, runs: List[Any]) -> ValidationResult:
        """Validate a group of benchmark runs as a submission.

        Args:
            runs: List of BenchmarkRun instances.

        Returns:
            ValidationResult for the entire submission.
        """
        pass

    @abstractmethod
    def get_required_run_count(self) -> int:
        """Return the required number of runs for a valid submission."""
        pass

    @abstractmethod
    def check_run_consistency(self, runs: List[Any]) -> List[str]:
        """Check that all runs in submission are consistent.

        Args:
            runs: List of BenchmarkRun instances.

        Returns:
            List of inconsistency warning messages.
        """
        pass
