"""
Issue dataclass for validation results.

This module defines the Issue class used to represent validation findings
during benchmark parameter and result validation.
"""

from dataclasses import dataclass
from typing import Any, Optional

from mlpstorage.config import PARAM_VALIDATION


@dataclass
class Issue:
    """Represents a validation issue found during rules checking.

    Attributes:
        validation: The validation category (CLOSED, OPEN, or INVALID).
        message: Human-readable description of the issue.
        parameter: The parameter name that caused the issue (optional).
        expected: The expected value for the parameter (optional).
        actual: The actual value found (optional).
        severity: Severity level ('error', 'warning', 'info').
    """
    validation: PARAM_VALIDATION
    message: str
    parameter: Optional[str] = None
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    severity: str = "error"

    def __str__(self) -> str:
        """Format issue as a human-readable string."""
        result = f"[{self.validation.value.upper()}] {self.message}"
        if self.parameter:
            result += f" (Parameter: {self.parameter}"
            if self.expected is not None and self.actual is not None:
                result += f", Expected: {self.expected}, Actual: {self.actual}"
            result += ")"
        return result

    def to_dict(self) -> dict:
        """Convert issue to dictionary for serialization."""
        return {
            'validation': self.validation.value,
            'message': self.message,
            'parameter': self.parameter,
            'expected': self.expected,
            'actual': self.actual,
            'severity': self.severity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Issue':
        """Create Issue from dictionary."""
        return cls(
            validation=PARAM_VALIDATION(data['validation']),
            message=data['message'],
            parameter=data.get('parameter'),
            expected=data.get('expected'),
            actual=data.get('actual'),
            severity=data.get('severity', 'error'),
        )
