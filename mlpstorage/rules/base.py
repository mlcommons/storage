"""
Base classes for rules checkers.

This module defines the abstract base classes for rule checking:
- RulesChecker: Base class for all rule checkers
- RuleState: Enum for rule states
"""

import abc
import enum
from typing import List, Optional

from mlpstorage.config import PARAM_VALIDATION
from mlpstorage.rules.issues import Issue


class RuleState(enum.Enum):
    """State of a rule check."""
    OPEN = "open"
    CLOSED = "closed"
    INVALID = "invalid"


class RulesChecker(abc.ABC):
    """
    Base class for rule checkers that verify benchmark parameters.

    Subclasses should implement check_* methods that return Issue instances
    or lists of Issue instances. The run_checks() method will automatically
    discover and run all methods that start with 'check_'.
    """

    def __init__(self, logger, *args, **kwargs):
        """
        Initialize the rules checker.

        Args:
            logger: Logger instance for output.
        """
        self.logger = logger
        self.issues: List[Issue] = []

        # Dynamically find all check methods
        self.check_methods = [
            getattr(self, method) for method in dir(self)
            if callable(getattr(self, method)) and method.startswith('check_')
        ]

    def run_checks(self) -> List[Issue]:
        """
        Run all check methods and return a list of issues.

        Each check method should return:
        - None if no issue found
        - An Issue instance if one issue found
        - A list of Issue instances if multiple issues found

        Returns:
            List of all issues found during checking.
        """
        self.issues = []
        for check_method in self.check_methods:
            try:
                self.logger.debug(f"Running check {check_method.__name__}")
                method_issues = check_method()
                if method_issues:
                    if isinstance(method_issues, list):
                        self.issues.extend(method_issues)
                    else:
                        self.issues.append(method_issues)
            except Exception as e:
                self.logger.error(f"Error running check {check_method.__name__}: {e}")
                self.issues.append(Issue(
                    validation=PARAM_VALIDATION.INVALID,
                    message=f"Check {check_method.__name__} failed with error: {e}",
                    severity="error"
                ))

        return self.issues

    def get_category(self) -> PARAM_VALIDATION:
        """
        Determine the overall category based on issues.

        Returns:
            PARAM_VALIDATION.INVALID if any invalid issues
            PARAM_VALIDATION.OPEN if any open issues
            PARAM_VALIDATION.CLOSED if all checks passed for closed
        """
        has_invalid = any(i.validation == PARAM_VALIDATION.INVALID for i in self.issues)
        has_open = any(i.validation == PARAM_VALIDATION.OPEN for i in self.issues)

        if has_invalid:
            return PARAM_VALIDATION.INVALID
        elif has_open:
            return PARAM_VALIDATION.OPEN
        else:
            return PARAM_VALIDATION.CLOSED
