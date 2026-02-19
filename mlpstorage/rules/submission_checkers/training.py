"""
Training benchmark submission rules checker.

Validates training benchmark submissions (multiple runs).
"""

from typing import Optional

from mlpstorage.config import MODELS, PARAM_VALIDATION
from mlpstorage.rules.issues import Issue
from mlpstorage.rules.submission_checkers.base import MultiRunRulesChecker


class TrainingSubmissionRulesChecker(MultiRunRulesChecker):
    """Rules checker for training benchmark submissions."""

    supported_models = MODELS
    REQUIRED_RUNS = 5

    def check_num_runs(self) -> Optional[Issue]:
        """
        Require 5 runs for training benchmark closed submission.
        """
        num_runs = len(self.benchmark_runs)
        if num_runs < self.REQUIRED_RUNS:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Training submission requires {self.REQUIRED_RUNS} runs",
                parameter="num_runs",
                expected=self.REQUIRED_RUNS,
                actual=num_runs
            )

        return Issue(
            validation=PARAM_VALIDATION.CLOSED,
            message=f"Training submission has required {self.REQUIRED_RUNS} runs",
            parameter="num_runs",
            expected=self.REQUIRED_RUNS,
            actual=num_runs
        )
