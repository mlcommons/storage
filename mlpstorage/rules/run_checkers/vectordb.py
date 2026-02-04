"""
VectorDB benchmark run rules checker.

Validates VectorDB benchmark parameters for individual runs.
Note: VectorDB is currently a preview benchmark, so all runs
automatically receive OPEN status regardless of other validations.
"""

from typing import Optional

from mlpstorage.config import (
    BENCHMARK_TYPES,
    PARAM_VALIDATION,
)
from mlpstorage.rules.issues import Issue
from mlpstorage.rules.run_checkers.base import RunRulesChecker


class VectorDBRunRulesChecker(RunRulesChecker):
    """Rules checker for VectorDB benchmarks.

    VectorDB benchmark validates vector database storage performance including:
    - Index building and search operations
    - Concurrent query handling
    - Various distance metrics and index types

    Currently in preview mode - all runs return OPEN status regardless
    of other validation results, as the benchmark is not yet accepted
    for closed submissions.
    """

    # Minimum requirements for valid VectorDB runs
    MIN_RUNTIME_SECONDS = 30

    def check_benchmark_type(self) -> Optional[Issue]:
        """Verify this is a VectorDB benchmark."""
        if self.benchmark_run.benchmark_type != BENCHMARK_TYPES.vector_database:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Invalid benchmark type: {self.benchmark_run.benchmark_type}",
                parameter="benchmark_type",
                expected=BENCHMARK_TYPES.vector_database,
                actual=self.benchmark_run.benchmark_type
            )
        return None

    def check_runtime(self) -> Optional[Issue]:
        """Verify benchmark runtime is valid."""
        runtime = self.benchmark_run.parameters.get('runtime', 60)

        if runtime < self.MIN_RUNTIME_SECONDS:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Runtime must be at least {self.MIN_RUNTIME_SECONDS} seconds",
                parameter="runtime",
                expected=f">= {self.MIN_RUNTIME_SECONDS}",
                actual=runtime
            )

        return None

    def check_preview_status(self) -> Optional[Issue]:
        """
        Return informational issue that VectorDB is in preview.

        This is always returned to inform users that this benchmark
        is not yet accepted for closed submissions.
        """
        return Issue(
            validation=PARAM_VALIDATION.OPEN,
            message="VectorDB benchmark is in preview status - not accepted for closed submissions",
            parameter="benchmark_status"
        )
