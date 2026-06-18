"""
Base class for multi-run submission validation checkers.

MultiRunRulesCheckers validate groups of benchmark runs as a submission:
- Required number of runs per workload
- Consistency across runs (same model, accelerator, etc.)
- Aggregate metrics requirements
"""

from typing import Optional, List, Union

from mlpstorage_py.config import PARAM_VALIDATION
from mlpstorage_py.rules.base import RulesChecker
from mlpstorage_py.rules.issues import Issue


class MultiRunRulesChecker(RulesChecker):
    """
    Base class for multi-run submission validation.

    This class validates groups of benchmark runs as a complete submission.
    Subclasses should implement check_* methods for specific validations.
    """

    def __init__(self, benchmark_runs, *args, **kwargs):
        """
        Initialize the multi-run rules checker.

        Args:
            benchmark_runs: List or tuple of BenchmarkRun instances.
            *args: Additional positional arguments passed to parent.
            **kwargs: Must include 'logger' for parent class.

        Raises:
            TypeError: If benchmark_runs is not a list or tuple.
        """
        super().__init__(*args, **kwargs)
        if type(benchmark_runs) not in [list, tuple]:
            raise TypeError("benchmark_runs must be a list or tuple")
        self.benchmark_runs = benchmark_runs

    def check_runs_valid(self) -> Optional[Issue]:
        """
        Verify all individual runs passed validation.

        Returns:
            Issue if runs contain invalid or open categories.
        """
        category_set = {run.category for run in self.benchmark_runs}

        if PARAM_VALIDATION.INVALID in category_set:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message="Invalid runs found.",
                parameter="category",
                expected="OPEN or CLOSED",
                actual=[cat.value.upper() for cat in category_set]
            )
        elif PARAM_VALIDATION.OPEN in category_set:
            return Issue(
                validation=PARAM_VALIDATION.OPEN,
                message="All runs satisfy the OPEN or CLOSED category",
                parameter="category",
                expected="OPEN or CLOSED",
                actual=[cat.value.upper() for cat in category_set]
            )
        elif {PARAM_VALIDATION.CLOSED} == category_set:
            return Issue(
                validation=PARAM_VALIDATION.CLOSED,
                message="All runs satisfy the CLOSED category",
                parameter="category",
                expected="OPEN or CLOSED",
                actual=[cat.value.upper() for cat in category_set]
            )

        return None

    def check_run_consistency(self) -> Optional[Issue]:
        """
        Verify all runs have consistent configuration.

        Returns:
            Issue if runs have inconsistent models.
        """
        models = {run.model for run in self.benchmark_runs}
        if len(models) > 1:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message="Inconsistent models across runs in submission",
                parameter="model",
                expected="Single model",
                actual=list(models)
            )
        return None

    def check_benchmark_type_consistency(self) -> Optional[Issue]:
        """
        Verify all runs share the same benchmark_type.

        BenchmarkVerifier already rejects mixed-type inputs in its dispatch
        (verifier.py:_create_rules_checker), but a caller that constructs a
        MultiRunRulesChecker directly bypasses that guard. Enforcing it here
        defends against silent grouping bugs when a model name happens to
        coincide across benchmark types.
        """
        types = {run.benchmark_type for run in self.benchmark_runs}
        if len(types) > 1:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message="Inconsistent benchmark types across runs in submission",
                parameter="benchmark_type",
                expected="Single benchmark_type",
                actual=sorted(
                    t.name if t is not None else "None" for t in types
                ),
            )
        return None
