"""
Base class for single-run validation checkers.

RunRulesCheckers validate individual benchmark runs for:
- Parameter correctness (model, accelerator, batch size, etc.)
- System requirements (memory, CPU, etc.)
- Configuration validity (allowed parameter overrides)
"""

from mlpstorage.rules.base import RulesChecker


class RunRulesChecker(RulesChecker):
    """
    Base class for single-run validation.

    This class provides the foundation for validating individual benchmark runs.
    Subclasses should implement check_* methods for specific validations.

    Each check method should:
    1. Return None if the check passes for CLOSED submission
    2. Return Issue with PARAM_VALIDATION.OPEN if allowed for OPEN only
    3. Return Issue with PARAM_VALIDATION.INVALID if the check fails
    """

    def __init__(self, benchmark_run, *args, **kwargs):
        """
        Initialize the run rules checker.

        Args:
            benchmark_run: BenchmarkRun instance to validate.
            *args: Additional positional arguments passed to parent.
            **kwargs: Must include 'logger' for parent class.
        """
        super().__init__(*args, **kwargs)
        self.benchmark_run = benchmark_run

    @property
    def parameters(self):
        """Access benchmark run parameters."""
        return self.benchmark_run.parameters

    @property
    def override_parameters(self):
        """Access override parameters."""
        return self.benchmark_run.override_parameters

    @property
    def system_info(self):
        """Access system information."""
        return self.benchmark_run.system_info
