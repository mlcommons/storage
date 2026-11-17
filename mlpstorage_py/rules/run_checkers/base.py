"""
Base class for single-run validation checkers.

RunRulesCheckers validate individual benchmark runs for:
- Parameter correctness (model, accelerator, batch size, etc.)
- System requirements (memory, CPU, etc.)
- Configuration validity (allowed parameter overrides)
"""

from typing import Optional

from mlpstorage_py.config import PARAM_VALIDATION
from mlpstorage_py.rules.base import RulesChecker
from mlpstorage_py.rules.issues import Issue


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
        # Set benchmark_run BEFORE calling super().__init__() because
        # the parent class dynamically discovers check methods that access self.benchmark_run
        self.benchmark_run = benchmark_run
        super().__init__(*args, **kwargs)

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

    def check_daos_test_stand_in(self) -> Optional[Issue]:
        """Reject a run that DLIO served from its test stand-in.

        DLIO marks ``summary.json`` with ``daos_test_stand_in`` when
        ``storage_type: daos`` was served by a local directory.  Such a run
        is INVALID in every division.  A run without a summary (before
        execution) or of another benchmark type never carries the mark.
        """
        if getattr(self.benchmark_run, "daos_test_stand_in", False) is not True:
            return None
        return Issue(
            validation=PARAM_VALIDATION.INVALID,
            message=(
                "summary.json records daos_test_stand_in: DLIO served "
                "storage_type daos from its local directory test stand-in, "
                "not from DAOS. The results are not storage measurements "
                "and cannot be reported."
            ),
            parameter="daos_test_stand_in",
            expected=False,
            actual=True,
        )
