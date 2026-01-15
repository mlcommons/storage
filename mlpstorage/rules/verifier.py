"""
Benchmark verifier for validating runs against submission rules.

This module provides the BenchmarkVerifier class that orchestrates
validation of benchmark runs using the appropriate rules checkers.
"""

from mlpstorage.config import BENCHMARK_TYPES, PARAM_VALIDATION
from mlpstorage.rules.models import BenchmarkRun
from mlpstorage.rules.run_checkers import TrainingRunRulesChecker, CheckpointingRunRulesChecker
from mlpstorage.rules.submission_checkers import TrainingSubmissionRulesChecker, CheckpointSubmissionRulesChecker


class BenchmarkVerifier:
    """
    Verifies benchmark runs against submission rules.

    Accepts various input types:
        - BenchmarkRun instances (preferred)
        - Benchmark instances (live benchmarks, pre-execution)
        - str paths to result directories (post-execution)

    Usage:
        # Single run verification
        verifier = BenchmarkVerifier(benchmark_run, logger=logger)
        result = verifier.verify()

        # From a Benchmark instance
        verifier = BenchmarkVerifier(training_benchmark, logger=logger)

        # From a result directory
        verifier = BenchmarkVerifier("/path/to/results", logger=logger)

        # Multi-run verification
        verifier = BenchmarkVerifier(run1, run2, run3, logger=logger)
    """

    def __init__(self, *sources, logger=None):
        """
        Initialize the verifier.

        Args:
            *sources: BenchmarkRun instances, Benchmark instances, or result directory paths.
            logger: Logger instance for output.

        Raises:
            ValueError: If no sources provided or unsupported benchmark type.
            TypeError: If unsupported source type.
        """
        self.logger = logger
        self.issues = []

        if len(sources) == 0:
            raise ValueError("At least one source is required")

        # Convert all sources to BenchmarkRun instances
        self.benchmark_runs = []
        for source in sources:
            if isinstance(source, BenchmarkRun):
                self.benchmark_runs.append(source)
            elif isinstance(source, str):
                # Assume it's a result directory path
                self.benchmark_runs.append(BenchmarkRun.from_result_dir(source, logger))
            elif "mlpstorage.benchmarks." in str(type(source)):
                # It's a Benchmark instance - use the factory method
                self.benchmark_runs.append(BenchmarkRun.from_benchmark(source, logger))
            else:
                raise TypeError(f"Unsupported source type: {type(source)}. "
                               f"Expected BenchmarkRun, Benchmark instance, or result directory path.")

        # Determine mode
        if len(self.benchmark_runs) == 1:
            self.mode = "single"
        else:
            self.mode = "multi"

        # Create appropriate rules checker
        self._create_rules_checker()

    def _create_rules_checker(self):
        """Create the appropriate rules checker based on mode and benchmark type."""
        if self.mode == "single":
            benchmark_run = self.benchmark_runs[0]
            if benchmark_run.benchmark_type == BENCHMARK_TYPES.training:
                self.rules_checker = TrainingRunRulesChecker(benchmark_run, logger=self.logger)
            elif benchmark_run.benchmark_type == BENCHMARK_TYPES.checkpointing:
                self.rules_checker = CheckpointingRunRulesChecker(benchmark_run, logger=self.logger)
            else:
                raise ValueError(f"Unsupported benchmark type: {benchmark_run.benchmark_type}")

        elif self.mode == "multi":
            benchmark_types = {br.benchmark_type for br in self.benchmark_runs}
            if len(benchmark_types) > 1:
                raise ValueError(f"Multi-run verification requires all runs are from the same "
                               f"benchmark type. Got types: {benchmark_types}")
            benchmark_type = benchmark_types.pop()

            if benchmark_type == BENCHMARK_TYPES.training:
                self.rules_checker = TrainingSubmissionRulesChecker(self.benchmark_runs, logger=self.logger)
            elif benchmark_type == BENCHMARK_TYPES.checkpointing:
                self.rules_checker = CheckpointSubmissionRulesChecker(self.benchmark_runs, logger=self.logger)
            else:
                raise ValueError(f"Unsupported benchmark type for multi-run: {benchmark_type}")

    def verify(self) -> PARAM_VALIDATION:
        """
        Run verification and return the overall category.

        Returns:
            PARAM_VALIDATION indicating CLOSED, OPEN, or INVALID.
        """
        run_ids = [br.run_id for br in self.benchmark_runs]

        if self.mode == "single":
            self.logger.status(f"Verifying benchmark run for {run_ids[0]}")
        elif self.mode == "multi":
            self.logger.status(f"Verifying benchmark runs for {', '.join(str(rid) for rid in run_ids)}")

        self.issues = self.rules_checker.run_checks()

        num_invalid = 0
        num_open = 0
        num_closed = 0

        for issue in self.issues:
            if issue.validation == PARAM_VALIDATION.INVALID:
                self.logger.error(f"INVALID: {issue}")
                num_invalid += 1
            elif issue.validation == PARAM_VALIDATION.CLOSED:
                self.logger.status(f"Closed: {issue}")
                num_closed += 1
            elif issue.validation == PARAM_VALIDATION.OPEN:
                self.logger.status(f"Open: {issue}")
                num_open += 1
            else:
                raise ValueError(f"Unknown validation type: {issue.validation}")

        if self.mode == "single":
            self.benchmark_runs[0].issues = self.issues

        if num_invalid > 0:
            self.logger.status(f'Benchmark run is INVALID due to {num_invalid} issues ({run_ids})')
            if self.mode == "single":
                self.benchmark_runs[0].category = PARAM_VALIDATION.INVALID
            return PARAM_VALIDATION.INVALID
        elif num_open > 0:
            if self.mode == "single":
                self.benchmark_runs[0].category = PARAM_VALIDATION.OPEN
            self.logger.status(f'Benchmark run qualifies for OPEN category ({run_ids})')
            return PARAM_VALIDATION.OPEN
        else:
            if self.mode == "single":
                self.benchmark_runs[0].category = PARAM_VALIDATION.CLOSED
            self.logger.status(f'Benchmark run qualifies for CLOSED category ({run_ids})')
            return PARAM_VALIDATION.CLOSED
