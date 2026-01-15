"""
Report generation for MLPerf Storage benchmark results.

This module provides the ReportGenerator class for validating and
reporting on benchmark results with clear OPEN vs CLOSED messaging.
"""

import csv
import json
import os.path
import pprint
import sys

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union

from mlpstorage.mlps_logging import setup_logging, apply_logging_options
from mlpstorage.config import MLPS_DEBUG, BENCHMARK_TYPES, EXIT_CODE, PARAM_VALIDATION, LLM_MODELS, MODELS, ACCELERATORS
from mlpstorage.rules import get_runs_files, BenchmarkVerifier, BenchmarkRun, Issue
from mlpstorage.utils import flatten_nested_dict, remove_nan_values
from mlpstorage.reporting import (
    ResultsDirectoryValidator,
    ValidationMessageFormatter,
    ClosedRequirementsFormatter,
    ReportSummaryFormatter,
)


@dataclass
class Result:
    """Container for a single benchmark run result."""
    multi: bool
    benchmark_type: BENCHMARK_TYPES
    benchmark_command: str
    benchmark_model: Union[LLM_MODELS, MODELS, str]
    benchmark_run: Union[BenchmarkRun, List[BenchmarkRun]]
    issues: List[Issue]
    category: PARAM_VALIDATION
    metrics: Dict[str, Any]


class ReportGenerator:
    """
    Generate validation reports for benchmark results.

    This class provides:
    - Directory structure validation before processing
    - Clear OPEN vs CLOSED submission messaging
    - Error isolation for individual runs
    - Summary reports by category

    Args:
        results_dir: Path to the results directory.
        args: Optional argparse namespace with configuration.
        logger: Optional logger instance.
        validate_structure: Whether to validate directory structure (default True).
        use_colors: Whether to use terminal colors in output (default True).
    """

    def __init__(self, results_dir: str, args=None, logger=None,
                 validate_structure: bool = True, use_colors: bool = True):
        self.args = args
        if self.args is not None:
            self.debug = self.args.debug or MLPS_DEBUG
        else:
            self.debug = MLPS_DEBUG

        if logger:
            self.logger = logger
        else:
            # Ensure there is always a logger available
            self.logger = setup_logging(name="mlpstorage_reporter")
            apply_logging_options(self.logger, args)

        self.results_dir = results_dir

        # Initialize formatters
        self.msg_formatter = ValidationMessageFormatter(use_colors=use_colors)
        self.summary_formatter = ReportSummaryFormatter(use_colors=use_colors)

        # Validate directory structure first if requested
        if validate_structure:
            if not self._validate_directory_structure():
                sys.exit(EXIT_CODE.FILE_NOT_FOUND)

        self.run_results: Dict[str, Result] = {}
        self.workload_results: Dict[tuple, Result] = {}
        self.processing_errors: List[str] = []

        self.accumulate_results()
        self.print_results()

    def _validate_directory_structure(self) -> bool:
        """
        Validate the results directory structure before processing.

        Returns:
            True if structure is valid, False otherwise.
        """
        validator = ResultsDirectoryValidator(self.results_dir, logger=self.logger)
        result = validator.validate()

        if not result.is_valid:
            self.logger.error("Results directory structure validation failed:")
            self.logger.error(validator.get_error_report())
            self.logger.error("")
            self.logger.error("Expected structure:")
            self.logger.error(validator.get_expected_structure_help())
            return False

        # Log warnings if any
        if result.warnings:
            for warning in result.warnings:
                self.logger.warning(warning)

        self.logger.info(
            f"Directory validation passed: found {result.found_runs} runs "
            f"in {len(result.found_benchmark_types)} benchmark types"
        )
        return True

    def generate_reports(self):
        # Verify the results directory exists:
        self.logger.info(f'Generating reports for {self.results_dir}')
        run_result_dicts = [report.benchmark_run.as_dict() for report in self.run_results.values()]

        self.write_csv_file(run_result_dicts)
        self.write_json_file(run_result_dicts)
            
        return EXIT_CODE.SUCCESS

    def accumulate_results(self) -> None:
        """
        Accumulate and validate results from all benchmark runs.

        This method:
        1. Scans the results directory for benchmark runs
        2. Validates each run individually (with error isolation)
        3. Groups runs by workload for submission validation
        4. Runs multi-run verifiers on workload groups

        Errors in individual runs are logged but do not stop processing.
        """
        try:
            benchmark_runs = get_runs_files(self.results_dir, logger=self.logger)
        except Exception as e:
            self.logger.error(f"Failed to scan results directory: {e}")
            self.processing_errors.append(f"Directory scan failed: {e}")
            return

        if not benchmark_runs:
            self.logger.warning(
                f"No valid benchmark runs found in {self.results_dir}. "
                "Ensure runs have completed and contain metadata files."
            )
            return

        self.logger.info(f'Accumulating results from {len(benchmark_runs)} runs')

        # Process individual runs with error isolation
        for benchmark_run in benchmark_runs:
            try:
                self._process_single_run(benchmark_run)
            except Exception as e:
                error_msg = f"Failed to process run {benchmark_run.run_id}: {e}"
                self.logger.error(f"{error_msg}. Skipping.")
                self.processing_errors.append(error_msg)
                continue

        # Group runs for workload-level validation
        self._process_workload_groups(benchmark_runs)

    def _process_single_run(self, benchmark_run: BenchmarkRun) -> None:
        """
        Process and validate a single benchmark run.

        Args:
            benchmark_run: The benchmark run to process.

        Raises:
            Exception: If processing fails critically.
        """
        self.logger.ridiculous(f'Processing run: \n{pprint.pformat(benchmark_run)}')

        verifier = BenchmarkVerifier(benchmark_run, logger=self.logger)
        category = verifier.verify()
        issues = verifier.issues

        result = Result(
            multi=False,
            benchmark_run=benchmark_run,
            benchmark_type=benchmark_run.benchmark_type,
            benchmark_command=benchmark_run.command,
            benchmark_model=benchmark_run.model,
            issues=issues,
            category=category,
            metrics=benchmark_run.metrics or {}
        )
        self.run_results[benchmark_run.run_id] = result

        # Log category for the run
        self.logger.debug(
            f"Run {benchmark_run.run_id} validated as {category.value.upper()}"
        )

    def _process_workload_groups(self, benchmark_runs: List[BenchmarkRun]) -> None:
        """
        Group runs by workload and run submission-level validation.

        Args:
            benchmark_runs: List of all benchmark runs.
        """
        # Group by (model, accelerator) for training, (model,) for others
        workload_runs: Dict[tuple, List[BenchmarkRun]] = {}

        for benchmark_run in benchmark_runs:
            workload_key = (benchmark_run.model, benchmark_run.accelerator)
            if workload_key not in workload_runs:
                workload_runs[workload_key] = []
            workload_runs[workload_key].append(benchmark_run)

        # Run workload-level verifiers
        for workload_key, runs in workload_runs.items():
            model, accelerator = workload_key
            if not runs:
                continue

            try:
                self.logger.info(
                    f'Running submission verifiers for model: {model}, '
                    f'accelerator: {accelerator} ({len(runs)} runs)'
                )
                verifier = BenchmarkVerifier(*runs, logger=self.logger)
                category = verifier.verify()
                issues = verifier.issues

                result = Result(
                    multi=True,
                    benchmark_run=runs,
                    benchmark_type=runs[0].benchmark_type,
                    benchmark_command=runs[0].command,
                    benchmark_model=runs[0].model,
                    issues=issues,
                    category=category,
                    metrics={}  # TODO: Add function to aggregate metrics
                )
                self.workload_results[workload_key] = result

            except Exception as e:
                error_msg = f"Failed to validate workload {workload_key}: {e}"
                self.logger.error(f"{error_msg}. Skipping workload.")
                self.processing_errors.append(error_msg)

    def print_results(self) -> None:
        """
        Print results with clear OPEN/CLOSED distinction.

        Results are organized by category with INVALID runs first (most critical),
        followed by OPEN runs, then CLOSED runs.
        """
        if not self.run_results and not self.workload_results:
            print("\nNo results to display.")
            if self.processing_errors:
                print("\nProcessing errors occurred:")
                for error in self.processing_errors:
                    print(f"  - {error}")
            return

        # Calculate summary counts
        closed_count = sum(1 for r in self.run_results.values()
                          if r.category == PARAM_VALIDATION.CLOSED)
        open_count = sum(1 for r in self.run_results.values()
                        if r.category == PARAM_VALIDATION.OPEN)
        invalid_count = sum(1 for r in self.run_results.values()
                           if r.category == PARAM_VALIDATION.INVALID)

        # Print summary header
        print(self.summary_formatter.format_summary_header(
            len(self.run_results), closed_count, open_count, invalid_count
        ))

        # Print INVALID runs first (most important to address)
        if invalid_count > 0:
            print(self.summary_formatter.format_section_header(
                PARAM_VALIDATION.INVALID, invalid_count
            ))
            for result in self.run_results.values():
                if result.category == PARAM_VALIDATION.INVALID:
                    self._print_run_details(result)

        # Print OPEN runs
        if open_count > 0:
            print(self.summary_formatter.format_section_header(
                PARAM_VALIDATION.OPEN, open_count
            ))
            for result in self.run_results.values():
                if result.category == PARAM_VALIDATION.OPEN:
                    self._print_run_details(result)

        # Print CLOSED runs
        if closed_count > 0:
            print(self.summary_formatter.format_section_header(
                PARAM_VALIDATION.CLOSED, closed_count
            ))
            for result in self.run_results.values():
                if result.category == PARAM_VALIDATION.CLOSED:
                    self._print_run_details(result)

        # Print submission-level results
        self._print_submission_results()

        # Print any processing errors at the end
        if self.processing_errors:
            print("\n" + "-" * 70)
            print("PROCESSING ERRORS")
            print("-" * 70)
            for error in self.processing_errors:
                print(f"  - {error}")

    def _print_run_details(self, result: Result) -> None:
        """
        Print details for a single run result.

        Args:
            result: The Result object to print.
        """
        # Print header with badge
        run_id = result.benchmark_run.run_id
        print(self.msg_formatter.format_run_header(
            run_id=run_id,
            category=result.category,
            benchmark_type=result.benchmark_type.value if result.benchmark_type else "unknown",
            model=str(result.benchmark_model),
            command=result.benchmark_command
        ))

        # Print issues (only non-CLOSED for brevity)
        print(self.msg_formatter.format_issues_list(result.issues, show_all=False))

        # Print metrics
        print(self.msg_formatter.format_metrics(result.metrics))
        print()

    def _print_submission_results(self) -> None:
        """Print submission-level (workload group) results."""
        if not self.workload_results:
            return

        print("\n" + "=" * 70)
        print("SUBMISSION VALIDATION REPORT")
        print("=" * 70)

        # Group by category
        for category in [PARAM_VALIDATION.INVALID, PARAM_VALIDATION.OPEN, PARAM_VALIDATION.CLOSED]:
            category_results = [
                (k, v) for k, v in self.workload_results.items()
                if v.category == category
            ]

            if not category_results:
                continue

            badge = self.msg_formatter.format_category_badge(category)
            print(f"\n{badge} Submissions ({len(category_results)})")
            print("-" * 40)

            for workload_key, workload_result in category_results:
                self._print_workload_details(workload_key, workload_result)

    def _print_workload_details(self, workload_key: tuple, workload_result: Result) -> None:
        """
        Print details for a workload submission.

        Args:
            workload_key: Tuple of (model, accelerator).
            workload_result: The Result object for the workload.
        """
        model, accelerator = workload_key

        # Determine workload type
        if workload_result.benchmark_model in LLM_MODELS:
            workload_id = f"Checkpointing - {workload_result.benchmark_model}"
        elif workload_result.benchmark_model in MODELS:
            workload_id = f"Training - {workload_result.benchmark_model}, Accelerator: {accelerator}"
        else:
            workload_id = f"{workload_result.benchmark_type.value} - {workload_result.benchmark_model}"

        badge = self.msg_formatter.format_category_badge(workload_result.category)
        print(f"\n{badge} {workload_id}")
        print(f"    Benchmark Type: {workload_result.benchmark_type.value}")

        if workload_result.benchmark_command:
            print(f"    Command: {workload_result.benchmark_command}")

        # Print run summary
        print("    Runs:")
        for run in workload_result.benchmark_run:
            run_category = self.run_results[run.run_id].category
            run_badge = self.msg_formatter.format_category_badge(run_category)
            print(f"      - {run.run_id} {run_badge}")

        # Print submission-level issues
        print(self.msg_formatter.format_issues_list(workload_result.issues, show_all=False))

        # Print requirements checklist for non-CLOSED
        if workload_result.category != PARAM_VALIDATION.CLOSED:
            benchmark_type = workload_result.benchmark_type.value
            checklist = ClosedRequirementsFormatter.format_checklist(benchmark_type)
            if checklist:
                print(f"\n    {checklist}")


    def write_json_file(self, results):
        json_file = os.path.join(self.results_dir,'results.json')
        self.logger.info(f'Writing results to {json_file}')
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)

    def write_csv_file(self, results):
        csv_file = os.path.join(self.results_dir,'results.csv')
        self.logger.info(f'Writing results to {csv_file}')
        flattened_results = [flatten_nested_dict(r) for r in results]
        flattened_results = [remove_nan_values(r) for r in flattened_results]
        fieldnames = set()
        for l in flattened_results:
            fieldnames.update(l.keys())

        with open(csv_file, 'w+', newline='') as file_object:
            csv_writer = csv.DictWriter(f=file_object, fieldnames=sorted(fieldnames), lineterminator='\n')
            csv_writer.writeheader()
            csv_writer.writerows(flattened_results)
