import csv
import json
import os.path
import pprint

from dataclasses import dataclass
from typing import List, Dict, Any

from mlpstorage.mlps_logging import setup_logging, apply_logging_options
from mlpstorage.config import MLPS_DEBUG, BENCHMARK_TYPES
from mlpstorage.reporting import ReportGenerator
from mlpstorage.rules import get_runs_files
from mlpstorage.utils import flatten_nested_dict, remove_nan_values


@dataclass
class ProcessedRun:
    run_id: str
    benchmark_type: str
    run_parameters: Dict[str, Any]
    run_metrics: Dict[str, Any]
    issues: List[str]


class SubmissionChecker:

    def __init__(self, results_dir, args, logger):
        self.results_dir = results_dir
        self.args = args
        self.logger = logger

        report_generator = ReportGenerator(results_dir, args, logger)
        report_generator.generate_reports(write_files=False)
        self.raw_results = report_generator.results
        self.invalid_results = []
        self.valid_results = []
        self.validate_results_for_submission()

        self.checks = {getattr(self, obj) for obj in dir(self) if callable(getattr(self, obj)) and obj.startswith('check_')}
        self.logger.info(f'Found {len(self.checks)} checks to run')
        self.logger.info(f'Running checks {[obj.__name__ for obj in self.checks]}')

        self.processed_results = list()  # Items in this list will ProcessedRun objects

        self.submission_issues = dict()
        self.checks_run = []

        self.run_checks()
        self.generate_summary_tables()

    def run_checks(self):
        for result in self.valid_results:
            result_issues = []
            for check in self.checks:
                result_issues = check(result)

            if result_issues:
                self.submission_issues[result['run_id']] = result_issues
            else:
                self.processed_results.append(
                    ProcessedRun(
                        run_id=result['run_id'],
                        benchmark_type=result['benchmark_type'],
                        run_parameters=result['run_info']['run_parameters'],
                        run_metrics=result['dlio']['summary'],
                        issues=result_issues
                ))

    def check_required_files(self, result):
        print('Checking required files...')
        # If the required files exist then we return None
        # If there is an issue we retrurn an issue string
        return None


    def validate_results_for_submission(self):
        """
        We need to walk through each result and verify we have data for the rest of the checks
        """
        valid_results = []
        for result in self.raw_results:
            if not result.get("mlps"):
                self.logger.error(f'Validating a run without a metadata.json file. Please contact the developer. run information: {result}')
                self.invalid_results.append(result)
                continue

            mlps_metadata_file = result['run_info']['mlps_metadata_file']
            if not result.get("dlio"):
                self.logger.verbose(f'INVALID - No dlio summary.json information found. Metadata file: {mlps_metadata_file}')
                self.invalid_results.append(result)
                continue

            valid_results.append(result)

        self.valid_results = valid_results

    def generate_summary_tables(self):
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate reports from MLPerf Storage')
    parser.add_argument('--results-dir', type=str, required=True, help='Path to the results directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()
    logger = setup_logging(name=f"mlpstorage_reporter")
    apply_logging_options(logger, args)

    checker = SubmissionChecker(results_dir=args.results_dir, args=args, logger=logger)