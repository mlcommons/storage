import csv
import json
import os.path
import pprint

from dataclasses import dataclass
from typing import List, Dict, Any

from mlpstorage.mlps_logging import setup_logging, apply_logging_options
from mlpstorage.config import MLPS_DEBUG, BENCHMARK_TYPES, EXIT_CODE, PARAM_VALIDATION, LLM_MODELS, MODELS
from mlpstorage.rules import get_runs_files, BenchmarkRunVerifier, BenchmarkRun, Issue
from mlpstorage.utils import flatten_nested_dict, remove_nan_values

@dataclass
class Result:
    benchmark_type: BENCHMARK_TYPES
    benchmark_command: str
    benchmark_model: [LLM_MODELS, MODELS]
    benchmark_run: BenchmarkRun
    issues: List[Issue]
    category: PARAM_VALIDATION
    metrics: Dict[str, Any]


class ReportGenerator:

    def __init__(self, results_dir, args=None, logger=None):
        self.args = args
        if self.args is not None:
            self.debug = self.args.debug or MLPS_DEBUG
        else:
            self.debug = MLPS_DEBUG

        if logger:
            self.logger = logger
        else:
            # Ensure there is always a logger available
            self.logger = setup_logging(name=f"mlpstorage_reporter")
            apply_logging_options(self.logger, args)

        self.results_dir = results_dir
        if not os.path.exists(self.results_dir):
            self.logger.error(f'Results directory {self.results_dir} does not exist')
            sys.exit(EXIT_CODE.FILE_NOT_FOUND)

        self.results = []
        self.accumulate_results()
        self.print_results()

    def generate_reports(self):
        # Verify the results directory exists:
        self.logger.info(f'Generating reports for {self.results_dir}')
        results_dicts = [report.benchmark_run.as_dict() for report in self.results]

        self.write_csv_file(results_dicts)
        self.write_json_file(results_dicts)
            
        return EXIT_CODE.SUCCESS

    def accumulate_results(self):
        """
        This function will look through the result_files and generate a result dictionary for each run by reading the metadata.json and summary.json files.

        If the metadata.json file does not exist, log an error and continue
        If summary.json files does not exist, set status=Failed and only use data from metadata.json the run_info from the result_files dictionary
        :return:
        """
        results = []
        benchmark_runs = get_runs_files(self.results_dir, logger=self.logger)

        self.logger.info(f'Accumulating results from {len(benchmark_runs)} runs')
        for benchmark_run in benchmark_runs:
            self.logger.ridiculous(f'Processing run: \n{pprint.pformat(benchmark_run)}')
            verifier = BenchmarkRunVerifier(benchmark_run, logger=self.logger)
            category = verifier.verify()
            issues = verifier.issues
            result_dict = dict(
                benchmark_run=benchmark_run,
                benchmark_type=benchmark_run.benchmark_type,
                benchmark_command=benchmark_run.command,
                benchmark_model=benchmark_run.model,
                issues=issues,
                category=category,
                metrics=benchmark_run.metrics
            )
            self.results.append(Result(**result_dict))

    def print_results(self):
        print("\n========================= Results Report =========================")
        for category in [PARAM_VALIDATION.CLOSED, PARAM_VALIDATION.OPEN, PARAM_VALIDATION.INVALID]:
            print(f"\n------------------------- {category.value.upper()} Report -------------------------")
            for result in self.results:
                if result.category == category:
                    print(f'\tRunID: {result.benchmark_run.run_id}')
                    print(f'\t    Benchmark Type: {result.benchmark_type.value}')
                    print(f'\t    Command: {result.benchmark_command}')
                    print(f'\t    Model: {result.benchmark_model}')
                    if result.issues:
                        print(f'\t    Issues:')
                        for issue in result.issues:
                            print(f'\t\t- {issue}')
                    else:
                        print(f'\t\t- No issues found')

                    if result.metrics:
                        print(f'\t    Metrics:')
                        for metric, value in result.metrics.items():
                            if type(value) in (int, float):
                                if "percentage" in metric.lower():
                                    print(f'\t\t- {metric}: {value:,.1f}%')
                                else:
                                    print(f'\t\t- {metric}: {value:,.1f}')
                            elif type(value) in (list, tuple):
                                if "percentage" in metric.lower():
                                    print(f'\t\t- {metric}: {", ".join(f"{v:,.1f}%" for v in value)}')
                                else:
                                    print(f'\t\t- {metric}: {", ".join(f"{v:,.1f}" for v in value)}')
                            else:
                                print(f'\t\t- {metric}: {value}')

                    print("\n")

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
