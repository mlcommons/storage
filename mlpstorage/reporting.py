import csv
import json
import os.path
import pprint

from typing import List, Dict, Any

from mlpstorage.mlps_logging import setup_logging, apply_logging_options
from mlpstorage.config import MLPS_DEBUG, BENCHMARK_TYPES
from mlpstorage.rules import get_runs_files, BenchmarkRunVerifier, BenchmarkRun
from mlpstorage.utils import flatten_nested_dict, remove_nan_values

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
        self.benchmark_runs = get_runs_files(self.results_dir, logger=self.logger)
        self.logger.info(f'Found {len(self.benchmark_runs)} runs')
        self.results = []

        self.verify_results()

    def verify_results(self):
        for benchmark_run in self.benchmark_runs:
            verifier = BenchmarkRunVerifier(benchmark_run, logger=self.logger)
            verifier.verify()

    def generate_reports(self, write_files=True):
        self.logger.info(f'Generating reports for {self.results_dir}')
        self.results = self.accumulate_results()

        if write_files:
            self.write_csv_file()
            self.write_json_file()

    def accumulate_results(self):
        """
        This function will look through the result_files and generate a result dictionary for each run by reading the metadata.json and summary.json files.

        If the metadata.json file does not exist, log an error and continue
        If summary.json files does not exist, set status=Failed and only use data from metadata.json the run_info from the result_files dictionary
        :return:
        """
        results = []
        self.logger.info(f'Accumulating results from {len(self.benchmark_runs)} runs')
        for benchmark_run in self.benchmark_runs:
            self.logger.ridiculous(f'Processing run: \n{pprint.pformat(benchmark_run)}')
            results.append(benchmark_run.as_dict())

        return results

    def write_json_file(self):
        json_file = os.path.join(self.results_dir,'results.json')
        self.logger.info(f'Writing results to {json_file}')
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def write_csv_file(self):
        csv_file = os.path.join(self.results_dir,'results.csv')
        self.logger.info(f'Writing results to {csv_file}')
        flattened_results = [flatten_nested_dict(r) for r in self.results]
        flattened_results = [remove_nan_values(r) for r in flattened_results]
        fieldnames = set()
        for l in flattened_results:
            fieldnames.update(l.keys())

        with open(csv_file, 'w+', newline='') as file_object:
            csv_writer = csv.DictWriter(f=file_object, fieldnames=sorted(fieldnames), lineterminator='\n')
            csv_writer.writeheader()
            csv_writer.writerows(flattened_results)
