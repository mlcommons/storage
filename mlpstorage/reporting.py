import csv
import json
import mlps_logging
import os.path
import pprint

from typing import List, Dict, Any

from mlpstorage.mlps_logging import setup_logging, apply_logging_options
from mlpstorage.config import MLPS_DEBUG, BENCHMARK_TYPES
from mlpstorage.rules import get_runs_files
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
        self.result_files = []
        self.results = []

    def generate_reports(self, write_files=True):
        self.logger.info(f'Generating reports for {self.results_dir}')
        self.result_files = get_runs_files(self.results_dir, logger=self.logger)
        self.logger.info(f'Found {len(self.result_files)} runs')
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
        self.logger.info(f'Accumulating results from {len(self.result_files)} runs')
        for run_info in self.result_files:
            self.logger.ridiculous(f'Processing run: \n{pprint.pformat(run_info)}')

            if not run_info.get("mlps_metadata_file"):
                self.logger.error(f"No metadata.json file found in {run_info['run_dir']}")
                continue

            with open(run_info["mlps_metadata_file"], "r") as f:
                try:
                    metadata = json.load(f)
                except Exception as e:
                    self.logger.error(f"Error loading metadata.json from {run_info['mlps_metadata_file']}: {e}")
                    continue

            if run_info.get("dlio_summary_json_file"):
                with open(run_info["dlio_summary_json_file"], "r") as f:
                    try:
                        summary = json.load(f)
                    except Exception as e:
                        self.logger.error(f"Error loading summary.json from {run_info['dlio_summary_json_file']}: {e}")
                        continue
                status = "Completed"
            else:
                summary = dict()
                status = "Failed"

            run_id = metadata['args']['program']
            if command := metadata['args'].get("command"):
                run_id += f"_{command}"
            if subcommand := metadata['args'].get("subcommand"):
                run_id += f"_{subcommand}"
            if model := metadata['args'].get("model"):
                run_id += f"_{model}"
            run_id += f"_{metadata['run_datetime']}"

            combined_result_dict = dict(
                run_id=run_id,
                run_info=run_info,
                mlps=metadata,
                dlio=summary,
                status=status
            )
            results.append(combined_result_dict)

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

    def validate_results_for_submission(self, print=True):
        """
        We need to walk through each result and verify the following:
            1. Ensure every run has the metadata file
            2. Ensure every run has a summary.json file
            3. Ensure the parameters for a run abide by the rules
            4. Ensure status if "Completed"

        Then we print a set of helpful tables showing the results for the various benchmarks
        """
        training_tests = []
        checkpointing_tests = []
        invalid_reasons = []  # List of dictionaries iwth keys "run_id"
        for result in self.results:
            if not result.get("mlps"):
                self.logger.error(f'Validating a run without a metadata.json file. Please contact the developer. run information: {result}')
                continue

            mlps_metadata_file = result['run_info']['mlps_metadata_file']
            if not result.get("dlio"):
                self.logger.verbose(f'INVALID - No dlio summary.json information found. Metadata file: {mlps_metadata_file}')
                invalid_reasons.append(dict(mlps_metadata_file=mlps_metadata_file,
                                            result_information=result,
                                            reason='No dlio summary.json information found.'))

            program = result['mlps']['args']['program']
            if program in (BENCHMARK_TYPES.training, BENCHMARK_TYPES.checkpointing):
                # Training and Checkpointing use DLIO
                runtime_params = result['mlps']['combined_params']


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate reports from MLPerf Storage')
    parser.add_argument('--results-dir', type=str, required=True, help='Path to the results directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()
    logger = setup_logging(name=f"mlpstorage_reporter")
    apply_logging_options(logger, args)

    reporter = ReportGenerator(results_dir=args.results_dir, args=args, logger=logger)
    reporter.generate_reports(write_files=False)
    reporter.validate_results_for_submission(print=True)
