import csv
import json

from typing import List, Dict, Any

from mlpstorage.config import MLPS_DEBUG
from mlpstorage.logging import setup_logging, apply_logging_options
from mlpstorage.rules import get_runs_files
from mlpstorage.utils import flatten_nested_dict, remove_nan_values

class ReportGenerator:

    def __init__(self, result_dir, args=None, logger=None):
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

        self.result_dir = result_dir
        self.result_files = get_runs_files(self.result_dir)
        self.results = self.accumulate_results()
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
        for run_info in self.result_files:
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

            combined_result_dict = dict(
                run_info=run_info,
                mlps=metadata,
                dlio=summary,
                status=status
            )
            results.append(combined_result_dict)

        return results

    def write_json_file(self):
        with open(f'{self.result_dir}/results.json', 'w') as f:
            json.dump(self.results, f, indent=2)

    def write_csv_file(self):
        flattened_results = [flatten_nested_dict(r) for r in self.results]
        flattened_results = [remove_nan_values(r) for r in flattened_results]
        fieldnames = set()
        for l in flattened_results:
            fieldnames.update(l.keys())

        with open(f'{self.result_dir}/results.csv', 'w+', newline='') as file_object:
            csv_writer = csv.DictWriter(f=file_object, fieldnames=sorted(fieldnames), lineterminator='\n')
            csv_writer.writeheader()
            csv_writer.writerows(flattened_results)
