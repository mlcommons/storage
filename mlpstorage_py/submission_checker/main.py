import argparse
import logging
import os
import sys

# Constants
from .constants import *

# Import config
from .configuration.configuration import Config

# Import loader
from .loader import Loader

# Import checkers
from .checks.checkpointing_checks import CheckpointingCheck
from .checks.directory_checks import DirectoryCheck
from .checks.training_checks import TrainingCheck


# Import result exporter
from .results import ResultExporter

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s"
)
log = logging.getLogger("main")

def get_args():
    """Parse command-line arguments for the submission checker.

    Sets up an ArgumentParser with options for input directory, version,
    filtering, output files, and various skip flags for different checks.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="submission directory")
    parser.add_argument("--submitters", help="Comma separated submitters to run the checker")
    parser.add_argument(
        "--version",
        default="v5.1",
        choices=list(VERSIONS),
        help="mlperf version",
    )
    parser.add_argument(
        "--csv",
        default="summary.csv",
        help="csv file with results")
    parser.add_argument(
        "--skip-output-file",
        action="store_true",
        help="Skip check output file"
    )
    args = parser.parse_args()
    return args

def main():
    """Run the MLPerf submission checker on the provided directory.

    Parses arguments, initializes configuration and loader, iterates
    through all submissions, runs validation checks (performance,
    accuracy, system, measurements, power), collects results, and
    exports summaries. Logs pass/fail status and statistics.

    Returns:
        int: 0 if all submissions pass checks, 1 if any errors found.
    """
    args = get_args()
    
    submitters = str(args.submitters).split(",")
    config = Config(
        version=args.version,
        submitters=submitters,
        skip_output_file=args.skip_output_file
    )
    
    loader = Loader(args.input, args.version, config)
    exporter = ResultExporter(args.csv, config)


    results = {}
    systems = {}
    errors = []
    checkers = [DirectoryCheck, TrainingCheck, CheckpointingCheck]
    # Main loop over all the submissions
    for logs in loader.load():
        # TODO: Initialize checkers
        checkers_pipe = []
        valid = True
        #TODO: Run checks
        for checker in checkers:
            valid &= checker(log, config, logs)()

        # TODO: Add results to summary
        if valid:
            exporter.add_result(logs)
        else:
            errors.append(logs.loader_metadata.folder)
    
    # Export results
    exporter.export()

    if len(errors) > 0:
        log.error("SUMMARY: submission has errors")
        return 1
    else:
        log.info("SUMMARY: submission looks OK")
        return 0

if __name__ == "__main__":
    sys.exit(main())