
import os
from typing import Generator, Literal
from .utils import *
from .constants import *
import logging
from dataclasses import dataclass
from .parsers.json_parser import JSONParser
from .configuration.configuration import Config

@dataclass
class LoaderMetadata:
    division: str = None
    submitter: str = None
    system: str = None
    mode: str = None
    benchmark: str = None
    folder: str = None

@dataclass
class SubmissionLogs:
    """Container for parsed submission log artifacts and metadata.

    The `SubmissionLogs` class holds references to parsed log files and
    associated metadata for a single submission. It serves as a data
    transfer object passed between loading and validation phases.
    """
    datagen_files: list = None
    run_files: list = None
    checkpoint_files: list = None
    system_file: dict = None
    loader_metadata: LoaderMetadata = None


class Loader:
    """Loads and parses submission artifacts from the filesystem.

    The `Loader` class traverses the submission directory structure,
    identifies valid submissions, and parses their log files and metadata.
    It yields `SubmissionLogs` objects for each valid submission found,
    handling version-specific path formats and optional artifacts.
    """
    def __init__(self, root, version, config: Config) -> None:
        """Initialize the submission loader.

        Sets up path templates based on the MLPerf version and root
        directory.

        Args:
            root (str): Root directory containing submissions.
            version (str): MLPerf version for path resolution.
        """
        self.root = root
        self.version = version
        self.logger = logging.getLogger("Loader")
        self.system_log_path = os.path.join(
            self.root, SYSTEM_PATH.get(
                version, SYSTEM_PATH["default"]))
        self.parser_map = PARSER_MAP
        self.config = config

    def load_single_log(self, path, log_type):
        log = None
        if os.path.exists(path):
            self.logger.info("Loading %s log from %s", log_type, path)
            log = self.parser_map.get(log_type, self.parser_map["default"])(path, log_type).get_dict()
        else:
            self.logger.warning(
                "Could not load %s log from %s, path does not exists",
                log_type,
                path)
        return log
    
    def find_metadata_path(self, path):
        files = [f for f in list_files(path) if "metadata" in f]
        if len(files) == 0:
            self.logger.warning("Could not find metadata file at %s", path)
            return os.path.join(path, "metadata.json")
        elif len(files) > 1:
            self.logger.warning("More than one metadata file found at %s", path)
        return os.path.join(path, files[0])

    def load(self) -> Generator[SubmissionLogs, None, None]:
        # Iterate over submission folder.
        # Division -> submitter -> system -> benchmark -> runs
        for division in list_dir(self.root):
            if division not in VALID_DIVISIONS:
                continue
            division_path = os.path.join(self.root, division)
            for submitter in list_dir(division_path):
                if not self.config.check_submitter(submitter):
                    continue
                results_path = os.path.join(
                    division_path, submitter, "results")
                for system in list_dir(results_path):
                    system_path = os.path.join(results_path, system)
                    system_file_path = self.system_log_path.format(division = division, submitter = submitter, system = system)
                    system_file = self.load_single_log(system_file_path, "System")
                    for mode in list_dir(system_path):
                        mode_path = os.path.join(system_path, mode)
                        for benchmark in list_dir(mode_path):
                            benchmark_path = os.path.join(mode_path, benchmark)
                            loader_metadata = LoaderMetadata(division=division, submitter=submitter, system=system, mode=mode, benchmark=benchmark, folder=benchmark_path)
                            if mode == "training":
                                datagen_path = os.path.join(benchmark_path, "datagen")
                                run_path = os.path.join(benchmark_path, "run")
                                datagen_files = []
                                run_files = []
                                for timestamp in list_dir(datagen_path):
                                    timestamp_path = os.path.join(datagen_path, timestamp)
                                    summary_path = os.path.join(timestamp_path, "summary.json")
                                    metadata_path = self.find_metadata_path(timestamp_path)
                                    metadata_file = self.load_single_log(metadata_path, "Metadata")
                                    datagen_file = self.load_single_log(summary_path, "Summary")
                                    datagen_files.append((datagen_file, metadata_file, timestamp))

                                for timestamp in list_dir(run_path):
                                    timestamp_path = os.path.join(run_path, timestamp)
                                    summary_path = os.path.join(timestamp_path, "summary.json")
                                    metadata_file = self.load_single_log(metadata_path, "Metadata")
                                    run_file = self.load_single_log(summary_path, "Summary")
                                    run_files.append((run_file, metadata_file, timestamp))
                                
                                yield SubmissionLogs(datagen_files, run_files, system_file=system_file, loader_metadata=loader_metadata)
                            else:
                                checkpoint_path = os.path.join(mode_path, benchmark)
                                checkpoint_files = []
                                for timestamp in list_dir(checkpoint_path):
                                    timestamp_path = os.path.join(checkpoint_path, timestamp)
                                    summary_path = os.path.join(timestamp_path, "summary.json")
                                    metadata_path = self.find_metadata_path(timestamp_path)
                                    metadata_file = self.load_single_log(metadata_path, "Metadata")
                                    checkpoint_file = self.load_single_log(summary_path, "Summary")
                                    checkpoint_files.append((checkpoint_file, metadata_file, timestamp))
                                yield SubmissionLogs(checkpoint_files=checkpoint_files, system_file=system_file, loader_metadata=loader_metadata)

                            
                            



