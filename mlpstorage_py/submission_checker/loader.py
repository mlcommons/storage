
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
    datasize_files: list = None
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
            self.logger.debug("Loading %s log from %s", log_type, path)
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

    def _collect_timestamped_logs(self, command_path):
        """Walk a command directory (.../{datagen,run,datasize}/) and load each
        <datetime>/{summary.json,*_metadata.json} pair.

        Returns ``[(summary_dict, metadata_dict, timestamp_str), ...]`` —
        the tuple shape every downstream rule iterates over
        (`_iter_run_files` / `_iter_datagen_files` in vdb_checks,
        DirectoryCheck.datagen_files_check in directory_checks, etc.).

        Missing ``command_path`` is a STRUCT-12 structural violation handled
        elsewhere; this helper returns an empty list so corpus traversal
        continues.
        """
        out = []
        if not os.path.isdir(command_path):
            return out
        for timestamp in list_dir(command_path):
            timestamp_path = os.path.join(command_path, timestamp)
            summary_path = os.path.join(timestamp_path, "summary.json")
            # BUG-01 (D-E1): refresh per-run metadata; do NOT reuse a
            # previously-looked-up metadata_path across timestamps.
            metadata_path = self.find_metadata_path(timestamp_path)
            metadata_file = self.load_single_log(metadata_path, "Metadata")
            summary_file = self.load_single_log(summary_path, "Summary")
            out.append((summary_file, metadata_file, timestamp))
        return out

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
                                # training/<model>/{datagen,run}/<datetime>/
                                datagen_path = os.path.join(benchmark_path, "datagen")
                                datasize_path = os.path.join(benchmark_path, "datasize")
                                run_path = os.path.join(benchmark_path, "run")
                                datagen_files = []
                                datasize_files = []
                                run_files = []
                                # Missing datagen/ / datasize/ / run/ is a structural violation
                                # caught by SubmissionStructureCheck STRUCT-12 (2.1.12) and by
                                # rule 3.3.1's DATASIZE-MISSING / DATAGEN-MISSING warnings.
                                # The loader yields empty file lists so the rest of the corpus
                                # traversal continues.
                                datagen_timestamps = list_dir(datagen_path) if os.path.isdir(datagen_path) else []
                                datasize_timestamps = list_dir(datasize_path) if os.path.isdir(datasize_path) else []
                                run_timestamps = list_dir(run_path) if os.path.isdir(run_path) else []
                                for timestamp in datagen_timestamps:
                                    timestamp_path = os.path.join(datagen_path, timestamp)
                                    summary_path = os.path.join(timestamp_path, "summary.json")
                                    metadata_path = self.find_metadata_path(timestamp_path)
                                    metadata_file = self.load_single_log(metadata_path, "Metadata")
                                    datagen_file = self.load_single_log(summary_path, "Summary")
                                    datagen_files.append((datagen_file, metadata_file, timestamp))

                                # Issue #608: walk datasize/<ts>/ so rule 3.3.1 can cross-check
                                # run.num_files_train against the value the datasize phase
                                # actually prescribed for this submission. Datasize directories
                                # carry only a metadata file (no summary.json); the summary slot
                                # in each tuple is therefore None.
                                for timestamp in datasize_timestamps:
                                    timestamp_path = os.path.join(datasize_path, timestamp)
                                    metadata_path = self.find_metadata_path(timestamp_path)
                                    metadata_file = self.load_single_log(metadata_path, "Metadata")
                                    datasize_files.append((None, metadata_file, timestamp))

                                for timestamp in run_timestamps:
                                    timestamp_path = os.path.join(run_path, timestamp)
                                    summary_path = os.path.join(timestamp_path, "summary.json")
                                    # BUG-01 (D-E1): refresh per-run metadata; do NOT reuse datagen-loop metadata_path.
                                    metadata_path = self.find_metadata_path(timestamp_path)
                                    metadata_file = self.load_single_log(metadata_path, "Metadata")
                                    run_file = self.load_single_log(summary_path, "Summary")
                                    run_files.append((run_file, metadata_file, timestamp))

                                yield SubmissionLogs(datagen_files=datagen_files, datasize_files=datasize_files, run_files=run_files, system_file=system_file, loader_metadata=loader_metadata)
                            elif mode == "checkpointing":
                                # checkpointing/<model>/<datetime>/   (no <command> segment)
                                checkpoint_path = os.path.join(mode_path, benchmark)
                                checkpoint_files = []
                                checkpoint_timestamps = list_dir(checkpoint_path) if os.path.isdir(checkpoint_path) else []
                                for timestamp in checkpoint_timestamps:
                                    timestamp_path = os.path.join(checkpoint_path, timestamp)
                                    summary_path = os.path.join(timestamp_path, "summary.json")
                                    metadata_path = self.find_metadata_path(timestamp_path)
                                    metadata_file = self.load_single_log(metadata_path, "Metadata")
                                    checkpoint_file = self.load_single_log(summary_path, "Summary")
                                    checkpoint_files.append((checkpoint_file, metadata_file, timestamp))
                                yield SubmissionLogs(checkpoint_files=checkpoint_files, system_file=system_file, loader_metadata=loader_metadata)
                            elif mode == "kv_cache":
                                # kv_cache/<model>/{datagen,run,datasize}/<datetime>/
                                # (issue #612: pre-fix this branch was the shared
                                # `else` and treated <command> dirs as timestamps —
                                # missed the command layer, so metadata.json was
                                # looked for one level too high and never found.)
                                datagen_path = os.path.join(benchmark_path, "datagen")
                                run_path = os.path.join(benchmark_path, "run")
                                datagen_files = self._collect_timestamped_logs(datagen_path)
                                run_files = self._collect_timestamped_logs(run_path)
                                yield SubmissionLogs(
                                    datagen_files=datagen_files,
                                    run_files=run_files,
                                    system_file=system_file,
                                    loader_metadata=loader_metadata,
                                )
                            elif mode == "vector_database":
                                # vector_database/<engine>/<index>/{datagen,run,datasize}/<datetime>/
                                # (issue #612: pre-fix this branch was the shared
                                # `else` and treated <index> dirs as timestamps —
                                # missed both the index AND command layers.)
                                # `benchmark` is the <engine> segment; yield once
                                # per (engine, index) pair so VdbCheck.path lands
                                # on the index dir (vdb_closed_index_types reads
                                # os.path.basename(self.path) and expects the
                                # index token, e.g. DISKANN).
                                if not os.path.isdir(benchmark_path):
                                    continue
                                for index_name in list_dir(benchmark_path):
                                    index_path = os.path.join(benchmark_path, index_name)
                                    if not os.path.isdir(index_path):
                                        continue
                                    datagen_path = os.path.join(index_path, "datagen")
                                    run_path = os.path.join(index_path, "run")
                                    datagen_files = self._collect_timestamped_logs(datagen_path)
                                    run_files = self._collect_timestamped_logs(run_path)
                                    # Per-index loader_metadata: folder points at
                                    # <engine>/<index> so VdbCheck's self.path is
                                    # the index dir (NOT the engine dir).
                                    index_metadata = LoaderMetadata(
                                        division=division, submitter=submitter,
                                        system=system, mode=mode,
                                        benchmark=benchmark, folder=index_path,
                                    )
                                    yield SubmissionLogs(
                                        datagen_files=datagen_files,
                                        run_files=run_files,
                                        system_file=system_file,
                                        loader_metadata=index_metadata,
                                    )
                            else:
                                # Unknown mode — yield an empty SubmissionLogs so
                                # main.py's MODE_TO_CHECKERS.get(mode) is None
                                # branch fires its locked [2.1.10] message with
                                # the right `folder` context attached.
                                yield SubmissionLogs(system_file=system_file, loader_metadata=loader_metadata)

                            
                            



