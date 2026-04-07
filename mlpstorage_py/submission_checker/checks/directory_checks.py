
from .base import BaseCheck
from ..constants import *
from ..configuration.configuration import Config
from ..loader import SubmissionLogs
from ..utils import *

import os
import re
from datetime import datetime


class DirectoryCheck(BaseCheck):
    """
    A check class for validating directory structure and related properties.
    Inherits from BaseCheck and receives a config and loader instance.
    """

    def __init__(self, log, config: Config, submissions_logs: SubmissionLogs):
        """
        Initialize DirectoryChecks with configuration and loader.

        Args:
            config: A Config instance containing submission configuration.
            loader: A SubmissionLogs instance for accessing submission logs.
        """
        # Call parent constructor with the loader's log and submission path
        super().__init__(log=log, path=submissions_logs.loader_metadata.folder)
        self.config = config
        self.submissions_logs = submissions_logs
        self.name = "directory checks"
        self.datagen_path = os.path.join(self.path, "datagen")
        self.run_path = os.path.join(self.path, "run")
        self.checkpointing_path = self.path
        self.init_checks()

    def init_checks(self):
        self.checks = []
        mode = getattr(self.submissions_logs.loader_metadata, 'mode', 'training')
        if mode == "training":
            # Training mode checks
            self.checks.extend([
                self.datagen_files_check,
                self.datagen_dlio_config_check,
                self.run_results_json_check,
                self.run_files_check,
                self.run_files_timestamp_check,
                self.run_dlio_config_check,
                self.run_duration_valid_check,
            ])
        else:
            # Checkpointing mode checks
            self.checks.extend([
                self.checkpointing_results_json_check,
                self.checkpointing_timestamps_check,
                self.checkpointing_timestamp_gap_check,
                self.checkpointing_files_check,
                self.checkpointing_dlio_config_check,
            ])
    
    
    def datagen_files_check(self):
        """
        Check that each datagen timestamp directory contains:
        - training_datagen.stdout.log
        - training_datagen.stderr.log
        - *output.json
        - *per_epoch_stats.json
        - *summary.json
        - dlio.log
        - dlio_config/ (subdirectory)
        """
        valid = True
        for _, _, timestamp in self.submissions_logs.datagen_files:
            timestamp_path = os.path.join(self.datagen_path, timestamp)
            files = list_files(timestamp_path)
            for required_file in self.config.get_datagen_required_files():
                if self.config.skip_output_file and required_file == "*output.json":
                    continue
                if not regex_matches_any(required_file, files):
                    self.log.error("%s not found in %s", required_file, timestamp_path)
                    valid = False
            
            # Check for dlio_config directory
            for required_folder in self.config.get_datagen_required_folders():
                if required_folder not in list_dir(timestamp_path):
                    self.log.error("%s directory not found in %s", required_folder, timestamp_path)
                    valid = False
        
        return valid
    
    def datagen_dlio_config_check(self):
        """
        Check that the dlio_config subdirectory in each datagen timestamp directory
        contains exactly: config.yaml, hydra.yaml, and overrides.yaml (case-sensitive).
        """
        valid = True
        required_files = {"config.yaml", "hydra.yaml", "overrides.yaml"}
        
        for _, _, timestamp in self.submissions_logs.datagen_files:
            dlio_config_path = os.path.join(self.datagen_path, timestamp, "dlio_config")
            
            if not os.path.exists(dlio_config_path):
                self.log.error("dlio_config directory not found in %s", dlio_config_path)
                valid = False
                continue
            
            files = set(list_files(dlio_config_path))
            
            # Check for exact match
            if files != required_files:
                self.log.error(
                    "dlio_config in %s has incorrect files. Expected %s, got %s",
                    dlio_config_path,
                    required_files,
                    files
                )
                valid = False
        
        return valid

    def run_results_json_check(self):
        """
        Check that there is exactly one results.json file in the run phase directory.
        """
        valid = True
        results_files = list_files(self.run_path)
        results_json_count = sum(1 for f in results_files if f == "results.json")
        
        if results_json_count != 1:
            self.log.error(
                "Expected exactly 1 results.json file in %s, found %d",
                self.run_path,
                results_json_count
            )
            valid = False
        
        return valid
    
    def run_files_check(self):
        """
        Check that each run timestamp directory contains:
        - training_run.stdout.log
        - training_run.stderr.log
        - *output.json
        - *per_epoch_stats.json
        - *summary.json
        - dlio.log
        - dlio_config/ (subdirectory)
        """
        valid = True
        for _, _, timestamp in self.submissions_logs.run_files:
            timestamp_path = os.path.join(self.run_path, timestamp)
            files = list_files(timestamp_path)
            for required_file in self.config.get_run_required_files():
                if self.config.skip_output_file and required_file == "*output.json":
                    continue
                if not regex_matches_any(required_file, files):
                    self.log.error("%s not found in %s", required_file, timestamp_path)
                    valid = False
            
            # Check for dlio_config directory
            for required_folder in self.config.get_run_required_folders():
                if required_folder not in list_dir(timestamp_path):
                    self.log.error("%s directory not found in %s", required_folder, timestamp_path)
                    valid = False
        
        return valid
    
    def run_files_timestamp_check(self):
        """
        Check that all run_files have timestamps matching format "YYYYMMDD_HHmmss"
        and that there are exactly 6 of them.
        """
        # Question: Not enough runs in reference
        # v2.0 only 5 required
        valid = True
        timestamp_pattern = r"^\d{8}_\d{6}$"
        timestamps = []
        
        for _, _, timestamp in self.submissions_logs.run_files:
            timestamps.append(timestamp)
            if not re.match(timestamp_pattern, timestamp):
                self.log.error(
                    "Invalid timestamp format '%s'. Expected format: YYYYMMDD_HHmmss",
                    timestamp
                )
                valid = False
        
        if len(timestamps) != 6:
            self.log.error(
                "Expected 6 run files, but found %d. Timestamps: %s",
                len(timestamps),
                timestamps
            )
            valid = False
        
        return valid
    
    def run_dlio_config_check(self):
        """
        Check that the dlio_config subdirectory in each run timestamp directory
        contains exactly: config.yaml, hydra.yaml, and overrides.yaml (case-sensitive).
        """
        valid = True
        required_files = {"config.yaml", "hydra.yaml", "overrides.yaml"}
        
        for _, _, timestamp in self.submissions_logs.run_files:
            dlio_config_path = os.path.join(self.run_path, timestamp, "dlio_config")
            
            if not os.path.exists(dlio_config_path):
                self.log.error("dlio_config directory not found in %s", dlio_config_path)
                valid = False
                continue
            
            files = set(list_files(dlio_config_path))
            
            # Check for exact match
            if files != required_files:
                self.log.error(
                    "dlio_config in %s has incorrect files. Expected %s, got %s",
                    dlio_config_path,
                    required_files,
                    files
                )
                valid = False
        
        return valid
    
    def run_duration_valid_check(self):
        """
        Check that the gap between consecutive timestamp directories is less than
        the duration of a single run. The gap must be short enough to ensure there
        was no benchmark activity between consecutive runs.
        
        Compares the time delta between consecutive run directory names with the
        duration of each individual run (from start to end time).
        """
        valid = True
        
        # Parse all run data: (run_dict, _, timestamp_dir_name)
        run_dir_time = []
        max_gap = float("inf")
        time_factor = 2
        for run_dict, _, timestamp_dir in self.submissions_logs.run_files:
            try:
                # Parse timestamps from run_dict
                start_time = datetime.fromisoformat(run_dict["start"])
                end_time = datetime.fromisoformat(run_dict["end"])
                
                # Parse the directory timestamp (YYYYMMDD_HHmmss format)
                dir_time = datetime.strptime(timestamp_dir, "%Y%m%d_%H%M%S")
                
                run_duration = (end_time - start_time).total_seconds() * time_factor
                if run_duration < max_gap:
                    max_gap = run_duration

                run_dir_time.append(dir_time)
            except (ValueError, KeyError, TypeError) as e:
                self.log.error(
                    "Failed to parse timestamp data for %s: %s",
                    timestamp_dir,
                    str(e)
                )
                valid = False
                continue
        
        # Check gaps between consecutive runs
        for i in range(len(run_dir_time) - 1):
            current_run = run_dir_time[i]
            next_run = run_dir_time[i + 1]
            
            # Calculate gap between end of current run and start of next run
            gap = (next_run - current_run).total_seconds()
            
            # Gap should be less than the max gap
            if gap >= max_gap:
                self.log.error(
                    "Gap between runs is %s, which is >= the run duration %s. "
                    "Benchmark activity between runs can't be discarted.",
                    gap,
                    max_gap
                )
                valid = False
        
        return valid
    
    def checkpointing_results_json_check(self):
        """
        Check that there is exactly one results.json file in each workload directory
        within the checkpointing directory hierarchy.
        """
        valid = True
        
        if not hasattr(self.submissions_logs, 'checkpointing_files') or not self.submissions_logs.checkpointing_files:
            self.log.warning("No checkpointing files found in submission logs")
            return valid
        
        # Get workload directories
        workload_dirs = list_dir(self.checkpointing_path)
        
        for workload_dir in workload_dirs:
            workload_path = os.path.join(self.checkpointing_path, workload_dir)
            results_files = list_files(workload_path)
            results_json_count = sum(1 for f in results_files if f == "results.json")
            
            if results_json_count != 1:
                self.log.error(
                    "Expected exactly 1 results.json in %s, found %d",
                    workload_path,
                    results_json_count
                )
                valid = False
        
        return valid
    
    def checkpointing_timestamps_check(self):
        """
        Check that there are exactly 10 timestamp directories in YYYYMMDD_HHmmss format
        within the workload directories in the checkpointing hierarchy.
        """
        valid = True
        timestamp_pattern = r"^\d{8}_\d{6}$"
        
        if not hasattr(self.submissions_logs, 'checkpointing_files') or not self.submissions_logs.checkpointing_files:
            self.log.warning("No checkpointing files found in submission logs")
            return valid
        
        # Get workload directories
        workload_dirs = list_dir(self.checkpointing_path)
        
        for workload_dir in workload_dirs:
            workload_path = os.path.join(self.checkpointing_path, workload_dir)
            timestamp_dirs = list_dir(workload_path)
            
            # Validate format of each timestamp directory
            for timestamp_dir in timestamp_dirs:
                if not re.match(timestamp_pattern, timestamp_dir):
                    self.log.error(
                        "Invalid timestamp format '%s' in %s. Expected format: YYYYMMDD_HHmmss",
                        timestamp_dir,
                        workload_path
                    )
                    valid = False
            
            # Check count
            if len(timestamp_dirs) != 10:
                self.log.error(
                    "Expected 10 timestamp directories in %s, found %d",
                    workload_path,
                    len(timestamp_dirs)
                )
                valid = False
        
        return valid
    
    def checkpointing_timestamp_gap_check(self):
        """
        Check that the gap between consecutive timestamp directories is less than
        the duration of a single checkpoint run.
        """
        valid = True
        
        if not hasattr(self.submissions_logs, 'checkpointing_files') or not self.submissions_logs.checkpointing_files:
            self.log.warning("No checkpointing files found in submission logs")
            return valid
        
        # Parse all checkpoint run data
        checkpoint_run_data = []
        max_gap = float("inf")
        
        for checkpoint_dict, _, timestamp_dir in self.submissions_logs.checkpointing_files:
            try:
                # Parse timestamps from checkpoint_dict
                start_time = datetime.fromisoformat(checkpoint_dict["start"])
                end_time = datetime.fromisoformat(checkpoint_dict["end"])
                
                # Parse the directory timestamp (YYYYMMDD_HHmmss format)
                dir_time = datetime.strptime(timestamp_dir, "%Y%m%d_%H%M%S")
                
                run_duration = end_time - start_time
                if run_duration < max_gap:
                    max_gap = run_duration
                
                checkpoint_run_data.append(dir_time)
            except (ValueError, KeyError, TypeError) as e:
                self.log.error(
                    "Failed to parse timestamp data for checkpointing %s: %s",
                    timestamp_dir,
                    str(e)
                )
                valid = False
                continue
        
        # Sort timestamps to check gaps
        checkpoint_run_data.sort()
        
        # Check gaps between consecutive checkpoints
        for i in range(len(checkpoint_run_data) - 1):
            gap = checkpoint_run_data[i + 1] - checkpoint_run_data[i]
            
            if gap >= max_gap:
                self.log.error(
                    "Gap between checkpoints is %s, which is >= the checkpoint duration %s. "
                    "Benchmark activity between checkpoints can't be discarded.",
                    gap,
                    max_gap
                )
                valid = False
        
        return valid
    
    def checkpointing_files_check(self):
        """
        Check that each checkpointing timestamp directory contains:
        - checkpointing_run.stdout.log
        - checkpointing_run.stderr.log
        - *output.json
        - *per_epoch_stats.json
        - *summary.json
        - dlio.log
        - dlio_config/ (subdirectory)
        """
        valid = True
        
        if not hasattr(self.submissions_logs, 'checkpointing_files') or not self.submissions_logs.checkpointing_files:
            self.log.warning("No checkpointing files found in submission logs")
            return valid
        
        for _, _, timestamp in self.submissions_logs.checkpointing_files:
            timestamp_path = os.path.join(self.checkpointing_path, timestamp)
            files = list_files(timestamp_path)
            dirs = list_dir(timestamp_path)
            
            for required_file in self.config.get_checkpoint_required_files():
                if not regex_matches_any(required_file, files):
                    self.log.error("%s not found in %s", required_file, timestamp_path)
                    valid = False
            
            # Check for dlio_config directory
            for required_folder in self.config.get_checkpoint_required_folders():
                if required_folder not in dirs:
                    self.log.error("%s directory not found in %s", required_folder, timestamp_path)
                    valid = False
        
        return valid
    
    def checkpointing_dlio_config_check(self):
        """
        Check that the dlio_config subdirectory in each checkpointing timestamp directory
        contains exactly: config.yaml, hydra.yaml, and overrides.yaml (case-sensitive).
        """
        valid = True
        required_files = {"config.yaml", "hydra.yaml", "overrides.yaml"}
        
        if not hasattr(self.submissions_logs, 'checkpointing_files') or not self.submissions_logs.checkpointing_files:
            self.log.warning("No checkpointing files found in submission logs")
            return valid
        
        for _, _, timestamp in self.submissions_logs.checkpointing_files:
            dlio_config_path = os.path.join(self.checkpointing_path, timestamp, "dlio_config")
            
            if not os.path.exists(dlio_config_path):
                self.log.error("dlio_config directory not found in %s", dlio_config_path)
                valid = False
                continue
            
            files = set(list_files(dlio_config_path))
            
            # Check for exact match
            if files != required_files:
                self.log.error(
                    "dlio_config in %s has incorrect files. Expected %s, got %s",
                    dlio_config_path,
                    required_files,
                    files
                )
                valid = False
        
        return valid
