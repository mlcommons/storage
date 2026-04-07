
from .base import BaseCheck
from ..constants import *
from ..configuration.configuration import Config
from ..loader import SubmissionLogs

import os
import hashlib
import re


class TrainingCheck(BaseCheck):
    """
    A check class for validating training parameters and related properties.
    Inherits from BaseCheck and receives a config and loader instance.
    """

    def __init__(self, log, config: Config, submissions_logs: SubmissionLogs):
        """
        Initialize TrainingChecks with configuration and loader.

        Args:
            config: A Config instance containing submission configuration.
            loader: A SubmissionLogs instance for accessing submission logs.
        """
        # Call parent constructor with the loader's log and submission path
        super().__init__(log=log, path=submissions_logs.loader_metadata.folder)
        self.config = config
        self.submissions_logs = submissions_logs
        self.mode = self.submissions_logs.loader_metadata.mode
        self.model = self.submissions_logs.loader_metadata.benchmark
        self.name = "training checks"
        self.datagen_path = os.path.join(self.path, "datagen")
        self.run_path = os.path.join(self.path, "run")
        self.init_checks()

    def init_checks(self):
        self.checks = []
        self.checks.extend([
            self.verify_datasize_usage,
            self.recalculate_dataset_size,
            self.datagen_minimum_size,
            self.run_data_matches_datasize,
            self.accelerator_utilization_check,
            self.single_host_simulated_accelerators,
            self.identical_accelerators_per_node,
            self.closed_submission_checksum,
            self.closed_submission_parameters,
            self.open_submission_parameters,
            self.mlpstorage_path_args,
            self.mlpstorage_filesystem_check,
        ])

    def verify_datasize_usage(self):
        """
        Verify that the datasize option was used by finding it in the run metadata.
        """
        valid = True
        if self.mode != "training":
            return valid
        
        for summary, metadata, _ in self.submissions_logs.run_files:
            # Check if datasize-related parameters are in the metadata
            params = metadata.get("args", {})
            combined_params = metadata.get("combined_params", {})
            
            if not params and not combined_params:
                self.log.error("No parameters found in metadata to verify datasize usage")
                valid = False
                continue
            
            # Check if dataset-related params are present
            dataset_params = combined_params.get("dataset", {})
            if not dataset_params:
                self.log.error("Dataset parameters not found in metadata")
                valid = False
        
        return valid
    
    def recalculate_dataset_size(self):
        """
        Recalculate minimum dataset size and verify it matches the run's logfile.
        """
        valid = True
        if self.mode != "training":
            return valid
        HOST_MEMORY_MULTIPLIER = 5
        MIN_STEPS_PER_EPOCH = 500
        
        for summary, metadata, _ in self.submissions_logs.run_files:
            try:
                # Get parameters
                combined_params = metadata.get("combined_params", {})
                dataset_params = combined_params.get("dataset", {})
                reader_params = combined_params.get("reader", {})
                
                num_files_train = int(dataset_params.get("num_files_train", 0))
                num_samples_per_file = int(dataset_params.get("num_samples_per_file", 1))
                record_length = float(dataset_params.get("record_length_bytes", 0))
                batch_size = int(reader_params.get("batch_size", 1))
                
                # From summary
                num_accelerators = summary.get("num_accelerators", 1)
                num_hosts = summary.get("num_hosts", 1)
                host_memory_gb = summary.get("host_memory_GB", [0])[0]
                
                if record_length == 0:
                    self.log.error("Record length is 0, cannot calculate dataset size")
                    valid = False
                    continue
                
                # Calculate min samples from steps per epoch
                num_steps_per_epoch = max(MIN_STEPS_PER_EPOCH, 
                                        num_files_train * num_samples_per_file // (batch_size * num_accelerators))
                min_samples_steps = num_steps_per_epoch * batch_size * num_accelerators
                
                # Calculate min samples from host memory
                total_host_memory = num_hosts * host_memory_gb
                min_samples_memory = (total_host_memory * HOST_MEMORY_MULTIPLIER * 
                                    1024 * 1024 * 1024 / record_length)
                
                # Take max of both constraints
                min_samples = max(min_samples_steps, min_samples_memory)
                min_total_files = min_samples / num_samples_per_file
                min_files_size_gb = min_samples * record_length / 1024 / 1024 / 1024
                
                # Verify actual matches expected
                actual_num_files = num_files_train
                if actual_num_files < min_total_files:
                    self.log.error(
                        "Dataset size mismatch: actual files %d < minimum required %d",
                        actual_num_files,
                        int(min_total_files)
                    )
                    valid = False
                
            except (KeyError, ValueError, TypeError) as e:
                self.log.error("Failed to calculate dataset size: %s", str(e))
                valid = False
        
        return valid
    
    def datagen_minimum_size(self):
        """
        Verify that datagen data generated >= datasize calculated.
        """
        valid = True
        if self.mode != "training":
            return valid
        if not self.submissions_logs.datagen_files:
            self.log.warning("No datagen files found")
            return valid
        
        # Get expected size from run
        expected_size = None
        for summary, metadata, _ in self.submissions_logs.run_files:
            dataset_params = metadata.get("combined_params", {}).get("dataset", {})
            num_files = int(dataset_params.get("num_files_train", 0))
            record_length = float(dataset_params.get("record_length_bytes", 0))
            num_samples_per_file = int(dataset_params.get("num_samples_per_file", 1))
            expected_size = num_files * num_samples_per_file * record_length / 1024 / 1024 / 1024
            break
        
        # Check datagen produced at least that much
        for summary, metadata, _ in self.submissions_logs.datagen_files:
            dataset_params = metadata.get("combined_params", {}).get("dataset", {})
            num_files = int(dataset_params.get("num_files_train", 0))
            record_length = float(dataset_params.get("record_length_bytes", 0))
            num_samples_per_file = int(dataset_params.get("num_samples_per_file", 1))
            datagen_size = num_files * num_samples_per_file * record_length / 1024 / 1024 / 1024
            
            if expected_size and datagen_size < expected_size:
                self.log.error(
                    "Datagen size %.2fGiB is less than required %.2fGiB",
                    datagen_size,
                    expected_size
                )
                valid = False
        
        return valid
    
    def run_data_matches_datasize(self):
        """
        Verify that run data matches the calculated datasize exactly.
        """
        # Question: Subfolders? 
        # What are the true values of the dataset
        valid = True
        if self.mode != "training":
            return valid
        
        for summary, metadata, _ in self.submissions_logs.run_files:
            num_files_train = summary.get("num_files_train", None)
            num_files_eval = summary.get("num_files_eval", None)
            
            if num_files_train is None:
                self.log.error("num_files_train not set")
                valid = False
            
            if num_files_train > self.config.get_num_train_files(self.model):
                self.log.error("num_files_train should be lower than in dataset")
                valid = False

            if num_files_eval is None:
                self.log.error("num_files_eval not set")
                valid = False

            if num_files_eval > self.config.get_num_eval_files(self.model):
                self.log.error("num_files_eval should be lower than in dataset")
                valid = False
        
        return valid
    
    def accelerator_utilization_check(self):
        """
        Check that AU (Accelerator Utilization) meets minimum requirements.
        """
        valid = True
        if self.mode != "training":
            return valid
        for summary, metadata, _ in self.submissions_logs.run_files:
            metrics = summary.get("metric", {})
            au_mean = metrics.get("train_au_mean_percentage", 0)
            au_expectation = metrics.get("train_au_meet_expectation", "")
            
            if au_expectation != "success":
                self.log.error(
                    "AU check failed: expected 'success', got '%s' (AU: %.2f%%)",
                    au_expectation,
                    au_mean
                )
                valid = False
        
        return valid
    
    def single_host_simulated_accelerators(self):
        """
        For single-host submissions, verify sufficient simulated accelerators.
        """
        valid = True
        if self.mode != "training":
            return valid
        for summary, metadata, _ in self.submissions_logs.run_files:
            num_hosts = summary.get("num_hosts", 1)
            num_accelerators = summary.get("num_accelerators", 1)
            
            if num_hosts == 1 and num_accelerators < 4:
                self.log.warning(
                    "Single-host submission has only %d accelerators. Consider increasing via --num-accelerators",
                    num_accelerators
                )
        
        return valid
    
    def single_host_client_limit(self):
        """
        For single-host submissions, fail if more than one client node used.
        """
        valid = True
        if self.mode != "training":
            return valid
        for summary, metadata, _ in self.submissions_logs.run_files:
            num_hosts = summary.get("num_hosts", 1)
            
            if num_hosts == 1:
                args = metadata.get("args", {})
                hosts = args.get("hosts", [])
                
                if len(hosts) > 1:
                    self.log.error(
                        "Single-host submission but %d client nodes specified: %s",
                        len(hosts),
                        hosts
                    )
                    valid = False
        
        return valid
    
    def identical_accelerators_per_node(self):
        """
        For distributed submissions, verify all nodes have identical accelerator count.
        """
        valid = True
        if self.mode != "training":
            return valid
        
        for summary, metadata, _ in self.submissions_logs.run_files:
            num_hosts = summary.get("num_hosts", 1)
            num_accelerators = summary.get("num_accelerators", 1)
            
            if num_hosts > 1:
                # For distributed runs, accelerators should be divisible by hosts
                if num_accelerators % num_hosts != 0:
                    self.log.error(
                        "Distributed submission: %d accelerators not evenly divisible by %d hosts",
                        num_accelerators,
                        num_hosts
                    )
                    valid = False
        
        return valid
    
    def closed_submission_checksum(self):
        """
        For CLOSED submissions, verify code directory MD5 checksum.
        """
        # TODO
        return True
    
    def closed_submission_parameters(self):
        """
        For CLOSED submissions, verify only allowed parameters are modified.
        """
        valid = True
        if self.mode != "training":
            return valid
        
        # Allowed parameters for CLOSED
        allowed_params = {
            "dataset.num_files_train",
            "dataset.num_subfolders_train",
            "dataset.data_folder",
            "reader.read_threads",
            "reader.computation_threads",
            "reader.transfer_size",
            "reader.prefetch_size",
            "reader.odirect",
            "storage.storage_root",
            "storage.storage_type"
        }
        
        for summary, metadata, _ in self.submissions_logs.run_files:
            verification = metadata.get("verification", "open")
            
            if verification == "closed":
                params_dict = metadata.get("params_dict", {})
                
                for param_key in params_dict.keys():
                    if param_key not in allowed_params:
                        self.log.error(
                            "CLOSED submission modifies disallowed parameter: %s",
                            param_key
                        )
                        valid = False
        
        return valid
    
    def open_submission_parameters(self):
        """
        For OPEN submissions, verify only allowed parameters are modified.
        """
        valid = True
        if self.mode != "training":
            return valid

        # Additional allowed parameters for OPEN (beyond CLOSED)
        open_allowed_params = {
            "framework",
            "dataset.format",
            "dataset.num_samples_per_file",
            "reader.data_loader"
        }
        
        # All CLOSED params are also allowed in OPEN
        closed_params = {
            "dataset.num_files_train",
            "dataset.num_subfolders_train",
            "dataset.data_folder",
            "reader.read_threads",
            "reader.computation_threads",
            "reader.transfer_size",
            "reader.prefetch_size",
            "reader.odirect",
            "storage.storage_root",
            "storage.storage_type"
        }
        
        allowed_params = closed_params | open_allowed_params
        
        for summary, metadata, _ in self.submissions_logs.run_files:
            verification = metadata.get("verification", "open")
            
            if verification == "open":
                params_dict = metadata.get("params_dict", {})
                
                for param_key in params_dict.keys():
                    if param_key not in allowed_params:
                        self.log.error(
                            "OPEN submission modifies disallowed parameter: %s",
                            param_key
                        )
                        valid = False
        
        return valid
    
    def mlpstorage_path_args(self):
        """
        Verify dataset and output paths are set and different.
        """
        valid = True
        if self.mode != "training":
            return valid
        
        for summary, metadata, _ in self.submissions_logs.run_files:
            args = metadata.get("args", {})
            data_dir = args.get("data_dir")
            results_dir = args.get("results_dir")
            
            if not data_dir:
                self.log.error("data_dir not set in arguments")
                valid = False
            
            if not results_dir:
                self.log.error("results_dir not set in arguments")
                valid = False
            
            if data_dir and results_dir and data_dir == results_dir:
                self.log.error(
                    "data_dir and results_dir must be different: both are %s",
                    data_dir
                )
                valid = False
        
        return valid
    
    def mlpstorage_filesystem_check(self):
        """
        Verify dataset and output are on different filesystems.
        This would require checking 'df' output in the logfiles.
        """
        valid = True
        # Question: where to look for this?
        if self.mode != "training":
            return valid
        # TODO      
        return valid
