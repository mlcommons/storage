
from .base import BaseCheck
from ..constants import *
from ..configuration.configuration import Config
from ..loader import SubmissionLogs

import os
import re
import yaml


class CheckpointingCheck(BaseCheck):
    """
    A check class for validating checkpointing parameters and related properties.
    Inherits from BaseCheck and receives a config and loader instance.
    """

    def __init__(self, log, config: Config, submissions_logs: SubmissionLogs):
        """
        Initialize CheckpointingChecks with configuration and loader.

        Args:
            config: A Config instance containing submission configuration.
            loader: A SubmissionLogs instance for accessing submission logs.
        """
        # Call parent constructor with the loader's log and submission path
        super().__init__(log=log, path=submissions_logs.loader_metadata.folder)
        self.config = config
        self.submissions_logs = submissions_logs.checkpoint_files
        self.name = "checkpointing checks"
        self.mode = submissions_logs.loader_metadata.mode
        self.benchmark = submissions_logs.loader_metadata.benchmark
        self.checks = []
        self.checkpointing_path = self.path
        self.init_checks()
    
    def init_checks(self):
        """Initialize the list of checks to run."""
        self.checks = [
            self.checkpoint_data_size_ratio,
            self.fsync_verification,
            self.model_configuration_req,
            self.closed_mpi_processes,
            self.closed_accelerators_per_host,
            self.aggregate_accelerator_memory,
            self.closed_checkpoint_parameters,
            self.checkpoint_path_args,
            self.subset_run_validation,
        ]
    
    def checkpoint_data_size_ratio(self):
        """
        Verify that checkpoint data written per node > 3x node memory.
        """
        valid = True
        if self.mode != "checkpointing":
            return valid
        
        for summary, metadata, _ in self.submissions_logs:
            checkpoint_size_gb = summary.get("metric", {}).get("checkpoint_size_GB", 0)
            host_memory_gb = summary.get("host_memory_GB", [0])[0]
            num_hosts = summary.get("num_hosts", 1)
            
            if checkpoint_size_gb == 0 or host_memory_gb == 0:
                continue
            
            # Data written per node
            data_per_node = checkpoint_size_gb / num_hosts
            min_required = 3 * host_memory_gb
            
            if data_per_node < min_required:
                self.log.warning(
                    "Checkpoint data per node %.2fGiB < 3x memory %.2fGiB. "
                    "Cache flush may be needed.",
                    data_per_node,
                    min_required
                )
        
        return valid
    
    def fsync_verification(self):
        """
        Verify that fsync is enabled in checkpoint configuration.
        """
        valid = True
        if self.mode != "checkpointing":
            return valid
        
        for summary, metadata, _ in self.submissions_logs:
            combined_params = metadata.get("combined_params", {})
            checkpoint_params = combined_params.get("checkpoint", {})
            fsync_enabled = checkpoint_params.get("fsync", False)
            
            if not fsync_enabled:
                self.log.error("Checkpoint fsync is not enabled in configuration")
                valid = False
        
        return valid
    
    def model_configuration_req(self):
        """
        Verify benchmark uses one of the four supported models.
        """
        valid = True
        if self.mode != "checkpointing":
            return valid
        
        allowed_models = {"8b", "70b", "405b", "1t"}
        
        for summary, metadata, _ in self.submissions_logs:
            model_name = metadata.get("args", {}).get("model", "").lower()
            
            # Extract just the size part (8b, 70b, etc.)
            model_size = re.search(r"(8b|70b|405b|1t)", model_name)
            
            if not model_size or model_size.group(1) not in allowed_models:
                self.log.error(
                    "Invalid model '%s'. Must be one of: %s",
                    model_name,
                    allowed_models
                )
                valid = False
        
        return valid
    
    def closed_mpi_processes(self):
        """
        For CLOSED submissions, verify MPI processes match requirements per model.
        """
        valid = True
        if self.mode != "checkpointing":
            return valid
        
        model_process_requirements = {
            "8b": 8,
            "70b": 64,
            "405b": 512,
            "1t": 1024
        }
        
        for summary, metadata, _ in self.submissions_logs:
            verification = metadata.get("verification", "closed")
            
            if verification == "closed":
                checkpoint_mode = metadata.get("params_dict", {}).get("checkpoint.mode", "").lower()
                model_name = metadata.get("args", {}).get("model", "").lower()
                num_processes = metadata.get("args", {}).get("num_processes", 0)
                if checkpoint_mode == "subset":
                    if num_processes != 8:
                        self.log.error(
                            "CLOSED submission with model %s in subset mode requires %d processes, got %d",
                            model_key,
                            8,
                            num_processes
                        )
                        valid = False
                else:
                    model_size = re.search(r"(8b|70b|405b|1t)", model_name)
                    if model_size:
                        model_key = model_size.group(1)
                        required_processes = model_process_requirements.get(model_key)
                        
                        if required_processes and num_processes != required_processes:
                            self.log.error(
                                "CLOSED submission with model %s requires %d processes, got %d",
                                model_key,
                                required_processes,
                                num_processes
                            )
                            valid = False
        
        return valid
    
    def closed_accelerators_per_host(self):
        """
        For CLOSED submissions, verify accelerators per host > 4 and total matches requirement.
        """
        valid = True
        if self.mode != "checkpointing":
            return valid

        for summary, metadata, _ in self.submissions_logs:
            verification = metadata.get("verification", "open")
            
            if verification == "closed":
                num_accelerators = summary.get("num_accelerators", 0)
                num_hosts = summary.get("num_hosts", 1)
                
                accelerators_per_host = num_accelerators / num_hosts if num_hosts > 0 else 0
                
                if accelerators_per_host <= 4:
                    self.log.error(
                        "CLOSED submission: accelerators per host %.2f must be > 4",
                        accelerators_per_host
                    )
                    valid = False
        
        return valid
    
    def aggregate_accelerator_memory(self):
        """
        Verify total accelerator memory >= checkpoint size.
        H100 has 80GB per accelerator.
        """
        valid = True
        if self.mode != "checkpointing":
            return valid
        
        ACCELERATOR_MEMORY_GB = 80  # H100
        
        for summary, metadata, _ in self.submissions_logs:
            checkpoint_size_gb = summary.get("metric", {}).get("checkpoint_size_GB", 0)
            num_accelerators = summary.get("num_accelerators", 0)
            
            total_accelerator_memory = num_accelerators * ACCELERATOR_MEMORY_GB
            
            if total_accelerator_memory < checkpoint_size_gb:
                self.log.error(
                    "Aggregate accelerator memory %.2fGiB < checkpoint size %.2fGiB",
                    total_accelerator_memory,
                    checkpoint_size_gb
                )
                valid = False
        
        return valid
    
    def _get_nested_value(self, config_dict, key_path):
        """
        Get a value from nested dictionary using dot notation.
        Example: "checkpoint.fsync" -> config_dict["checkpoint"]["fsync"]
        
        Args:
            config_dict: The dictionary to search
            key_path: Dot-separated key path
            
        Returns:
            The value if found, None otherwise
        """
        keys = key_path.split(".")
        current = config_dict
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _get_nested_items(self, d, prefix = ""):
        for key, value in d.items():
            if isinstance(value, dict):
                p = prefix + "." if prefix != "" else ""
                yield from self._get_nested_items(value, prefix = p + key)
            else:
                p = prefix + "." if prefix != "" else prefix
                yield (p + key, value)
    
    def closed_checkpoint_parameters(self):
        """
        For CLOSED submissions, verify yaml parameters match reference file.
        Only checkpoint_folder is allowed to differ.
        """
        
        config_ref_path = os.path.join(os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            os.pardir,
            "configs",
            "dlio",
            "workload"
        )
        config_ref_file = self.config.get_checkpoint_file(self.benchmark)
        valid = True
        if self.mode != "checkpointing":
            return valid
        # Load reference YAML file
        config_ref_full_path = os.path.join(config_ref_path, config_ref_file)
        if not os.path.exists(config_ref_full_path):
            self.log.error(
                "Reference config file not found: %s",
                config_ref_full_path
            )
            return False
        
        try:
            with open(config_ref_full_path, 'r') as f:
                reference_config = yaml.safe_load(f)
        except Exception as e:
            self.log.error(
                "Failed to load reference config file %s: %s",
                config_ref_full_path,
                str(e)
            )
            return False
        
        allowed_diff_params = {
            "checkpoint.checkpoint_folder"
        }
        for summary, metadata, _ in self.submissions_logs:
            verification = metadata.get("verification", "open")
            if verification == "closed":
                yaml_params = metadata.get("yaml_params", {})
                
                # Compare yaml parameters with reference config
                for key, value in self._get_nested_items(yaml_params):
                    # Skip allowed differing parameters
                    if key in allowed_diff_params:
                        continue
                    
                    # Navigate reference config to find the parameter
                    ref_value = self._get_nested_value(reference_config, key)
                    
                    if ref_value is None:
                        self.log.error(
                            "Parameter %s not found in reference config",
                            key
                        )
                        valid = False
                    elif value != ref_value:
                        self.log.error(
                            "CLOSED submission parameter %s differs from reference. "
                            "Expected: %s, Got: %s",
                            key,
                            ref_value,
                            value
                        )
                        valid = False
        return valid
    
    def checkpoint_path_args(self):
        """
        Verify checkpoint folder and output paths are set and different.
        """
        valid = True
        if self.mode != "checkpointing":
            return valid
        
        for summary, metadata, _ in self.submissions_logs:
            args = metadata.get("args", {})
            checkpoint_folder = args.get("checkpoint_folder")
            results_dir = args.get("results_dir")
            
            if not checkpoint_folder:
                self.log.error("checkpoint_folder not set in arguments")
                valid = False
            
            if not results_dir:
                self.log.error("results_dir not set in arguments")
                valid = False
            
            if checkpoint_folder and results_dir and checkpoint_folder == results_dir:
                self.log.error(
                    "checkpoint_folder and results_dir must be different: both are %s",
                    checkpoint_folder
                )
                valid = False
        
        return valid
    
    def subset_run_validation(self):
        """
        For subset runs, verify exactly 8 accelerators and not 8B model.
        """
        valid = True
        if self.mode != "checkpointing":
            return valid
        
        for summary, metadata, _ in self.submissions_logs:
            params_dict = metadata.get("params_dict", {})
            checkpoint_mode = params_dict.get("checkpoint.mode", "")
            
            if checkpoint_mode == "subset":
                num_accelerators = summary.get("num_accelerators", 0)
                model_name = metadata.get("args", {}).get("model", "").lower()
                
                if num_accelerators != 8:
                    self.log.error(
                        "Subset run requires exactly 8 accelerators, got %d",
                        num_accelerators
                    )
                    valid = False
                
                if "8b" in model_name:
                    self.log.error(
                        "Subset run cannot use 8B model: %s",
                        model_name
                    )
                    valid = False
        
        return valid
