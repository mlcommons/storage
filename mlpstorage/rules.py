import abc
import enum
import json
import os
import yaml

from dataclasses import dataclass, field
from datetime import datetime
from pprint import pprint, pformat
from typing import List, Dict, Any, Optional, Tuple

from mlpstorage.config import (MODELS, PARAM_VALIDATION, MAX_READ_THREADS_TRAINING, LLM_MODELS, BENCHMARK_TYPES,
                               DATETIME_STR, LLM_ALLOWED_VALUES, LLM_SUBSET_PROCS, HYDRA_OUTPUT_SUBDIR, UNET)
from mlpstorage.mlps_logging import setup_logging
from mlpstorage.utils import is_valid_datetime_format


class RuleState(enum.Enum):
    OPEN = "open"
    CLOSED = "closed"
    INVALID = "invalid"


@dataclass
class Issue:
    validation: PARAM_VALIDATION
    message: str
    parameter: Optional[str] = None
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    severity: str = "error"
    
    def __str__(self):
        result = f"[{self.severity.upper()}] {self.message}"
        if self.parameter:
            result += f" (Parameter: {self.parameter}"
            if self.expected is not None and self.actual is not None:
                result += f", Expected: {self.expected}, Actual: {self.actual}"
            result += ")"
        return result


@dataclass
class RunID:
    program: str
    command: str
    model: str
    run_datetime: str

    def __str__(self):
        id_str = self.program
        if self.command:
            id_str += f"_{self.command}"
        if self.model:
            id_str += f"_{self.model}"
        id_str += f"_{self.run_datetime}"
        return id_str


@dataclass
class ProcessedRun:
    run_id: RunID
    benchmark_type: str
    run_parameters: Dict[str, Any]
    run_metrics: Dict[str, Any]
    issues: List[Issue] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if the run is valid (no issues with INVALID validation)"""
        return not any(issue.validation == PARAM_VALIDATION.INVALID for issue in self.issues)
    
    def is_closed(self) -> bool:
        """Check if the run is valid for closed submission"""
        if not self.is_valid():
            return False
        return all(issue.validation != PARAM_VALIDATION.OPEN for issue in self.issues)


@dataclass
class HostMemoryInfo:
    """Detailed memory information for a host"""
    total: int  # Total physical memory in bytes
    available: Optional[int]  # Memory available for allocation
    used: Optional[int]  # Memory currently in use
    free: Optional[int]  # Memory not being used
    active: Optional[int]  # Memory actively used
    inactive: Optional[int]  # Memory marked as inactive
    buffers: Optional[int]  # Memory used for buffers
    cached: Optional[int]  # Memory used for caching
    shared: Optional[int]  # Memory shared between processes

    @classmethod
    def from_psutil_dict(cls, data: Dict[str, int]) -> 'HostMemoryInfo':
        """Create a HostMemoryInfo instance from a dictionary"""
        return cls(
            total=data.get('total', 0),
            available=data.get('available', 0),
            used=data.get('used', 0),
            free=data.get('free', 0),
            active=data.get('active', 0),
            inactive=data.get('inactive', 0),
            buffers=data.get('buffers', 0),
            cached=data.get('cached', 0),
            shared=data.get('shared', 0)
        )

    @classmethod
    def from_proc_meminfo_dict(cls, data: Dict[str, Any]) -> 'HostMemoryInfo':
        """Create a HostMemoryInfo instance from a dictionary"""
        converted_dict = dict(
            total=data.get('MemTotal', 0) * 1024,
            available=data.get('MemAvailable', 0) * 1024,
            used=data.get('MemUsed', 0) * 1024,
            free=data.get('MemFree', 0) * 1024,
            active=data.get('Active', 0) * 1024,
            inactive=data.get('Inactive', 0) * 1024,
            buffers=data.get('Buffers', 0) * 1024,
            cached=data.get('Cached', 0) * 1024,
            shared=data.get('Shmem', 0) * 1024
        )
        converted_dict = {k: int(v.split(" ")[0]) for k, v in converted_dict.items()}
        return cls(**converted_dict)

    @classmethod
    def from_total_mem_int(cls, total_mem_int: int) -> 'HostMemoryInfo':
        """Create a HostMemoryInfo instance from total memory in bytes"""
        return cls(
            total=total_mem_int,
            available=None,
            used=None,
            free=None,
            active=None,
            inactive=None,
            buffers=None,
            cached=None,
            shared=None
        )


@dataclass
class HostCPUInfo:
    """CPU information for a host"""
    num_cores: int = 0  # Number of physical CPU cores
    num_logical_cores: int = 0  # Number of logical CPU cores (with hyperthreading)
    model: str = ""  # CPU model name
    architecture: str = ""  # CPU architecture

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HostCPUInfo':
        """Create a HostCPUInfo instance from a dictionary"""
        return cls(
            num_cores=data.get('num_cores', 0),
            num_logical_cores=data.get('num_logical_cores', 0),
            model=data.get('model', ""),
            architecture=data.get('architecture', ""),
        )


@dataclass
class HostInfo:
    """Information about a single host in the system"""
    hostname: str
    memory: HostMemoryInfo = field(default_factory=HostMemoryInfo)
    cpu: Optional[HostCPUInfo] = None

    @classmethod
    def from_dict(cls, hostname: str, data: Dict[str, Any]) -> 'HostInfo':
        """Create a HostInfo instance from a dictionary"""
        memory_info = data.get('memory_info', {})
        cpu_info = data.get('cpu_info', {})

        # Determine which memory info constructor to use based on the data structure
        if isinstance(memory_info, dict):
            # Check if it looks like psutil data
            if 'total' in memory_info and isinstance(memory_info['total'], int):
                memory = HostMemoryInfo.from_psutil_dict(memory_info)
            # Check if it looks like proc_meminfo data
            elif 'MemTotal' in memory_info:
                memory = HostMemoryInfo.from_proc_meminfo_dict(memory_info)
            else:
                # Default to empty memory info if we can't determine the format
                memory = HostMemoryInfo()
        else:
            memory = HostMemoryInfo()

        # Handle the case where cpu_info is None or empty
        cpu = None
        if cpu_info:
            cpu = HostCPUInfo.from_dict(cpu_info)

        return cls(
            hostname=hostname,
            memory=memory,
            cpu=cpu,
        )


class ClusterInformation:
    """
    Comprehensive system information for all hosts in the benchmark environment.
    This includes detailed memory, CPU, and accelerator information.
    """

    def __init__(self, host_info_list: List[str], logger, calculate_aggregated_info=True):
        self.logger = logger
        self.host_info_list = host_info_list

        # Aggregated information across all hosts
        self.total_memory_bytes = 0
        self.total_cores = 0

        if calculate_aggregated_info:
            self.calculate_aggregated_info()

    def as_dict(self):
        return {
            "total_memory_bytes": self.total_memory_bytes,
            "total_cores": self.total_cores,
        }

    def calculate_aggregated_info(self):
        """Calculate aggregated system information across all hosts"""
        for host_info in self.host_info_list:
            self.total_memory_bytes += host_info.memory.total
            if host_info.cpu:
                self.total_cores += host_info.cpu.num_cores

    @classmethod
    def from_dlio_summary_json(cls, summary, logger) -> 'ClusterInformation':
        host_memories = summary.get("host_memory_GB")
        host_cpus = summary.get("host_cpu_count")
        num_hosts = summary.get("num_hosts")
        host_info_list = []
        inst = cls(host_info_list, logger, calculate_aggregated_info=False)
        inst.total_memory_bytes = sum(host_memories) * 1024 * 1024 * 1024
        inst.total_cores = sum(host_cpus)
        return inst



class BenchmarkResult:
    """
    Represents the result files from a benchmark run.
    Processes the directory structure to extract metadata and metrics.
    """

    def __init__(self, benchmark_result_root_dir, logger):
        self.benchmark_result_root_dir = benchmark_result_root_dir
        self.logger = logger
        self.metadata = None
        self.summary = None
        self.hydra_configs = {}
        self.issues = []
        self._process_result_directory()

    def _process_result_directory(self):
        """Process the result directory to extract metadata and metrics"""
        # Find and load metadata file
        metadata_files = [f for f in os.listdir(self.benchmark_result_root_dir)
                          if f.endswith('_metadata.json')]

        if metadata_files:
            metadata_path = os.path.join(self.benchmark_result_root_dir, metadata_files[0])
            try:
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.logger.verbose(f"Loaded metadata from {metadata_path}")
            except Exception as e:
                self.logger.error(f"Failed to load metadata from {metadata_path}: {e}")

        # Find and load DLIO summary file
        summary_path = os.path.join(self.benchmark_result_root_dir, 'summary.json')
        self.logger.debug(f'Looking for DLIO summary at {summary_path}...')
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    self.summary = json.load(f)
                self.logger.verbose(f"Loaded DLIO summary from {summary_path}")
            except Exception as e:
                self.logger.error(f"Failed to load DLIO summary from {summary_path}: {e}")

        # Find and load Hydra config files if they exist
        hydra_dir = os.path.join(self.benchmark_result_root_dir, HYDRA_OUTPUT_SUBDIR)
        self.logger.debug(f'Looking for Hydra configs at {hydra_dir}...')
        if os.path.exists(hydra_dir) and os.path.isdir(hydra_dir):
            for config_file in os.listdir(hydra_dir):
                if config_file.endswith('.yaml'):
                    config_path = os.path.join(hydra_dir, config_file)
                    try:
                        with open(config_path, 'r') as f:
                            self.hydra_configs[config_file] = yaml.load(f, Loader=yaml.Loader)
                        self.logger.verbose(f"Loaded Hydra config from {config_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to load Hydra config from {config_path}: {e}")


class BenchmarkRun:
    """
    Represents a benchmark run with all parameters and system information.
    Can be constructed either from a benchmark instance or from result files.
    """
    def __init__(self, benchmark_result=None, benchmark_instance=None, logger=None):
        self.logger = logger
        if benchmark_result is None and benchmark_instance is None:
            self.logger.error(f"The BenchmarkRun instance needs either a benchmark_result or a benchmark_instance.")
            raise ValueError("Either benchmark_result or benchmark_instance must be provided")
        if benchmark_result and benchmark_instance:
            self.logger.error(f"Both benchmark_result and benchmark_instance provided, which is not supported.")
            raise ValueError("Only one of benchmark_result and benchmark_instance can be provided")
            
        self.benchmark_type = None
        self.model = None
        self.command = None
        self.num_processes = None
        self.parameters = dict()
        self.override_parameters = dict()
        self.system_info = None
        self.metrics = {}
        self._run_id = None
        self.run_datetime = None
        self.result_root_dir = None

        self.benchmark_result = benchmark_result
        self.benchmark_instance = benchmark_instance

        if benchmark_instance:
            self._process_benchmark_instance(benchmark_instance)
            self.post_execution = False
        elif benchmark_result:
            self._process_benchmark_result(benchmark_result)
            self.post_execution = True
        else:
            self.logger.error(f"Neither benchmark_result nor benchmark_instance provided.")
            raise ValueError("Either benchmark_result or benchmark_instance must be provided")

        self._run_id = RunID(program=self.benchmark_type.name, command=self.command,  model=self.model,
                            run_datetime=self.run_datetime)
        self.logger.info(f"Found benchmark run: {self.run_id}")

    @property
    def run_id(self):
        if self.post_execution:
            return self.benchmark_result.benchmark_result_root_dir
        else:
            return self._run_id

    def as_dict(self):
        """Convert the BenchmarkRun object to a dictionary"""
        return {
            "run_id": str(self.run_id),
            "benchmark_type": self.benchmark_type.name,
            "model": self.model,
            "command": self.command,
            "parameters": self.parameters,
            "system_info": self.system_info.as_dict() if self.system_info else None,
            "metrics": self.metrics,
        }

    def _process_benchmark_instance(self, benchmark_instance):
        """Extract parameters and system info from a running benchmark instance"""
        self.benchmark_type = benchmark_instance.BENCHMARK_TYPE
        self.model = getattr(benchmark_instance.args, 'model', None)
        self.command = getattr(benchmark_instance.args, 'command', None)
        self.run_datetime = benchmark_instance.run_datetime
        self.num_processes = benchmark_instance.args.num_processes
        
        # Extract parameters from the benchmark instance
        if hasattr(benchmark_instance, 'combined_params'):
            self.parameters = benchmark_instance.combined_params
        else:
            # Fallback to args if combined_params not available
            self.parameters = vars(benchmark_instance.args)

        self.override_parameters = benchmark_instance.params_dict
            
        # Extract system information
        if hasattr(benchmark_instance, 'cluster_information'):
            self.system_info = benchmark_instance.cluster_information

    def _process_benchmark_result(self, benchmark_result):
        """Extract parameters and system info from result files"""
        # Process the summary and hydra configs to find what was run
        summary_workload = benchmark_result.summary.get('workload', {})
        hydra_workload_config = benchmark_result.hydra_configs.get("config.yaml", {}).get("workload", {})
        hydra_workflow = hydra_workload_config.get("workflow", {})
        workflow = (
            hydra_workflow.get('generate_data', {}),
            hydra_workflow.get('train', {}),
            hydra_workflow.get('checkpoint', {}),
        )
        # Get benchmark type based on workflow
        if workflow[0] or workflow[1]:
            self.benchmark_type = BENCHMARK_TYPES.training
        elif workflow[2]:
            self.benchmark_type = BENCHMARK_TYPES.checkpointing

        self.model = hydra_workload_config.get('model', {}).get("name")
        self.num_processes = benchmark_result.summary["num_accelerators"]

        # Set command for training
        if workflow[0] and not any([workflow[1], workflow[2]]):
            self.command = "datagen"
        if workflow[1] and not any([workflow[0], workflow[2]]):
            self.command = "run_benchmark"

        self.run_datetime = benchmark_result.summary.get("start")
        self.parameters = benchmark_result.hydra_configs.get("config.yaml", {}).get("workload", {})

        for param in benchmark_result.hydra_configs.get("overrides.yaml", list()):
            p, v = param.split('=')
            if p.startswith('++workload.'):
                self.override_parameters[p[len('++workload.'):]] = v

        self.metrics = benchmark_result.summary.get("metric")
        self.system_info = ClusterInformation.from_dlio_summary_json(benchmark_result.summary, self.logger)


class RulesChecker(abc.ABC):
    """
    Base class for rule checkers that verify benchmark runs against rules.
    """
    def __init__(self, benchmark_run, logger):
        self.benchmark_run = benchmark_run
        self.logger = logger
        self.issues = []
        
        # Dynamically find all check methods
        self.check_methods = [getattr(self, method) for method in dir(self) 
                             if callable(getattr(self, method)) and method.startswith('check_')]
        
    def run_checks(self) -> List[Issue]:
        """Run all check methods and return a list of issues"""
        self.issues = []
        for check_method in self.check_methods:
            try:
                self.logger.debug(f"Running check {check_method.__name__}")
                method_issues = check_method()
                if method_issues:
                    if isinstance(method_issues, list):
                        self.issues.extend(method_issues)
                    else:
                        self.issues.append(method_issues)
            except Exception as e:
                self.logger.error(f"Error running check {check_method.__name__}: {e}")
                self.issues.append(Issue(
                    validation=PARAM_VALIDATION.INVALID,
                    message=f"Check {check_method.__name__} failed with error: {e}",
                    severity="error"
                ))
        
        return self.issues
    
    @abc.abstractmethod
    def check_benchmark_type(self) -> Optional[Issue]:
        """Check if the benchmark type is valid"""
        pass


class TrainingRulesChecker(RulesChecker):
    """Rules checker for training benchmarks"""
    
    def check_benchmark_type(self) -> Optional[Issue]:
        if self.benchmark_run.benchmark_type != BENCHMARK_TYPES.training:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Invalid benchmark type: {self.benchmark_run.benchmark_type}",
                parameter="benchmark_type",
                expected=BENCHMARK_TYPES.training,
                actual=self.benchmark_run.benchmark_type
            )
        return None
    
    def check_num_files_train(self) -> Optional[Issue]:
        """Check if the number of training files meets the minimum requirement"""
        if 'dataset' not in self.benchmark_run.parameters:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message="Missing dataset parameters",
                parameter="dataset"
            )
            
        dataset_params = self.benchmark_run.parameters['dataset']
        if 'num_files_train' not in dataset_params:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message="Missing num_files_train parameter",
                parameter="dataset.num_files_train"
            )
            
        # Calculate required file count based on system info
        # This is a simplified version - in practice you'd use the calculate_training_data_size function
        configured_num_files = int(dataset_params['num_files_train'])
        dataset_params = self.benchmark_run.parameters['dataset']
        reader_params = self.benchmark_run.parameters['reader']
        required_num_files, _, _ = calculate_training_data_size(None, self.benchmark_run.system_info, dataset_params, reader_params, self.logger, self.benchmark_run.num_processes)

        if configured_num_files < required_num_files:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Insufficient number of training files",
                parameter="dataset.num_files_train",
                expected=f">= {required_num_files}",
                actual=configured_num_files
            )
        
        return None

    def check_allowed_params(self) -> Optional[Issue]:
        """
        This method will verify that the only parameters that were set were the allowed parameters.
        Allowed for closed:
          - dataset.num_files_train
          - dataset.num_subfolders_train
          -
        :return:
        """
        closed_allowed_params = ['dataset.num_files_train', 'dataset.num_subfolders_train', 'dataset.data_folder',
                                 'reader.read_threads', 'reader.computation_threads', 'reader.transfer_size',
                                 'reader.odirect', 'reader.prefetch_size', 'checkpoint.checkpoint_folder',
                                 'storage.storage_type', 'storage.storage_root']
        open_allowed_params = ['framework', 'dataset.format', 'dataset.num_samples_per_file', 'reader.data_loader']
        issues = []
        for param, value in self.benchmark_run.override_parameters.items():
            self.logger.debug(f"Processing override parameter: {param} = {value}")
            if param in closed_allowed_params:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.CLOSED,
                    message=f"Closed parameter override allowed: {param} = {value}",
                    parameter="Overrode Parameters",
                    actual=value
                ))
            elif param in open_allowed_params:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.OPEN,
                    message=f"Open parameter override allowed: {param} = {value}",
                    parameter="Overrode Parameters",
                    actual=value
                ))
            else:
                issues.append(Issue(
                    validation=PARAM_VALIDATION.INVALID,
                    message=f"Disallowed parameter override: {param} = {value}",
                    parameter="Overrode Parameters",
                    expected="None",
                    actual=value
                ))
        return issues

    def check_odirect_supported_model(self) -> Optional[Issue]:
        # The 'reader.odirect' option is only supported if the model is "Unet3d"
        if self.benchmark_run.model != UNET and self.benchmark_run.parameters.get('reader', {}).get('odirect'):
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message="The reader.odirect option is only supported for Unet3d model",
                parameter="reader.odirect",
                expected="False",
                actual=self.benchmark_run.parameters.get('reader', {}).get('odirect')
            )
        else:
            return None

    def check_checkpoint_files_in_code(self) -> Optional[Issue]:
        pass

    def check_num_epochs(self) -> Optional[Issue]:
        pass

    def check_inter_test_times(self) -> Optional[Issue]:
        pass

    def check_file_system_caching(self) -> Optional[Issue]:
        pass




class CheckpointingRulesChecker(RulesChecker):
    """Rules checker for checkpointing benchmarks"""
    def check_benchmark_type(self) -> Optional[Issue]:
        pass


class BenchmarkRunVerifier:

    def __init__(self, benchmark_run, logger):
        if isinstance(benchmark_run, BenchmarkRun):
            self.benchmark_run = benchmark_run
        elif "mlpstorage.benchmarks." in str(type(benchmark_run)):
            self.benchmark_run = BenchmarkRun(benchmark_instance=benchmark_run, logger=logger)

        self.logger = logger
        self.issues = []

        if self.benchmark_run.benchmark_type == BENCHMARK_TYPES.training:
            self.rules_checker = TrainingRulesChecker(self.benchmark_run, logger)
        elif self.benchmark_run.benchmark_type == BENCHMARK_TYPES.checkpointing:
            self.rules_checker = CheckpointingRulesChecker(self.benchmark_run, logger)

    def verify(self) -> PARAM_VALIDATION:
        self.logger.status(f"Verifying benchmark run for {self.benchmark_run.run_id}")
        self.issues = self.rules_checker.run_checks()
        self.benchmark_run.issues = self.issues
        num_invalid = 0
        num_open = 0
        num_closed = 0

        for issue in self.issues:
            if issue.validation == PARAM_VALIDATION.INVALID:
                self.logger.error(f"INVALID: {issue}")
                num_invalid += 1
            elif issue.validation == PARAM_VALIDATION.CLOSED:
                self.logger.status(f"Closed: {issue}")
                num_closed += 1
            elif issue.validation == PARAM_VALIDATION.OPEN:
                self.logger.status(f"Open: {issue}")
                num_open += 1
            else:
                raise ValueError(f"Unknown validation type: {issue.validation}")

        if num_invalid > 0:
            self.logger.status(f'Benchmark run is INVALID due to {num_invalid} issues ({self.benchmark_run.run_id})')
            return PARAM_VALIDATION.INVALID
        elif num_open > 0:
            self.logger.status(f'Benchmark run qualifies for OPEN category ({self.benchmark_run.run_id})')
            return PARAM_VALIDATION.OPEN
        else:
            self.logger.status(f'Benchmark run qualifies for CLOSED category ({self.benchmark_run.run_id})')
            return PARAM_VALIDATION.CLOSED


def calculate_training_data_size(args, cluster_information, dataset_params, reader_params, logger,
                                 num_processes=None) -> Tuple[int, int, int]:
    """
    Validate the parameters for the datasize operation and apply rules for a closed submission.

    Requirements:
      - Dataset needs to be 5x the amount of total memory
      - Training needs to do at least 500 steps per epoch

    Memory Ratio:
      - Collect "Total Memory" from /proc/meminfo on each host
      - sum it up
      - multiply by 5
      - divide by sample size
      - divide by batch size

    500 steps:
      - 500 steps per ecpoch
      - multiply by max number of processes
      - multiply by batch size

    If the number of files is greater than MAX_NUM_FILES_TRAIN, use the num_subfolders_train parameters to shard the
    dataset.
    :return:
    """
    required_file_count = 1
    required_subfolders_count = 0

    # Find the amount of memory in the cluster via args or measurements
    if not args:
        total_mem_bytes = cluster_information.total_memory_bytes
    elif args.client_host_memory_in_gb and args.num_client_hosts:
        # If host memory per client and num clients is provided, we use these values instead of the calculated memory
        per_host_memory_in_bytes = args.client_host_memory_in_gb * 1024 * 1024 * 1024
        num_hosts = args.num_client_hosts
        total_mem_bytes = per_host_memory_in_bytes * num_hosts
        num_processes = args.num_processes
    elif args.clienthost_host_memory_in_gb and not args.num_client_hosts:
        # If we have memory but not clients, we use the number of provided hosts and given memory amount
        per_host_memory_in_bytes = args.clienthost_host_memory_in_gb * 1024 * 1024 * 102
        num_hosts = len(args.hosts)
        total_mem_bytes = per_host_memory_in_bytes * num_hosts
        num_processes = args.num_processes
    else:
        raise ValueError('Either args or cluster_information is required')

    # Required Minimum Dataset size is 5x the total client memory
    dataset_size_bytes = 5 * total_mem_bytes
    file_size_bytes = dataset_params['num_samples_per_file'] * dataset_params['record_length_bytes']

    min_num_files_by_bytes = dataset_size_bytes // file_size_bytes
    num_samples_by_bytes = min_num_files_by_bytes * dataset_params['num_samples_per_file']
    min_samples = 500 * num_processes * reader_params['batch_size']
    min_num_files_by_samples = min_samples // dataset_params['num_samples_per_file']

    required_file_count = max(min_num_files_by_bytes, min_num_files_by_samples)
    total_disk_bytes = required_file_count * file_size_bytes

    logger.ridiculous(f'Required file count: {required_file_count}')
    logger.ridiculous(f'Required sample count: {min_samples}')
    logger.ridiculous(f'Min number of files by samples: {min_num_files_by_samples}')
    logger.ridiculous(f'Min number of files by size: {min_num_files_by_bytes}')
    logger.ridiculous(f'Required dataset size: {required_file_count * file_size_bytes / 1024 / 1024} MB')
    logger.ridiculous(f'Number of Samples by size: {num_samples_by_bytes}')

    if min_num_files_by_bytes > min_num_files_by_samples:
        logger.result(f'Minimum file count dictated by dataset size to memory size ratio.')
    else:
        logger.result(f'Minimum file count dictated by 500 step requirement of given accelerator count and batch size.')

    return int(required_file_count), int(required_subfolders_count), int(total_disk_bytes)


"""
The results directory structure is as follows:
results_dir:
    <benchmark_name>:
        <model>:
            <command>:
                    <datetime>:
                        run_<run_number> (Optional)
                    
This looks like:
results_dir:
    training:
        unet3d:
            datagen:
                <datetime>:
                    <output_files>
            run:
                <datetime>:
                    <output_files>
    checkpointing:
        llama3-8b:
            <datetime>:
                <output_files>
"""



def generate_output_location(benchmark, datetime_str=None, **kwargs):
    """
    Generate a standardized output location for benchmark results.

    Output structure follows this pattern:
    RESULTS_DIR:
        <benchmark_name>:
            <model>:
                <command>:
                        <datetime>:
                            run_<run_number> (Optional)

    Args:
        benchmark (Benchmark): benchmark (e.g., 'training', 'vectordb', 'checkpoint')
        datetime_str (str, optional): Datetime string for the run. If None, current datetime is used.
        **kwargs: Additional benchmark-specific parameters:
            - model (str): For training benchmarks, the model name (e.g., 'unet3d', 'resnet50')
            - category (str): For vectordb benchmarks, the category (e.g., 'throughput', 'latency')

    Returns:
        str: The full path to the output location
    """
    if datetime_str is None:
        datetime_str = DATETIME_STR

    output_location = benchmark.args.results_dir
    if hasattr(benchmark, "run_number"):
        run_number = benchmark.run_number
    else:
        run_number = 0

    # Handle different benchmark types
    if benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.training:
        if not hasattr(benchmark.args, "model"):
            raise ValueError("Model name is required for training benchmark output location")

        output_location = os.path.join(output_location, benchmark.BENCHMARK_TYPE.name)
        output_location = os.path.join(output_location, benchmark.args.model)
        output_location = os.path.join(output_location, benchmark.args.command)
        output_location = os.path.join(output_location, datetime_str)

    elif benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.vector_database:
        output_location = os.path.join(output_location, benchmark.BENCHMARK_TYPE.name)
        output_location = os.path.join(output_location, benchmark.args.command)
        output_location = os.path.join(output_location, datetime_str)

    elif benchmark.BENCHMARK_TYPE == BENCHMARK_TYPES.checkpointing:
        if not hasattr(benchmark.args, "model"):
            raise ValueError("Model name is required for training benchmark output location")

        output_location = os.path.join(output_location, benchmark.BENCHMARK_TYPE.name)
        output_location = os.path.join(output_location, benchmark.args.model)
        output_location = os.path.join(output_location, datetime_str)

    else:
        print(f'The given benchmark is not supported by mlpstorage.rules.generate_output_location()')
        sys.exit(1)

    return output_location


def get_runs_files(results_dir, logger=None):
    """
    Walk the results_dir location and return a list of BenchmarkResult objects that represent a single run

    :param results_dir: Base directory containing benchmark results
    :param benchmark_name: Optional filter for specific benchmark name
    :param command: Optional filter for specific command
    :return: List of dictionaries with run information
    """
    if logger is None:
        logger = setup_logging(name='mlpstorage.rules.get_runs_files')

    if not os.path.exists(results_dir):
        logger.warning(f'Results directory {results_dir} does not exist.')
        return []

    runs = []

    # Walk through all directories and files in results_dir
    for root, dirs, files in os.walk(results_dir):
        logger.ridiculous(f'Processing directory: {root}')

        # Look for metadata files
        metadata_files = [f for f in files if f.endswith('_metadata.json')]

        if not metadata_files:
            logger.debug(f'No metadata file found')
            continue
        else:
            logger.debug(f'Found metadata files in directory {root}: {metadata_files}')

        if len(metadata_files) > 1:
            logger.warning(f'Multiple metadata files found in directory {root}. Skipping this directory.')
            continue

        # Find DLIO summary.json file if it exists
        dlio_summary_file = None
        for f in files:
            if f == 'summary.json':
                dlio_summary_file = os.path.join(root, f)
                break

        if dlio_summary_file:
            runs.append(BenchmarkRun(benchmark_result=BenchmarkResult(root, logger), logger=logger))

    return runs
