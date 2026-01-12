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
        result = f"[{self.validation.value.upper()}] {self.message}"
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
class BenchmarkRunData:
    """
    Data contract for benchmark run information needed by RulesCheckers.

    This dataclass defines exactly what data is required for rules verification,
    regardless of whether the data comes from a live Benchmark instance or from
    result files on disk.
    """
    benchmark_type: BENCHMARK_TYPES
    model: Optional[str]
    command: Optional[str]
    run_datetime: str
    num_processes: int
    parameters: Dict[str, Any]
    override_parameters: Dict[str, Any]
    system_info: Optional['ClusterInformation'] = None
    metrics: Optional[Dict[str, Any]] = None
    result_dir: Optional[str] = None
    accelerator: Optional[str] = None


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


class BenchmarkInstanceExtractor:
    """
    Extracts BenchmarkRunData from a live Benchmark instance.

    Used for pre-execution verification where we have direct access to
    the Benchmark object.
    """

    @staticmethod
    def extract(benchmark) -> BenchmarkRunData:
        """
        Extract BenchmarkRunData from a Benchmark instance.

        Args:
            benchmark: A Benchmark instance (or subclass like DLIOBenchmark)

        Returns:
            BenchmarkRunData populated from the benchmark instance
        """
        # Get parameters - prefer combined_params if available
        if hasattr(benchmark, 'combined_params'):
            parameters = benchmark.combined_params
        else:
            parameters = {}

        # Get override parameters
        if hasattr(benchmark, 'params_dict'):
            override_parameters = benchmark.params_dict
        else:
            override_parameters = {}

        # Get system info
        system_info = None
        if hasattr(benchmark, 'cluster_information'):
            system_info = benchmark.cluster_information

        return BenchmarkRunData(
            benchmark_type=benchmark.BENCHMARK_TYPE,
            model=getattr(benchmark.args, 'model', None),
            command=getattr(benchmark.args, 'command', None),
            run_datetime=benchmark.run_datetime,
            num_processes=benchmark.args.num_processes,
            parameters=parameters,
            override_parameters=override_parameters,
            system_info=system_info,
            accelerator=getattr(benchmark.args, 'accelerator_type', None),
            result_dir=benchmark.run_result_output if hasattr(benchmark, 'run_result_output') else None,
            metrics=None,  # No metrics available pre-execution
        )


class DLIOResultParser:
    """
    Parses DLIO benchmark result files into BenchmarkRunData.

    This class handles DLIO-specific result file formats including:
    - summary.json: DLIO benchmark metrics and run info
    - Hydra config files: Configuration used for the run
    """

    def __init__(self, logger=None):
        self.logger = logger

    def parse(self, result_dir: str, metadata: Optional[Dict] = None) -> BenchmarkRunData:
        """
        Parse DLIO result files from a directory.

        Args:
            result_dir: Path to the result directory
            metadata: Optional pre-loaded metadata dict (from _metadata.json)

        Returns:
            BenchmarkRunData populated from the result files
        """
        summary = self._load_summary(result_dir)
        hydra_configs = self._load_hydra_configs(result_dir)

        if summary is None:
            raise ValueError(f"No summary.json found in {result_dir}")

        # Determine benchmark type and command from workflow
        hydra_workload_config = hydra_configs.get("config.yaml", {}).get("workload", {})
        hydra_workload_overrides = hydra_configs.get("overrides.yaml", [])
        hydra_workflow = hydra_workload_config.get("workflow", {})

        workflow = (
            hydra_workflow.get('generate_data', False),
            hydra_workflow.get('train', False),
            hydra_workflow.get('checkpoint', False),
        )

        # Determine benchmark type
        benchmark_type = None
        command = None
        accelerator = None

        if workflow[0] or workflow[1]:
            benchmark_type = BENCHMARK_TYPES.training
            workloads = [i for i in hydra_workload_overrides if i.startswith('workload=')]
            if workflow[1]:
                command = "run"
                if workloads:
                    accelerator = workloads[0].split('_')[1] if '_' in workloads[0] else None
            elif workflow[0]:
                command = "datagen"
        elif workflow[2]:
            benchmark_type = BENCHMARK_TYPES.checkpointing
            command = "run"

        # Extract model name and normalize it
        model = hydra_workload_config.get('model', {}).get("name")
        if model:
            model = model.replace("llama_", "llama3-")
            model = model.replace("_", "-")

        # Extract override parameters from hydra overrides
        override_parameters = {}
        for param in hydra_workload_overrides:
            if '=' in param:
                p, v = param.split('=', 1)
                if p.startswith('++workload.'):
                    override_parameters[p[len('++workload.'):]] = v

        # Build system info from summary
        system_info = ClusterInformation.from_dlio_summary_json(summary, self.logger)

        return BenchmarkRunData(
            benchmark_type=benchmark_type,
            model=model,
            command=command,
            run_datetime=summary.get("start", ""),
            num_processes=summary.get("num_accelerators", 0),
            parameters=hydra_workload_config,
            override_parameters=override_parameters,
            system_info=system_info,
            metrics=summary.get("metric"),
            result_dir=result_dir,
            accelerator=accelerator,
        )

    def _load_summary(self, result_dir: str) -> Optional[Dict]:
        """Load summary.json from result directory"""
        summary_path = os.path.join(result_dir, 'summary.json')
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to load summary from {summary_path}: {e}")
        return None

    def _load_hydra_configs(self, result_dir: str) -> Dict[str, Any]:
        """Load Hydra config files from result directory"""
        hydra_configs = {}
        hydra_dir = os.path.join(result_dir, HYDRA_OUTPUT_SUBDIR)

        if os.path.exists(hydra_dir) and os.path.isdir(hydra_dir):
            for config_file in os.listdir(hydra_dir):
                if config_file.endswith('.yaml'):
                    config_path = os.path.join(hydra_dir, config_file)
                    try:
                        with open(config_path, 'r') as f:
                            hydra_configs[config_file] = yaml.load(f, Loader=yaml.Loader)
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Failed to load Hydra config from {config_path}: {e}")

        return hydra_configs


class ResultFilesExtractor:
    """
    Extracts BenchmarkRunData from result files on disk.

    This extractor first attempts to load data from the metadata file
    (preferred, as it contains complete information). If the metadata
    is incomplete or missing, it falls back to using a tool-specific
    parser (e.g., DLIOResultParser).
    """

    def __init__(self, result_parser=None):
        """
        Initialize the extractor.

        Args:
            result_parser: A parser for tool-specific result files.
                          Defaults to DLIOResultParser if not provided.
        """
        self.result_parser = result_parser

    def extract(self, result_dir: str, logger=None) -> BenchmarkRunData:
        """
        Extract BenchmarkRunData from a result directory.

        Args:
            result_dir: Path to the result directory
            logger: Logger instance

        Returns:
            BenchmarkRunData populated from result files
        """
        # First, try to load from metadata file
        metadata = self._load_metadata(result_dir, logger)

        if metadata and self._is_complete_metadata(metadata):
            return self._from_metadata(metadata, result_dir)

        # Fall back to tool-specific parser
        if self.result_parser is None:
            self.result_parser = DLIOResultParser(logger=logger)

        return self.result_parser.parse(result_dir, metadata)

    def _load_metadata(self, result_dir: str, logger=None) -> Optional[Dict]:
        """Load metadata JSON file from result directory"""
        try:
            metadata_files = [f for f in os.listdir(result_dir) if f.endswith('_metadata.json')]
            if metadata_files:
                metadata_path = os.path.join(result_dir, metadata_files[0])
                with open(metadata_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            if logger:
                logger.debug(f"Could not load metadata from {result_dir}: {e}")
        return None

    def _is_complete_metadata(self, metadata: Dict) -> bool:
        """
        Check if metadata contains all required fields for BenchmarkRunData.

        Returns True if metadata can be used directly without falling back
        to tool-specific parsing.
        """
        required_fields = ['benchmark_type', 'run_datetime', 'num_processes', 'parameters']
        return all(field in metadata for field in required_fields)

    def _from_metadata(self, metadata: Dict, result_dir: str) -> BenchmarkRunData:
        """Create BenchmarkRunData from a complete metadata dict"""
        # Convert benchmark_type string to enum
        benchmark_type_str = metadata.get('benchmark_type', '')
        benchmark_type = None
        for bt in BENCHMARK_TYPES:
            if bt.name == benchmark_type_str or bt.value == benchmark_type_str:
                benchmark_type = bt
                break

        # Reconstruct system_info if present
        system_info = None
        if 'system_info' in metadata and metadata['system_info']:
            # This will need to be enhanced when we update Benchmark.metadata
            pass

        return BenchmarkRunData(
            benchmark_type=benchmark_type,
            model=metadata.get('model'),
            command=metadata.get('command'),
            run_datetime=metadata.get('run_datetime', ''),
            num_processes=metadata.get('num_processes', 0),
            parameters=metadata.get('parameters', {}),
            override_parameters=metadata.get('override_parameters', {}),
            system_info=system_info,
            metrics=metadata.get('metrics'),
            result_dir=result_dir,
            accelerator=metadata.get('accelerator'),
        )


class BenchmarkRun:
    """
    Represents a benchmark run with all parameters and system information.

    This class provides a unified interface for accessing benchmark run data
    regardless of whether it comes from a live Benchmark instance or from
    result files on disk.

    Preferred construction methods:
        - BenchmarkRun.from_benchmark(benchmark, logger) - from live instance
        - BenchmarkRun.from_result_dir(result_dir, logger) - from result files

    The legacy constructor (benchmark_result/benchmark_instance params) is
    maintained for backward compatibility but may be deprecated in future.
    """

    def __init__(self, data: BenchmarkRunData = None, logger=None,
                 # Legacy parameters for backward compatibility
                 benchmark_result=None, benchmark_instance=None):
        """
        Initialize a BenchmarkRun.

        Args:
            data: BenchmarkRunData instance (preferred)
            logger: Logger instance
            benchmark_result: (Legacy) BenchmarkResult instance
            benchmark_instance: (Legacy) Benchmark instance
        """
        self.logger = logger
        self._category = None
        self._issues = []

        # New path: direct initialization with BenchmarkRunData
        if data is not None:
            self._data = data
            self._run_id = RunID(
                program=data.benchmark_type.name if data.benchmark_type else "",
                command=data.command,
                model=data.model,
                run_datetime=data.run_datetime
            )
            if self.logger:
                self.logger.info(f"Created benchmark run: {self._run_id}")
            return

        # Legacy path: construct from benchmark_result or benchmark_instance
        if benchmark_result is None and benchmark_instance is None:
            if self.logger:
                self.logger.error("BenchmarkRun needs data, benchmark_result, or benchmark_instance")
            raise ValueError("Either data, benchmark_result, or benchmark_instance must be provided")

        if benchmark_result and benchmark_instance:
            if self.logger:
                self.logger.error("Both benchmark_result and benchmark_instance provided")
            raise ValueError("Only one of benchmark_result and benchmark_instance can be provided")

        # Use extractors to create BenchmarkRunData
        if benchmark_instance:
            self._data = BenchmarkInstanceExtractor.extract(benchmark_instance)
        elif benchmark_result:
            # Use DLIOResultParser via the result files
            parser = DLIOResultParser(logger=logger)
            self._data = parser.parse(benchmark_result.benchmark_result_root_dir)

        self._run_id = RunID(
            program=self._data.benchmark_type.name if self._data.benchmark_type else "",
            command=self._data.command,
            model=self._data.model,
            run_datetime=self._data.run_datetime
        )
        if self.logger:
            self.logger.info(f"Found benchmark run: {self._run_id}")

    @classmethod
    def from_benchmark(cls, benchmark, logger=None) -> 'BenchmarkRun':
        """
        Create a BenchmarkRun from a live Benchmark instance.

        Args:
            benchmark: A Benchmark instance (or subclass)
            logger: Logger instance

        Returns:
            BenchmarkRun instance
        """
        data = BenchmarkInstanceExtractor.extract(benchmark)
        return cls(data=data, logger=logger)

    @classmethod
    def from_result_dir(cls, result_dir: str, logger=None) -> 'BenchmarkRun':
        """
        Create a BenchmarkRun from result files on disk.

        Args:
            result_dir: Path to the result directory
            logger: Logger instance

        Returns:
            BenchmarkRun instance
        """
        extractor = ResultFilesExtractor()
        data = extractor.extract(result_dir, logger)
        return cls(data=data, logger=logger)

    # Property delegation to BenchmarkRunData
    @property
    def data(self) -> BenchmarkRunData:
        """Access the underlying BenchmarkRunData"""
        return self._data

    @property
    def benchmark_type(self):
        return self._data.benchmark_type

    @property
    def model(self):
        return self._data.model

    @property
    def command(self):
        return self._data.command

    @property
    def run_datetime(self):
        return self._data.run_datetime

    @property
    def num_processes(self):
        return self._data.num_processes

    @property
    def parameters(self):
        return self._data.parameters

    @property
    def override_parameters(self):
        return self._data.override_parameters

    @property
    def system_info(self):
        return self._data.system_info

    @property
    def metrics(self):
        return self._data.metrics

    @property
    def accelerator(self):
        return self._data.accelerator

    @property
    def result_dir(self):
        return self._data.result_dir

    # Verification state (not part of BenchmarkRunData)
    @property
    def issues(self):
        return self._issues

    @issues.setter
    def issues(self, issues):
        self._issues = issues

    @property
    def category(self):
        return self._category

    @category.setter
    def category(self, category):
        self._category = category

    @property
    def run_id(self):
        """Returns the RunID for this benchmark run"""
        return self._run_id

    @property
    def post_execution(self) -> bool:
        """Returns True if this run was loaded from result files (has metrics)"""
        return self._data.metrics is not None

    def as_dict(self):
        """Convert the BenchmarkRun object to a dictionary"""
        ret_dict = {
            "run_id": str(self.run_id),
            "benchmark_type": self.benchmark_type.name if self.benchmark_type else None,
            "model": self.model,
            "command": self.command,
            "num_processes": self.num_processes,
            "parameters": self.parameters,
            "system_info": self.system_info.as_dict() if self.system_info else None,
            "metrics": self.metrics,
        }
        if self.accelerator:
            ret_dict["accelerator"] = str(self.accelerator)

        return ret_dict


class RulesChecker(abc.ABC):
    """
    Base class for rule checkers that verify call the self.check_* methods
    """
    def __init__(self, logger, *args, **kwargs):
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


class RunRulesChecker(RulesChecker):
    """
    This class verifies rules against individual benchmark runs.
    """

    def __init__(self, benchmark_run, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.benchmark_run = benchmark_run


class MultiRunRulesChecker(RulesChecker):
    """Rules checker for multiple benchmark runs as for a single workload or for the full submission"""

    def __init__(self, benchmark_runs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if type(benchmark_runs) not in [list, tuple]:
            raise TypeError("benchmark_runs must be a list or tuple")
        self.benchmark_runs = benchmark_runs

    def check_runs_valid(self) -> Optional[Issue]:
        category_set = {run.category for run in self.benchmark_runs}
        if PARAM_VALIDATION.INVALID in category_set:
            return Issue(
                validation=PARAM_VALIDATION.INVALID,
                message="Invalid runs found.",
                parameter="category",
                expected="OPEN or CLOSED",
                actual=[cat.value.upper() for cat in category_set]
            )
        elif PARAM_VALIDATION.OPEN in category_set:
            return Issue(
                validation=PARAM_VALIDATION.OPEN,
                message="All runs satisfy the OPEN or CLOSED category",
                parameter="category",
                expected="OPEN or CLOSED",
                actual=[cat.value.upper() for cat in category_set]
            )
        elif {PARAM_VALIDATION.CLOSED} == category_set:
            return Issue(
                validation=PARAM_VALIDATION.CLOSED,
                message="All runs satisfy the CLOSED category",
                parameter="category",
                expected="OPEN or CLOSED",
                actual=[cat.value.upper() for cat in category_set]
            )
        return None


class TrainingRunRulesChecker(RunRulesChecker):
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
            if param.startswith("workflow"):
                # We handle workflow parameters separately
                continue
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

    def check_workflow_parameters(self) -> Optional[Issue]:
        issues = []
        # Check if the workflow parameters are valid
        workflow_params = self.benchmark_run.parameters.get('workflow', {})
        for param, value in workflow_params.items():
            if self.benchmark_run.model == UNET and self.benchmark_run.command == "run_benchmark":
                # Unet3d training requires the checkpoint workflow = True
                if param == "checkpoint":
                    if value == True:
                        return Issue(
                            validation=PARAM_VALIDATION.CLOSED ,
                            message="Unet3D training requires executing a checkpoing",
                            parameter="workflow.checkpoint",
                            expected="True",
                            actual=value
                        )
                    elif value == False:
                        return Issue(
                            validation=PARAM_VALIDATION.INVALID,
                            message="Unet3D training requires executing a checkpoint. The parameter 'workflow.checkpoint' is set to False",
                            parameter="workflow.checkpoint",
                            expected="True",
                            actual=value
                        )
        return None

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


class CheckpointingRunRulesChecker(RunRulesChecker):
    """Rules checker for checkpointing benchmarks"""
    def check_benchmark_type(self) -> Optional[Issue]:
        pass


#######################################################################################################################
# Define the checkers for groups of runs representing submissions
#######################################################################################################################

class CheckpointSubmissionRulesChecker(MultiRunRulesChecker):
    supported_models = LLM_MODELS

    def check_num_runs(self) -> Optional[Issue]:
        """
        Require 10 total writes and 10 total reads for checkpointing benchmarks.  It's possible for a submitter
         to have a single run with all checkpoints, two runs that separate reads and writes, or individual runs
         for each read and write operation.
        """
        issues = []
        num_writes = num_reads = 0
        for run in self.benchmark_runs:
            if run.benchmark_type == BENCHMARK_TYPES.checkpointing:
                num_writes += run.parameters.get('checkpoint', {}).get('num_checkpoints_write', 0)
                num_reads += run.parameters.get('checkpoint', {}).get('num_checkpoints_read', 0)

        if not num_reads == 10:
            issues.append(Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Expected 10 total read operations, but found {num_reads}",
                parameter="checkpoint.num_checkpoints_read",
                expected=10,
                actual=num_reads
            ))
        else:
            issues.append(Issue(
                validation=PARAM_VALIDATION.CLOSED,
                message=f"Found expected 10 total read operations",
                parameter="checkpoint.num_checkpoints_read",
                expected=10,
                actual=num_reads
            ))

        if not num_writes == 10:
            issues.append(Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Expected 10 total write operations, but found {num_writes}",
                parameter="checkpoint.num_checkpoints_write",
                expected=10,
                actual=num_writes
            ))
        else:
            issues.append(Issue(
                validation=PARAM_VALIDATION.CLOSED,
                message=f"Found expected 10 total write operations",
                parameter="checkpoint.num_checkpoints_write",
                expected=10,
                actual=num_writes
            ))

        if num_writes == 10 and num_reads == 10:
            issues.append(Issue(
                validation=PARAM_VALIDATION.CLOSED,
                message=f"Found expected 10 total read and write operations",
                parameter="checkpoint.num_checkpoints_read",
                expected=10,
                actual=10,
            ))

        return issues


class TrainingSubmissionRulesChecker(MultiRunRulesChecker):
    supported_models = MODELS

    def check_num_runs(self) -> Optional[Issue]:
        """
        Require 5 runs for training benchmarks
        """

class BenchmarkVerifier:
    """
    Verifies benchmark runs against submission rules.

    Accepts various input types:
        - BenchmarkRun instances (preferred)
        - Benchmark instances (live benchmarks, pre-execution)
        - str paths to result directories (post-execution)

    Usage:
        # Single run verification
        verifier = BenchmarkVerifier(benchmark_run, logger=logger)
        result = verifier.verify()

        # From a Benchmark instance
        verifier = BenchmarkVerifier(training_benchmark, logger=logger)

        # From a result directory
        verifier = BenchmarkVerifier("/path/to/results", logger=logger)

        # Multi-run verification
        verifier = BenchmarkVerifier(run1, run2, run3, logger=logger)
    """

    def __init__(self, *sources, logger=None):
        self.logger = logger
        self.issues = []

        if len(sources) == 0:
            raise ValueError("At least one source is required")

        # Convert all sources to BenchmarkRun instances
        self.benchmark_runs = []
        for source in sources:
            if isinstance(source, BenchmarkRun):
                self.benchmark_runs.append(source)
            elif isinstance(source, str):
                # Assume it's a result directory path
                self.benchmark_runs.append(BenchmarkRun.from_result_dir(source, logger))
            elif "mlpstorage.benchmarks." in str(type(source)):
                # It's a Benchmark instance - use the factory method
                self.benchmark_runs.append(BenchmarkRun.from_benchmark(source, logger))
            else:
                raise TypeError(f"Unsupported source type: {type(source)}. "
                               f"Expected BenchmarkRun, Benchmark instance, or result directory path.")

        # Determine mode
        if len(self.benchmark_runs) == 1:
            self.mode = "single"
        else:
            self.mode = "multi"

        # Create appropriate rules checker
        if self.mode == "single":
            benchmark_run = self.benchmark_runs[0]
            if benchmark_run.benchmark_type == BENCHMARK_TYPES.training:
                self.rules_checker = TrainingRunRulesChecker(benchmark_run, logger)
            elif benchmark_run.benchmark_type == BENCHMARK_TYPES.checkpointing:
                self.rules_checker = CheckpointingRunRulesChecker(benchmark_run, logger)
            else:
                raise ValueError(f"Unsupported benchmark type: {benchmark_run.benchmark_type}")

        elif self.mode == "multi":
            benchmark_types = {br.benchmark_type for br in self.benchmark_runs}
            if len(benchmark_types) > 1:
                raise ValueError(f"Multi-run verification requires all runs are from the same "
                               f"benchmark type. Got types: {benchmark_types}")
            benchmark_type = benchmark_types.pop()

            if benchmark_type == BENCHMARK_TYPES.training:
                self.rules_checker = TrainingSubmissionRulesChecker(self.benchmark_runs, logger)
            elif benchmark_type == BENCHMARK_TYPES.checkpointing:
                self.rules_checker = CheckpointSubmissionRulesChecker(self.benchmark_runs, logger)
            else:
                raise ValueError(f"Unsupported benchmark type for multi-run: {benchmark_type}")

    def verify(self) -> PARAM_VALIDATION:
        run_ids = [br.run_id for br in self.benchmark_runs]
        if self.mode == "single":
            self.logger.status(f"Verifying benchmark run for {run_ids[0]}")
        elif self.mode == "multi":
            self.logger.status(f"Verifying benchmark runs for {', '.join(str(rid) for rid in run_ids)}")
        self.issues = self.rules_checker.run_checks()
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

        if self.mode == "single":
            self.benchmark_runs[0].issues = self.issues

        if num_invalid > 0:
            self.logger.status(f'Benchmark run is INVALID due to {num_invalid} issues ({run_ids})')
            if self.mode == "single":
                self.benchmark_runs[0].category = PARAM_VALIDATION.INVALID
            return PARAM_VALIDATION.INVALID
        elif num_open > 0:
            if self.mode == "single":
                self.benchmark_runs[0].category = PARAM_VALIDATION.OPEN
            self.logger.status(f'Benchmark run qualifies for OPEN category ({run_ids})')
            return PARAM_VALIDATION.OPEN
        else:
            if self.mode == "single":
                self.benchmark_runs[0].category = PARAM_VALIDATION.CLOSED
            self.logger.status(f'Benchmark run qualifies for CLOSED category ({run_ids})')
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
        per_host_memory_in_bytes = args.clienthost_host_memory_in_gb * 1024 * 1024 * 1024
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


def get_runs_files(results_dir, logger=None) -> List[BenchmarkRun]:
    """
    Walk the results_dir location and return a list of BenchmarkRun objects.

    Scans the directory tree for benchmark result directories (identified by
    metadata files and summary.json) and creates BenchmarkRun instances for each.

    Args:
        results_dir: Base directory containing benchmark results
        logger: Logger instance

    Returns:
        List of BenchmarkRun instances
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

        # Check for summary.json (indicates a complete DLIO run)
        has_summary = 'summary.json' in files

        if has_summary:
            try:
                runs.append(BenchmarkRun.from_result_dir(root, logger))
            except Exception as e:
                logger.error(f'Failed to load benchmark run from {root}: {e}')
                continue

    return runs
