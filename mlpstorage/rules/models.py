"""
Data models for benchmark runs and validation.

This module contains all the data classes used for representing benchmark runs,
system information, and validation results. These classes are used throughout
the rules engine for validation and reporting.
"""

import json
import os
import yaml

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from mlpstorage.config import BENCHMARK_TYPES, PARAM_VALIDATION, HYDRA_OUTPUT_SUBDIR
from mlpstorage.cluster_collector import (
    HostDiskInfo,
    HostNetworkInfo,
    HostSystemInfo,
    summarize_cpuinfo,
)

if TYPE_CHECKING:
    from mlpstorage.rules.issues import Issue


@dataclass
class RunID:
    """Identifier for a benchmark run."""
    program: str
    command: str
    model: str
    run_datetime: str

    def __str__(self) -> str:
        id_str = self.program
        if self.command:
            id_str += f"_{self.command}"
        if self.model:
            id_str += f"_{self.model}"
        id_str += f"_{self.run_datetime}"
        return id_str


@dataclass
class ProcessedRun:
    """A processed benchmark run with validation results."""
    run_id: RunID
    benchmark_type: str
    run_parameters: Dict[str, Any]
    run_metrics: Dict[str, Any]
    issues: List['Issue'] = field(default_factory=list)

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
    """Detailed memory information for a host."""
    total: int = 0  # Total physical memory in bytes
    available: Optional[int] = None
    used: Optional[int] = None
    free: Optional[int] = None
    active: Optional[int] = None
    inactive: Optional[int] = None
    buffers: Optional[int] = None
    cached: Optional[int] = None
    shared: Optional[int] = None

    @classmethod
    def from_psutil_dict(cls, data: Dict[str, int]) -> 'HostMemoryInfo':
        """Create a HostMemoryInfo instance from a psutil dictionary."""
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
        """Create a HostMemoryInfo instance from a parsed /proc/meminfo dictionary."""
        def get_bytes(key: str, default: int = 0) -> int:
            val = data.get(key, default)
            if isinstance(val, (int, float)):
                return int(val) * 1024
            if isinstance(val, str):
                try:
                    return int(val.split()[0]) * 1024
                except (ValueError, IndexError):
                    return default
            return default

        return cls(
            total=get_bytes('MemTotal'),
            available=get_bytes('MemAvailable'),
            used=get_bytes('MemTotal') - get_bytes('MemFree') - get_bytes('Buffers') - get_bytes('Cached'),
            free=get_bytes('MemFree'),
            active=get_bytes('Active'),
            inactive=get_bytes('Inactive'),
            buffers=get_bytes('Buffers'),
            cached=get_bytes('Cached'),
            shared=get_bytes('Shmem'),
        )

    @classmethod
    def from_total_mem_int(cls, total_mem_int: int) -> 'HostMemoryInfo':
        """Create a HostMemoryInfo instance from total memory in bytes."""
        return cls(total=total_mem_int)


@dataclass
class HostCPUInfo:
    """CPU information for a host."""
    num_cores: int = 0
    num_logical_cores: int = 0
    model: str = ""
    architecture: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HostCPUInfo':
        """Create a HostCPUInfo instance from a dictionary."""
        return cls(
            num_cores=data.get('num_cores', 0),
            num_logical_cores=data.get('num_logical_cores', 0),
            model=data.get('model', ""),
            architecture=data.get('architecture', ""),
        )


@dataclass
class HostInfo:
    """Information about a single host in the system."""
    hostname: str
    memory: HostMemoryInfo = field(default_factory=HostMemoryInfo)
    cpu: Optional[HostCPUInfo] = None
    disks: Optional[List[HostDiskInfo]] = None
    network: Optional[List[HostNetworkInfo]] = None
    system: Optional[HostSystemInfo] = None
    collection_timestamp: Optional[str] = None

    @classmethod
    def from_dict(cls, hostname: str, data: Dict[str, Any]) -> 'HostInfo':
        """Create a HostInfo instance from a dictionary."""
        memory_info = data.get('memory_info', {})
        cpu_info = data.get('cpu_info', {})

        if isinstance(memory_info, dict):
            if 'total' in memory_info and isinstance(memory_info['total'], int):
                memory = HostMemoryInfo.from_psutil_dict(memory_info)
            elif 'MemTotal' in memory_info:
                memory = HostMemoryInfo.from_proc_meminfo_dict(memory_info)
            else:
                memory = HostMemoryInfo()
        else:
            memory = HostMemoryInfo()

        cpu = HostCPUInfo.from_dict(cpu_info) if cpu_info else None

        return cls(hostname=hostname, memory=memory, cpu=cpu)

    @classmethod
    def from_collected_data(cls, data: Dict[str, Any]) -> 'HostInfo':
        """Create a HostInfo instance from MPI-collected data."""
        hostname = data.get('hostname', 'unknown')

        meminfo = data.get('meminfo', {})
        if meminfo:
            memory = HostMemoryInfo.from_proc_meminfo_dict(meminfo)
        else:
            memory = HostMemoryInfo()

        cpuinfo = data.get('cpuinfo', [])
        cpu = None
        if cpuinfo:
            cpu_summary = summarize_cpuinfo(cpuinfo)
            cpu = HostCPUInfo(
                num_cores=cpu_summary.get('num_physical_cores', 0),
                num_logical_cores=cpu_summary.get('num_logical_cores', 0),
                model=cpu_summary.get('model', ''),
                architecture=cpu_summary.get('architecture', ''),
            )

        diskstats = data.get('diskstats', [])
        disks = [HostDiskInfo.from_dict(d) for d in diskstats] if diskstats else None

        netdev = data.get('netdev', [])
        network = [HostNetworkInfo.from_dict(n) for n in netdev] if netdev else None

        loadavg = data.get('loadavg', {})
        system = HostSystemInfo(
            hostname=hostname,
            kernel_version=data.get('version', ''),
            os_release=data.get('os_release', {}),
            uptime_seconds=data.get('uptime_seconds', 0.0),
            load_average_1min=loadavg.get('load_1min', 0.0),
            load_average_5min=loadavg.get('load_5min', 0.0),
            load_average_15min=loadavg.get('load_15min', 0.0),
            running_processes=loadavg.get('running_processes', 0),
            total_processes=loadavg.get('total_processes', 0),
        )

        return cls(
            hostname=hostname,
            memory=memory,
            cpu=cpu,
            disks=disks,
            network=network,
            system=system,
            collection_timestamp=data.get('collection_timestamp'),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert HostInfo to a dictionary for JSON serialization."""
        result = {
            'hostname': self.hostname,
            'memory': {
                'total': self.memory.total,
                'available': self.memory.available,
                'used': self.memory.used,
                'free': self.memory.free,
                'active': self.memory.active,
                'inactive': self.memory.inactive,
                'buffers': self.memory.buffers,
                'cached': self.memory.cached,
                'shared': self.memory.shared,
            },
            'collection_timestamp': self.collection_timestamp,
        }

        if self.cpu:
            result['cpu'] = {
                'num_cores': self.cpu.num_cores,
                'num_logical_cores': self.cpu.num_logical_cores,
                'model': self.cpu.model,
                'architecture': self.cpu.architecture,
            }

        if self.disks:
            result['disks'] = [d.to_dict() for d in self.disks]

        if self.network:
            result['network'] = [n.to_dict() for n in self.network]

        if self.system:
            result['system'] = self.system.to_dict()

        return result


class ClusterInformation:
    """Comprehensive system information for all hosts in the benchmark environment."""

    def __init__(self, host_info_list: List[HostInfo], logger, calculate_aggregated_info=True):
        self.logger = logger
        self.host_info_list = host_info_list
        self.total_memory_bytes = 0
        self.total_cores = 0
        self.num_hosts = len(host_info_list)
        self.min_memory_bytes = 0
        self.max_memory_bytes = 0
        self.collection_method = "unknown"
        self.collection_timestamp = None
        self.host_consistency_issues: List[str] = []

        if calculate_aggregated_info:
            self.calculate_aggregated_info()

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "total_memory_bytes": self.total_memory_bytes,
            "total_cores": self.total_cores,
            "num_hosts": self.num_hosts,
            "min_memory_bytes": self.min_memory_bytes,
            "max_memory_bytes": self.max_memory_bytes,
            "collection_method": self.collection_method,
            "collection_timestamp": self.collection_timestamp,
        }

        if self.host_info_list:
            result["hosts"] = [h.to_dict() if hasattr(h, 'to_dict') else str(h)
                               for h in self.host_info_list]

        if self.host_consistency_issues:
            result["host_consistency_issues"] = self.host_consistency_issues

        return result

    def to_detailed_dict(self) -> Dict[str, Any]:
        """Convert to detailed dictionary including all host information."""
        return self.as_dict()

    @property
    def info(self):
        """Property used by MLPSJsonEncoder for serialization."""
        return self.as_dict()

    def calculate_aggregated_info(self):
        """Calculate aggregated system information across all hosts."""
        if not self.host_info_list:
            return

        memory_values = []
        for host_info in self.host_info_list:
            if hasattr(host_info, 'memory') and host_info.memory:
                self.total_memory_bytes += host_info.memory.total
                memory_values.append(host_info.memory.total)
            if hasattr(host_info, 'cpu') and host_info.cpu:
                self.total_cores += host_info.cpu.num_cores

        if memory_values:
            self.min_memory_bytes = min(memory_values)
            self.max_memory_bytes = max(memory_values)

    def validate_cluster_consistency(self) -> List[str]:
        """Check that all nodes have consistent configurations."""
        issues = []

        if len(self.host_info_list) < 2:
            return issues

        memory_values = [h.memory.total for h in self.host_info_list
                        if hasattr(h, 'memory') and h.memory and h.memory.total > 0]
        if memory_values:
            min_mem = min(memory_values)
            max_mem = max(memory_values)
            if min_mem > 0:
                variance = (max_mem - min_mem) / min_mem
                if variance > 0.1:
                    issues.append(
                        f"Memory variance across hosts: {variance:.1%} "
                        f"(min: {min_mem / (1024**3):.1f}GB, max: {max_mem / (1024**3):.1f}GB)"
                    )

        core_counts = [h.cpu.num_cores for h in self.host_info_list
                      if hasattr(h, 'cpu') and h.cpu and h.cpu.num_cores > 0]
        if core_counts and len(set(core_counts)) > 1:
            issues.append(f"CPU core count varies across hosts: {sorted(set(core_counts))}")

        os_versions = []
        for h in self.host_info_list:
            if hasattr(h, 'system') and h.system and h.system.os_release:
                version = h.system.os_release.get('VERSION_ID', '')
                if version:
                    os_versions.append(version)
        if os_versions and len(set(os_versions)) > 1:
            issues.append(f"OS version varies across hosts: {sorted(set(os_versions))}")

        kernel_versions = []
        for h in self.host_info_list:
            if hasattr(h, 'system') and h.system and h.system.kernel_version:
                kv = h.system.kernel_version.split()[2] if len(h.system.kernel_version.split()) > 2 else h.system.kernel_version
                kernel_versions.append(kv)
        if kernel_versions and len(set(kernel_versions)) > 1:
            issues.append(f"Kernel version varies across hosts: {sorted(set(kernel_versions))}")

        self.host_consistency_issues = issues
        return issues

    @classmethod
    def from_mpi_collection(cls, collected_data: Dict[str, Any], logger) -> 'ClusterInformation':
        """Create ClusterInformation from MPI collector output."""
        host_info_list = []

        metadata = collected_data.pop('_metadata', {})
        collection_method = metadata.get('collection_method', 'mpi')
        collection_timestamp = metadata.get('collection_timestamp')

        for hostname, host_data in collected_data.items():
            if hostname.startswith('_'):
                continue
            host_info = HostInfo.from_collected_data(host_data)
            host_info_list.append(host_info)

        collected_data['_metadata'] = metadata

        inst = cls(host_info_list, logger, calculate_aggregated_info=True)
        inst.collection_method = collection_method
        inst.collection_timestamp = collection_timestamp
        inst.validate_cluster_consistency()

        return inst

    @classmethod
    def from_dlio_summary_json(cls, summary, logger) -> Optional['ClusterInformation']:
        """Create ClusterInformation from DLIO summary.json data."""
        host_memories = summary.get("host_memory_GB")
        host_cpus = summary.get("host_cpu_count")
        if host_memories is None or host_cpus is None:
            return None

        host_info_list = []
        num_hosts = len(host_memories)
        for i in range(num_hosts):
            memory_bytes = int(host_memories[i] * 1024 * 1024 * 1024)
            host_info = HostInfo(
                hostname=f"host_{i}",
                memory=HostMemoryInfo.from_total_mem_int(memory_bytes),
                cpu=HostCPUInfo(num_cores=host_cpus[i]) if i < len(host_cpus) else None,
            )
            host_info_list.append(host_info)

        inst = cls(host_info_list, logger, calculate_aggregated_info=True)
        inst.collection_method = "dlio_summary"
        return inst

    @classmethod
    def from_dict(cls, data: dict, logger) -> Optional['ClusterInformation']:
        """Create ClusterInformation from a dictionary."""
        if data is None:
            return None

        total_memory_bytes = data.get("total_memory_bytes")
        if total_memory_bytes is None:
            return None

        host_info_list = []
        hosts_data = data.get("hosts", [])
        for host_data in hosts_data:
            if isinstance(host_data, dict):
                host_info = HostInfo.from_collected_data(host_data)
                host_info_list.append(host_info)

        inst = cls(host_info_list, logger, calculate_aggregated_info=False)
        inst.total_memory_bytes = total_memory_bytes
        inst.total_cores = data.get("total_cores", 0)
        inst.num_hosts = data.get("num_hosts", len(host_info_list))
        inst.min_memory_bytes = data.get("min_memory_bytes", 0)
        inst.max_memory_bytes = data.get("max_memory_bytes", 0)
        inst.collection_method = data.get("collection_method", "unknown")
        inst.collection_timestamp = data.get("collection_timestamp")
        inst.host_consistency_issues = data.get("host_consistency_issues", [])

        return inst


class BenchmarkResult:
    """Represents the result files from a benchmark run."""

    def __init__(self, benchmark_result_root_dir, logger):
        self.benchmark_result_root_dir = benchmark_result_root_dir
        self.logger = logger
        self.metadata = None
        self.summary = None
        self.hydra_configs = {}
        self.issues = []
        self._process_result_directory()

    def _process_result_directory(self):
        """Process the result directory to extract metadata and metrics."""
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

        summary_path = os.path.join(self.benchmark_result_root_dir, 'summary.json')
        self.logger.debug(f'Looking for DLIO summary at {summary_path}...')
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    self.summary = json.load(f)
                self.logger.verbose(f"Loaded DLIO summary from {summary_path}")
            except Exception as e:
                self.logger.error(f"Failed to load DLIO summary from {summary_path}: {e}")

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
    """Extracts BenchmarkRunData from a live Benchmark instance."""

    @staticmethod
    def extract(benchmark) -> BenchmarkRunData:
        """Extract BenchmarkRunData from a Benchmark instance."""
        if hasattr(benchmark, 'combined_params'):
            parameters = benchmark.combined_params
        else:
            parameters = {}

        if hasattr(benchmark, 'params_dict'):
            override_parameters = benchmark.params_dict
        else:
            override_parameters = {}

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
            metrics=None,
        )


class DLIOResultParser:
    """Parses DLIO benchmark result files into BenchmarkRunData."""

    def __init__(self, logger=None):
        self.logger = logger

    def parse(self, result_dir: str, metadata: Optional[Dict] = None) -> BenchmarkRunData:
        """Parse DLIO result files from a directory."""
        summary = self._load_summary(result_dir)
        hydra_configs = self._load_hydra_configs(result_dir)

        if summary is None:
            raise ValueError(f"No summary.json found in {result_dir}")

        hydra_workload_config = hydra_configs.get("config.yaml", {}).get("workload", {})
        hydra_workload_overrides = hydra_configs.get("overrides.yaml", [])
        hydra_workflow = hydra_workload_config.get("workflow", {})

        workflow = (
            hydra_workflow.get('generate_data', False),
            hydra_workflow.get('train', False),
            hydra_workflow.get('checkpoint', False),
        )

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

        model = hydra_workload_config.get('model', {}).get("name")
        if model:
            model = model.replace("llama_", "llama3-")
            model = model.replace("_", "-")

        override_parameters = {}
        for param in hydra_workload_overrides:
            if '=' in param:
                p, v = param.split('=', 1)
                if p.startswith('++workload.'):
                    override_parameters[p[len('++workload.'):]] = v

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
        """Load summary.json from result directory."""
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
        """Load Hydra config files from result directory."""
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
    """Extracts BenchmarkRunData from result files on disk."""

    def __init__(self, result_parser=None):
        self.result_parser = result_parser

    def extract(self, result_dir: str, logger=None) -> BenchmarkRunData:
        """Extract BenchmarkRunData from a result directory."""
        metadata = self._load_metadata(result_dir, logger)

        if metadata and self._is_complete_metadata(metadata):
            return self._from_metadata(metadata, result_dir)

        if self.result_parser is None:
            self.result_parser = DLIOResultParser(logger=logger)

        return self.result_parser.parse(result_dir, metadata)

    def _load_metadata(self, result_dir: str, logger=None) -> Optional[Dict]:
        """Load metadata JSON file from result directory."""
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
        """Check if metadata contains all required fields for BenchmarkRunData."""
        required_fields = ['benchmark_type', 'run_datetime', 'num_processes', 'parameters']
        return all(field in metadata for field in required_fields)

    def _from_metadata(self, metadata: Dict, result_dir: str) -> BenchmarkRunData:
        """Create BenchmarkRunData from a complete metadata dict."""
        benchmark_type_str = metadata.get('benchmark_type', '')
        benchmark_type = None
        for bt in BENCHMARK_TYPES:
            if bt.name == benchmark_type_str or bt.value == benchmark_type_str:
                benchmark_type = bt
                break

        system_info = None
        if 'system_info' in metadata and metadata['system_info']:
            pass  # TODO: Reconstruct system_info

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
    """Represents a benchmark run with all parameters and system information."""

    def __init__(self, data: BenchmarkRunData = None, logger=None,
                 benchmark_result=None, benchmark_instance=None):
        self.logger = logger
        self._category = None
        self._issues = []

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

        if benchmark_result is None and benchmark_instance is None:
            if self.logger:
                self.logger.error("BenchmarkRun needs data, benchmark_result, or benchmark_instance")
            raise ValueError("Either data, benchmark_result, or benchmark_instance must be provided")

        if benchmark_result and benchmark_instance:
            if self.logger:
                self.logger.error("Both benchmark_result and benchmark_instance provided")
            raise ValueError("Only one of benchmark_result and benchmark_instance can be provided")

        if benchmark_instance:
            self._data = BenchmarkInstanceExtractor.extract(benchmark_instance)
        elif benchmark_result:
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
        """Create a BenchmarkRun from a live Benchmark instance."""
        data = BenchmarkInstanceExtractor.extract(benchmark)
        return cls(data=data, logger=logger)

    @classmethod
    def from_result_dir(cls, result_dir: str, logger=None) -> 'BenchmarkRun':
        """Create a BenchmarkRun from result files on disk."""
        extractor = ResultFilesExtractor()
        data = extractor.extract(result_dir, logger)
        return cls(data=data, logger=logger)

    @classmethod
    def from_data(cls, data: BenchmarkRunData, logger=None) -> 'BenchmarkRun':
        """Create a BenchmarkRun from BenchmarkRunData."""
        return cls(data=data, logger=logger)

    @property
    def data(self) -> BenchmarkRunData:
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
        return self._run_id

    @property
    def post_execution(self) -> bool:
        return self._data.metrics is not None

    def as_dict(self):
        """Convert the BenchmarkRun object to a dictionary."""
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
