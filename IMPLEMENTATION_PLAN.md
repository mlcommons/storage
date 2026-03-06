# Cluster Information Collection via MPI - Implementation Plan

## Overview

This plan outlines the implementation of MPI-based cluster information collection for the MLPerf Storage benchmark. The feature will collect system information (`/proc/meminfo`, `/proc/cpuinfo`, `/proc/diskstats`, and other relevant data) from all hosts in a distributed cluster using MPI, making it available for rules checking before and after benchmark execution.

## Current State Analysis

### Existing Infrastructure
- **MPI Support**: `generate_mpi_prefix_cmd()` in `utils.py` generates MPI command prefixes
- **Data Classes**: `HostMemoryInfo`, `HostCPUInfo`, `HostInfo`, `ClusterInformation` exist in `rules.py`
- **Parsing Support**: `HostMemoryInfo.from_proc_meminfo_dict()` already handles `/proc/meminfo` format
- **Metadata Storage**: `Benchmark.write_metadata()` serializes to JSON using `MLPSJsonEncoder`

### Current Limitations
- `DLIOBenchmark.accumulate_host_info()` uses only CLI arg `client_host_memory_in_gb` (single value for all hosts)
- No actual collection of real system information from hosts
- No disk statistics collection
- No validation that all nodes have consistent configurations

---

## Implementation Plan

### Phase 1: Create MPI Cluster Collector Module

**New File**: `mlpstorage/cluster_collector.py`

This module will handle all MPI-based data collection.

#### 1.1 Data Classes for Additional System Info

```python
@dataclass
class HostDiskInfo:
    """Disk statistics for a host from /proc/diskstats"""
    device_name: str
    reads_completed: int
    reads_merged: int
    sectors_read: int
    time_reading_ms: int
    writes_completed: int
    writes_merged: int
    sectors_written: int
    time_writing_ms: int
    ios_in_progress: int
    time_doing_ios_ms: int
    weighted_time_doing_ios_ms: int
    # Optional newer fields (kernel 4.18+)
    discards_completed: Optional[int] = None
    discards_merged: Optional[int] = None
    sectors_discarded: Optional[int] = None
    time_discarding_ms: Optional[int] = None
    flush_requests_completed: Optional[int] = None
    time_flushing_ms: Optional[int] = None

@dataclass
class HostNetworkInfo:
    """Network interface statistics from /proc/net/dev"""
    interface_name: str
    rx_bytes: int
    rx_packets: int
    rx_errors: int
    tx_bytes: int
    tx_packets: int
    tx_errors: int

@dataclass
class HostSystemInfo:
    """Extended system information for a host"""
    hostname: str
    kernel_version: str           # from /proc/version
    os_release: Dict[str, str]    # from /etc/os-release
    uptime_seconds: float         # from /proc/uptime
    load_average: Tuple[float, float, float]  # from /proc/loadavg
```

#### 1.2 Proc File Parsers

Create parsing functions for each `/proc` file:

```python
def parse_proc_meminfo(content: str) -> Dict[str, int]:
    """Parse /proc/meminfo content into a dictionary (values in kB)"""

def parse_proc_cpuinfo(content: str) -> List[Dict[str, Any]]:
    """Parse /proc/cpuinfo content into list of CPU dictionaries"""

def parse_proc_diskstats(content: str) -> List[HostDiskInfo]:
    """Parse /proc/diskstats content into list of disk info"""

def parse_proc_net_dev(content: str) -> List[HostNetworkInfo]:
    """Parse /proc/net/dev content into list of network info"""

def parse_proc_version(content: str) -> str:
    """Parse /proc/version to extract kernel version"""

def parse_proc_loadavg(content: str) -> Tuple[float, float, float]:
    """Parse /proc/loadavg to extract load averages"""
```

#### 1.3 Local Collection Function

```python
def collect_local_system_info() -> Dict[str, Any]:
    """
    Collect system information from the local node.

    Returns a dictionary containing:
    - hostname: str
    - meminfo: Dict from /proc/meminfo
    - cpuinfo: List[Dict] from /proc/cpuinfo
    - diskstats: List[Dict] from /proc/diskstats
    - netdev: List[Dict] from /proc/net/dev
    - version: str from /proc/version
    - loadavg: Tuple[float, float, float] from /proc/loadavg
    - uptime: float from /proc/uptime
    - os_release: Dict from /etc/os-release
    """
```

#### 1.4 MPI Collection Coordinator

```python
class MPIClusterCollector:
    """
    Collects system information from all nodes in a cluster using MPI.

    This class generates a Python script that will be executed via MPI
    on all nodes to collect and aggregate system information.
    """

    def __init__(self, hosts: List[str], mpi_bin: str, logger,
                 allow_run_as_root: bool = False,
                 timeout_seconds: int = 60):
        self.hosts = hosts
        self.mpi_bin = mpi_bin
        self.logger = logger
        self.allow_run_as_root = allow_run_as_root
        self.timeout = timeout_seconds

    def collect(self) -> Dict[str, Any]:
        """
        Execute MPI collection across all nodes.

        Returns:
            Dictionary mapping hostname -> system_info dict
        """

    def _generate_collector_script(self, output_path: str) -> str:
        """Generate the MPI collector Python script"""

    def _parse_collection_results(self, output_file: str) -> Dict[str, Any]:
        """Parse the JSON output from the MPI collection"""
```

#### 1.5 MPI Collection Script Template

The collector will generate and execute a Python script like:

```python
#!/usr/bin/env python3
"""MPI System Information Collector - Generated by MLPerf Storage"""
import json
import socket
import sys

def collect_local_info():
    info = {"hostname": socket.gethostname()}

    # Read /proc/meminfo
    try:
        with open("/proc/meminfo", "r") as f:
            info["meminfo"] = parse_meminfo(f.read())
    except Exception as e:
        info["meminfo_error"] = str(e)

    # Read /proc/cpuinfo
    try:
        with open("/proc/cpuinfo", "r") as f:
            info["cpuinfo"] = parse_cpuinfo(f.read())
    except Exception as e:
        info["cpuinfo_error"] = str(e)

    # Read /proc/diskstats
    try:
        with open("/proc/diskstats", "r") as f:
            info["diskstats"] = parse_diskstats(f.read())
    except Exception as e:
        info["diskstats_error"] = str(e)

    # Additional files...

    return info

def main():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Collect local info
    local_info = collect_local_info()

    # Gather all info to rank 0
    all_info = comm.gather(local_info, root=0)

    if rank == 0:
        # Write combined results to output file
        output = {info["hostname"]: info for info in all_info}
        with open(sys.argv[1], "w") as f:
            json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
```

---

### Phase 2: Extend Existing Data Classes

**File**: `mlpstorage/rules.py`

#### 2.1 Add Disk Info to HostInfo

```python
@dataclass
class HostInfo:
    """Information about a single host in the system"""
    hostname: str
    memory: HostMemoryInfo = field(default_factory=HostMemoryInfo)
    cpu: Optional[HostCPUInfo] = None
    disks: Optional[List[HostDiskInfo]] = None      # NEW
    network: Optional[List[HostNetworkInfo]] = None  # NEW
    system: Optional[HostSystemInfo] = None          # NEW
    collection_timestamp: Optional[str] = None       # NEW

    @classmethod
    def from_collected_data(cls, data: Dict[str, Any]) -> 'HostInfo':
        """Create HostInfo from MPI-collected data dictionary"""
```

#### 2.2 Extend ClusterInformation

```python
class ClusterInformation:
    def __init__(self, host_info_list: List[HostInfo], logger,
                 calculate_aggregated_info=True):
        # Existing attributes...
        self.total_memory_bytes = 0
        self.total_cores = 0

        # NEW aggregated attributes
        self.num_hosts = len(host_info_list)
        self.min_memory_bytes = 0
        self.max_memory_bytes = 0
        self.collection_method = "unknown"  # "mpi", "args", "dlio_summary"
        self.collection_timestamp = None
        self.host_consistency_issues = []  # List of detected inconsistencies

    @classmethod
    def from_mpi_collection(cls, collected_data: Dict[str, Any],
                            logger) -> 'ClusterInformation':
        """Create ClusterInformation from MPI collector output"""

    def validate_cluster_consistency(self) -> List[str]:
        """
        Check that all nodes have consistent configurations.
        Returns list of warning messages for any inconsistencies.
        """
```

---

### Phase 3: Integration with Benchmark Classes

**File**: `mlpstorage/benchmarks/base.py` and `mlpstorage/benchmarks/dlio.py`

#### 3.1 Add Collection to Base Benchmark

```python
class Benchmark(abc.ABC):
    def __init__(self, args, logger=None, run_datetime=None, run_number=0):
        # Existing init...

        # NEW: Collect cluster information before benchmark
        self.cluster_information = None
        if self._should_collect_cluster_info():
            self.cluster_information = self._collect_cluster_information()

    def _should_collect_cluster_info(self) -> bool:
        """Determine if we should collect cluster info"""
        return (hasattr(self.args, 'hosts') and
                self.args.hosts and
                len(self.args.hosts) > 0 and
                self.args.command not in ['datagen', 'configview'])

    def _collect_cluster_information(self) -> Optional[ClusterInformation]:
        """
        Collect cluster information using MPI if available,
        otherwise fall back to CLI args.
        """
        if self.args.exec_type == EXEC_TYPE.MPI:
            try:
                from mlpstorage.cluster_collector import MPIClusterCollector
                collector = MPIClusterCollector(
                    hosts=self.args.hosts,
                    mpi_bin=self.args.mpi_bin,
                    logger=self.logger,
                    allow_run_as_root=self.args.allow_run_as_root
                )
                return collector.collect()
            except Exception as e:
                self.logger.warning(f"MPI collection failed: {e}, falling back to args")

        # Fallback to existing behavior
        return self._collect_cluster_info_from_args()

    def _collect_cluster_info_from_args(self) -> ClusterInformation:
        """Collect cluster info from CLI arguments (existing behavior)"""
```

#### 3.2 Update DLIOBenchmark

```python
class DLIOBenchmark(Benchmark):
    def accumulate_host_info(self, args):
        """
        UPDATED: Use MPI-collected data if available,
        otherwise fall back to CLI args.
        """
        # If we already have collected cluster info, use it
        if hasattr(self, 'cluster_information') and self.cluster_information:
            return self.cluster_information

        # Existing fallback behavior...
```

---

### Phase 4: Metadata Storage

**File**: `mlpstorage/utils.py`

#### 4.1 Update JSON Encoder

```python
class MLPSJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        # Add handling for new dataclasses
        if isinstance(obj, (HostDiskInfo, HostNetworkInfo, HostSystemInfo)):
            return asdict(obj)
        # Existing handling...
```

#### 4.2 New Cluster Info File (Optional)

Consider writing detailed cluster info to a separate file:

```python
def write_cluster_info(self):
    """Write detailed cluster information to cluster_info.json"""
    cluster_info_path = os.path.join(
        self.run_result_output,
        f"{self.BENCHMARK_TYPE.value}_cluster_info.json"
    )
    with open(cluster_info_path, 'w') as f:
        json.dump(self.cluster_information.to_detailed_dict(), f, indent=2)
```

---

### Phase 5: Integration with BenchmarkResult

**File**: `mlpstorage/rules.py`

#### 5.1 Update BenchmarkResult

```python
class BenchmarkResult:
    def __init__(self, benchmark_result_root_dir, logger):
        # Existing init...
        self.cluster_info = None  # NEW

    def _process_result_directory(self):
        # Existing processing...

        # NEW: Load cluster info from dedicated file or metadata
        cluster_info_file = os.path.join(
            self.benchmark_result_root_dir,
            "*_cluster_info.json"
        )
        cluster_info_files = glob.glob(cluster_info_file)
        if cluster_info_files:
            with open(cluster_info_files[0], 'r') as f:
                self.cluster_info = json.load(f)
        elif self.metadata and 'cluster_information' in self.metadata:
            self.cluster_info = self.metadata['cluster_information']
```

#### 5.2 Update BenchmarkRun

```python
class BenchmarkRun:
    def __init__(self, benchmark_result=None, benchmark_instance=None, logger=None):
        # Existing init...

        if benchmark_result:
            # Load from files (post-execution)
            self._load_from_result(benchmark_result)
        elif benchmark_instance:
            # Use live instance (pre-execution)
            self._load_from_instance(benchmark_instance)

    def _load_from_result(self, benchmark_result):
        """Load from BenchmarkResult (post-execution verification)"""
        # Use dedicated cluster_info if available
        if benchmark_result.cluster_info:
            self.system_info = ClusterInformation.from_dict(
                benchmark_result.cluster_info,
                self.logger
            )
```

---

### Phase 6: Rules Checking Updates

**File**: `mlpstorage/rules.py`

#### 6.1 Add Cluster Validation Rules

```python
class ClusterValidationRulesChecker(RulesChecker):
    """Validates cluster configuration before benchmark execution"""

    def __init__(self, benchmark_run, logger):
        super().__init__(logger)
        self.benchmark_run = benchmark_run

    def check_cluster_consistency(self):
        """Verify all nodes have consistent configurations"""
        issues = self.benchmark_run.system_info.validate_cluster_consistency()
        for issue in issues:
            self.issues.append(Issue(
                validation=PARAM_VALIDATION.OPEN,
                message=issue,
                severity="warning"
            ))

    def check_minimum_memory(self):
        """Verify minimum memory requirements are met"""
        min_memory_gb = self.benchmark_run.system_info.min_memory_bytes / (1024**3)
        if min_memory_gb < MINIMUM_HOST_MEMORY_GB:
            self.issues.append(Issue(
                validation=PARAM_VALIDATION.INVALID,
                message=f"Host memory {min_memory_gb}GB below minimum {MINIMUM_HOST_MEMORY_GB}GB",
                parameter="host_memory",
                expected=f">= {MINIMUM_HOST_MEMORY_GB}GB",
                actual=f"{min_memory_gb}GB"
            ))

    def check_mpi_collection_success(self):
        """Verify MPI collection succeeded on all nodes"""
        if self.benchmark_run.system_info.collection_method != "mpi":
            self.issues.append(Issue(
                validation=PARAM_VALIDATION.OPEN,
                message="Cluster info collected via CLI args, not MPI",
                severity="info"
            ))
```

#### 6.2 Integrate with BenchmarkVerifier

```python
class BenchmarkVerifier:
    def verify(self) -> PARAM_VALIDATION:
        # Existing verification...

        # NEW: Add cluster validation for multi-host runs
        if len(self.benchmark_run.system_info.host_info_list) > 1:
            cluster_checker = ClusterValidationRulesChecker(
                self.benchmark_run,
                self.logger
            )
            cluster_issues = cluster_checker.run_checks()
            self.issues.extend(cluster_issues)
```

---

### Phase 7: CLI Updates

**File**: `mlpstorage/cli.py`

#### 7.1 Add Collection Options

```python
def add_cluster_collection_arguments(parser):
    """Add arguments for cluster information collection"""
    cluster_group = parser.add_argument_group("Cluster Information Collection")
    cluster_group.add_argument(
        "--collect-cluster-info",
        action="store_true",
        default=True,
        help="Collect detailed system information from all hosts via MPI"
    )
    cluster_group.add_argument(
        "--skip-cluster-collection",
        action="store_true",
        default=False,
        help="Skip MPI-based cluster collection, use CLI args only"
    )
    cluster_group.add_argument(
        "--cluster-collection-timeout",
        type=int,
        default=60,
        help="Timeout in seconds for MPI cluster collection"
    )
```

---

## File Structure Summary

```
mlpstorage/
├── cluster_collector.py      # NEW: MPI-based cluster collection
├── rules.py                  # MODIFIED: Extended data classes, new validators
├── benchmarks/
│   ├── base.py              # MODIFIED: Integration with collection
│   └── dlio.py              # MODIFIED: Use MPI-collected data
├── utils.py                  # MODIFIED: JSON encoder updates
├── cli.py                    # MODIFIED: New CLI arguments
└── main.py                   # MODIFIED: Collection orchestration
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Pre-Execution                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  CLI Args (hosts)  ──────────────────────┐                          │
│        │                                  │                          │
│        ▼                                  ▼                          │
│  ┌──────────────┐                  ┌──────────────┐                 │
│  │ MPI Collector │◄───fallback────│ Args-based   │                 │
│  │ (preferred)   │                 │ Collection   │                 │
│  └──────────────┘                  └──────────────┘                 │
│        │                                  │                          │
│        └─────────────┬───────────────────┘                          │
│                      ▼                                               │
│              ┌───────────────────┐                                  │
│              │ ClusterInformation│                                  │
│              └───────────────────┘                                  │
│                      │                                               │
│        ┌─────────────┼─────────────┐                                │
│        ▼             ▼             ▼                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                          │
│  │ Rules    │  │ Benchmark│  │ Metadata │                          │
│  │ Checking │  │ Instance │  │ Storage  │                          │
│  └──────────┘  └──────────┘  └──────────┘                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        Post-Execution                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Results Directory                                                   │
│        │                                                             │
│        ▼                                                             │
│  ┌──────────────────┐                                               │
│  │ BenchmarkResult  │◄─── Loads metadata.json + cluster_info.json   │
│  └──────────────────┘                                               │
│        │                                                             │
│        ▼                                                             │
│  ┌──────────────────┐                                               │
│  │ BenchmarkRun     │◄─── Reconstructs ClusterInformation           │
│  └──────────────────┘                                               │
│        │                                                             │
│        ▼                                                             │
│  ┌──────────────────┐                                               │
│  │ BenchmarkVerifier│◄─── Runs rules with full system info          │
│  └──────────────────┘                                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Order

1. **Phase 1.2**: Implement `/proc` file parsers (standalone, testable)
2. **Phase 1.1**: Implement new data classes (`HostDiskInfo`, etc.)
3. **Phase 2**: Extend existing data classes in `rules.py`
4. **Phase 1.3-1.5**: Implement `MPIClusterCollector`
5. **Phase 4**: Update JSON encoder and metadata storage
6. **Phase 3**: Integrate with Benchmark classes
7. **Phase 5**: Update `BenchmarkResult` and `BenchmarkRun`
8. **Phase 6**: Add cluster validation rules
9. **Phase 7**: Add CLI arguments
10. **Testing**: Unit tests for parsers, integration tests for MPI collection

---

## Testing Strategy

### Unit Tests
- Parser functions for each `/proc` file format
- Data class serialization/deserialization
- ClusterInformation aggregation and validation

### Integration Tests
- MPI collection on single node (mpirun -n 1)
- MPI collection on multiple hosts (requires test cluster)
- Fallback behavior when MPI fails
- End-to-end metadata storage and retrieval

### Test Files
```
mlpstorage/tests/
├── test_cluster_collector.py     # NEW
├── test_proc_parsers.py          # NEW
└── test_rules.py                 # MODIFIED
```

---

## Error Handling

1. **MPI Not Available**: Fall back to CLI args-based collection
2. **Partial Node Failure**: Log warning, continue with available data
3. **Timeout**: Log error, fall back to CLI args
4. **Permission Denied**: Handle gracefully, skip unavailable files
5. **Inconsistent Data**: Flag as warning, allow run to continue

---

## Backwards Compatibility

- Existing CLI args (`--client-host-memory-in-gb`) remain functional
- MPI collection is the default when hosts are specified
- `--skip-cluster-collection` flag allows users to use old behavior
- Metadata format remains compatible (additional fields are optional)
- Existing `BenchmarkResult` loading works with or without new fields
