# Phase 6: SSH-Based Host Collection - Research

**Researched:** 2026-01-24
**Domain:** SSH-based remote system information collection for non-MPI benchmarks
**Confidence:** HIGH

## Summary

This phase implements SSH-based host collection to enable non-MPI benchmarks (KV Cache, VectorDB) to gather cluster information from remote hosts. The existing codebase has a robust foundation:

1. **MPI-based collection already exists** in `mlpstorage/cluster_collector.py` with comprehensive /proc file parsers
2. **SSH connectivity validation exists** in `mlpstorage/environment/validators.py` using subprocess + OpenSSH
3. **ClusterCollectorInterface** is defined in `mlpstorage/interfaces/collector.py` for dependency injection

The recommended approach is to create an `SSHClusterCollector` class that parallels `MPIClusterCollector`, reusing existing /proc parsing functions and integrating with the established `ClusterInformation` data model.

**Primary recommendation:** Implement SSH collection using subprocess + OpenSSH (not paramiko/fabric) for consistency with existing patterns, with concurrent.futures.ThreadPoolExecutor for parallel host collection.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| subprocess | stdlib | SSH command execution | Already used in validators.py; no new dependencies |
| concurrent.futures | stdlib | Parallel SSH execution | Thread-safe, built-in, simple API |
| dataclasses | stdlib | Data structures | Already used throughout codebase |
| socket | stdlib | Hostname resolution | Standard for localhost detection |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| shutil | stdlib | Binary discovery (`which`) | Check SSH availability |
| os | stdlib | File operations, environment | Script management |
| tempfile | stdlib | Temporary script storage | SSH script deployment |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| subprocess+SSH | paramiko | paramiko requires new dependency, password handling complexity |
| subprocess+SSH | fabric | fabric adds abstraction layer we don't need |
| subprocess+SSH | parallel-ssh | adds dependency; subprocess already proven in codebase |
| ThreadPoolExecutor | asyncio | async adds complexity; threads sufficient for I/O-bound SSH |

**Installation:**
```bash
# No additional packages required - all stdlib
pip install -e .
```

## Architecture Patterns

### Recommended Project Structure
```
mlpstorage/
  cluster_collector.py          # Add SSHClusterCollector alongside MPIClusterCollector
  interfaces/
    collector.py               # ClusterCollectorInterface (already exists)
  environment/
    validators.py              # validate_ssh_connectivity (already exists)
```

### Pattern 1: Collector Interface Implementation
**What:** SSHClusterCollector implements ClusterCollectorInterface
**When to use:** For all SSH-based collection operations
**Example:**
```python
# Source: Matches existing MPIClusterCollector pattern in cluster_collector.py
class SSHClusterCollector(ClusterCollectorInterface):
    """Collects system information from hosts using SSH."""

    def __init__(
        self,
        hosts: List[str],
        logger,
        ssh_username: Optional[str] = None,
        timeout_seconds: int = 60,
        max_workers: int = 10
    ):
        self.hosts = hosts
        self.logger = logger
        self.ssh_username = ssh_username
        self.timeout = timeout_seconds
        self.max_workers = max_workers

    def collect(self, hosts: List[str], timeout: int = 60) -> CollectionResult:
        """Collect info from all hosts in parallel."""
        pass

    def collect_local(self) -> CollectionResult:
        """Collect info from local host only."""
        pass

    def is_available(self) -> bool:
        """Check if SSH is available."""
        return shutil.which('ssh') is not None

    def get_collection_method(self) -> str:
        return "ssh"
```

### Pattern 2: Localhost Detection and Skip
**What:** Skip SSH for localhost/127.0.0.1, use direct local collection
**When to use:** Always, as optimization and to avoid SSH misconfiguration issues
**Example:**
```python
# Source: Existing pattern in mlpstorage/environment/validators.py
LOCALHOST_IDENTIFIERS = ('localhost', '127.0.0.1', '::1')

def _is_localhost(hostname: str) -> bool:
    """Check if hostname refers to local machine."""
    hostname_lower = hostname.lower()
    if hostname_lower in LOCALHOST_IDENTIFIERS:
        return True

    # Also check if hostname matches local hostname
    try:
        local_hostname = socket.gethostname()
        if hostname_lower == local_hostname.lower():
            return True
        # Check if IP resolves to localhost
        local_fqdn = socket.getfqdn()
        if hostname_lower == local_fqdn.lower():
            return True
    except Exception:
        pass

    return False
```

### Pattern 3: Parallel SSH Execution with ThreadPoolExecutor
**What:** Execute SSH commands concurrently with configurable parallelism
**When to use:** When collecting from multiple remote hosts
**Example:**
```python
# Source: Standard concurrent.futures pattern
from concurrent.futures import ThreadPoolExecutor, as_completed

def _collect_from_hosts_parallel(self) -> Dict[str, Any]:
    """Collect from all hosts in parallel using thread pool."""
    results = {}

    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        future_to_host = {
            executor.submit(self._collect_from_single_host, host): host
            for host in self._get_unique_hosts()
        }

        for future in as_completed(future_to_host):
            host = future_to_host[future]
            try:
                host_data = future.result()
                results[host] = host_data
            except Exception as e:
                self.logger.warning(f"Failed to collect from {host}: {e}")
                results[host] = {'error': str(e)}

    return results
```

### Pattern 4: SSH with BatchMode
**What:** Use SSH BatchMode to prevent password prompts in automation
**When to use:** Always for automated SSH execution
**Example:**
```python
# Source: Established pattern in environment/validators.py
def _build_ssh_command(
    self,
    hostname: str,
    remote_cmd: str
) -> List[str]:
    """Build SSH command with proper options."""
    cmd = [
        'ssh',
        '-o', 'BatchMode=yes',
        '-o', f'ConnectTimeout={self.timeout}',
        '-o', 'StrictHostKeyChecking=accept-new',
    ]

    if self.ssh_username:
        cmd.extend(['-l', self.ssh_username])

    cmd.extend([hostname, remote_cmd])
    return cmd
```

### Pattern 5: Start/End Collection Snapshots
**What:** Collect host information at benchmark start and end
**When to use:** For HOST-03 requirement (collection at start and end)
**Example:**
```python
# Source: New pattern based on existing base.py structure
class Benchmark:
    def _collect_cluster_snapshots(self) -> None:
        """Collect cluster info at start and end of run."""
        if self._should_collect_cluster_info():
            self.cluster_info_start = self._do_collection()

    def _finalize_cluster_snapshots(self) -> None:
        """Finalize collection after benchmark completion."""
        if hasattr(self, 'cluster_info_start'):
            self.cluster_info_end = self._do_collection()
            self.cluster_information = ClusterSnapshots(
                start=self.cluster_info_start,
                end=self.cluster_info_end
            )
```

### Anti-Patterns to Avoid
- **Using paramiko for simple commands:** subprocess + OpenSSH is already proven in this codebase and adds no dependencies
- **Blocking serial SSH execution:** Always use parallel execution for multiple hosts
- **Ignoring localhost optimization:** Direct local collection is faster and more reliable
- **Hardcoding SSH paths:** Use `shutil.which('ssh')` for portability
- **Ignoring timeout:** Always set ConnectTimeout and subprocess timeout

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| /proc parsing | Custom parsers | Existing functions in `cluster_collector.py` | `parse_proc_meminfo`, `parse_proc_cpuinfo`, etc. already tested |
| Local collection | New function | `collect_local_system_info()` | Already handles all /proc files with error handling |
| Localhost detection | Simple string check | Full resolution check | Must handle hostname, FQDN, IP variations |
| SSH availability | File existence check | `shutil.which('ssh')` | Handles PATH correctly |
| Data structures | New classes | Existing `HostInfo`, `ClusterInformation` | Already serializable, used throughout |
| Parallel execution | asyncio/threading | `ThreadPoolExecutor` | Simpler, sufficient for I/O-bound SSH |

**Key insight:** The existing `cluster_collector.py` already contains 90% of the needed functionality (all /proc parsing, data classes, local collection). SSH collection should reuse these, not duplicate.

## Common Pitfalls

### Pitfall 1: Duplicate /proc Parsers
**What goes wrong:** Creating new parsing functions for SSH output when they already exist
**Why it happens:** Not recognizing that MPI collector script uses same format as SSH would return
**How to avoid:** The embedded `MPI_COLLECTOR_SCRIPT` parses same /proc files - reuse those parsers
**Warning signs:** Writing code that looks similar to existing `parse_proc_*` functions

### Pitfall 2: SSH Password Prompts Hanging
**What goes wrong:** SSH waits for password input, causing subprocess to hang
**Why it happens:** BatchMode not set, or user doesn't have passwordless SSH configured
**How to avoid:** Always use `-o BatchMode=yes`; validate SSH connectivity before benchmark
**Warning signs:** Subprocess timeout errors, tests hanging

### Pitfall 3: Host:Slots Format Not Parsed
**What goes wrong:** SSH to "node1:4" instead of "node1"
**Why it happens:** MPI host specification format includes slot counts
**How to avoid:** Parse host entry: `hostname = host_entry.split(':')[0].strip()`
**Warning signs:** SSH connection failures to hosts with colons

### Pitfall 4: Missing Error Aggregation
**What goes wrong:** One SSH failure causes entire collection to fail
**Why it happens:** Not catching exceptions per-host in parallel execution
**How to avoid:** Catch exceptions in worker function, return error info in result
**Warning signs:** All-or-nothing collection results

### Pitfall 5: Localhost SSH When Not Configured
**What goes wrong:** Trying to SSH to localhost fails even though it's the local machine
**Why it happens:** SSH to localhost requires SSH server running and authorized_keys
**How to avoid:** Detect localhost and use direct local collection instead
**Warning signs:** SSH failures on localhost, tests failing on CI without SSH

### Pitfall 6: Inconsistent Data Format
**What goes wrong:** SSH-collected data doesn't match MPI-collected data structure
**Why it happens:** Different parsing approaches for same /proc files
**How to avoid:** Use same parsing functions; convert to same `HostInfo` structure
**Warning signs:** `ClusterInformation.from_ssh_collection` vs `from_mpi_collection` having different structures

## Code Examples

Verified patterns from existing codebase and official sources:

### SSH Connectivity Check (Existing)
```python
# Source: mlpstorage/environment/validators.py (lines 54-145)
def validate_ssh_connectivity(
    hosts: List[str],
    timeout: int = 5
) -> List[Tuple[str, bool, str]]:
    """Validate SSH connectivity to a list of remote hosts."""
    ssh_path = shutil.which('ssh')
    if ssh_path is None:
        raise ValidationIssue(...)

    results: List[Tuple[str, bool, str]] = []
    for host_entry in hosts:
        hostname = host_entry.split(':')[0].strip()
        if hostname.lower() in ('localhost', '127.0.0.1'):
            results.append((hostname, True, 'localhost (skipped)'))
            continue

        cmd = [
            'ssh',
            '-o', 'BatchMode=yes',
            '-o', f'ConnectTimeout={timeout}',
            '-o', 'StrictHostKeyChecking=accept-new',
            hostname,
            'echo', 'ok'
        ]
        # ... subprocess.run with timeout handling
```

### Local System Collection (Existing)
```python
# Source: mlpstorage/cluster_collector.py (lines 453-557)
def collect_local_system_info() -> Dict[str, Any]:
    """Collect system information from the local node."""
    result = {
        'hostname': socket.gethostname(),
        'collection_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'errors': {}
    }

    # Collect /proc/meminfo
    try:
        with open('/proc/meminfo', 'r') as f:
            result['meminfo'] = parse_proc_meminfo(f.read())
    except Exception as e:
        result['errors']['meminfo'] = str(e)
        result['meminfo'] = {}

    # ... similar for cpuinfo, diskstats, netdev, version, loadavg, uptime, os_release
```

### HostInfo from Collected Data (Existing)
```python
# Source: mlpstorage/rules/models.py (lines 199-247)
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
        cpu = HostCPUInfo(...)

    # ... disks, network, system
    return cls(hostname=hostname, memory=memory, cpu=cpu, ...)
```

### Proposed SSH Collection (New)
```python
# Pattern for SSH collection command - reads all /proc files
REMOTE_COLLECTION_SCRIPT = '''
import json, socket, time

def collect():
    result = {"hostname": socket.gethostname(), "errors": {}}

    files = [
        ("/proc/meminfo", "meminfo"),
        ("/proc/cpuinfo", "cpuinfo"),
        ("/proc/diskstats", "diskstats"),
        ("/proc/net/dev", "netdev"),
        ("/proc/version", "version"),
        ("/proc/loadavg", "loadavg"),
        ("/proc/uptime", "uptime"),
        ("/proc/vmstat", "vmstat"),
        ("/proc/mounts", "mounts"),
        ("/proc/cgroups", "cgroups"),
    ]

    for path, key in files:
        try:
            with open(path) as f:
                result[key] = f.read()
        except Exception as e:
            result["errors"][key] = str(e)
            result[key] = ""

    # /etc/os-release
    try:
        with open("/etc/os-release") as f:
            result["os_release_raw"] = f.read()
    except Exception as e:
        result["errors"]["os_release"] = str(e)

    print(json.dumps(result))

collect()
'''
```

## New Data Requirements (HOST-02)

Requirements specify collecting additional /proc data not currently collected:

### /proc/vmstat
**Format:** key-value pairs, one per line
**Parser needed:** Similar to meminfo parser
```python
def parse_proc_vmstat(content: str) -> Dict[str, int]:
    """Parse /proc/vmstat content into a dictionary."""
    result = {}
    for line in content.strip().split('\n'):
        parts = line.split()
        if len(parts) == 2:
            try:
                result[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return result
```

### /proc/mounts (filesystems)
**Format:** mount entries, space-separated fields
**Parser needed:** New parser for mount info
```python
@dataclass
class MountInfo:
    device: str
    mount_point: str
    fs_type: str
    options: str

def parse_proc_mounts(content: str) -> List[MountInfo]:
    """Parse /proc/mounts content."""
    mounts = []
    for line in content.strip().split('\n'):
        parts = line.split()
        if len(parts) >= 4:
            mounts.append(MountInfo(
                device=parts[0],
                mount_point=parts[1],
                fs_type=parts[2],
                options=parts[3]
            ))
    return mounts
```

### /proc/cgroups
**Format:** header line + data rows
**Parser needed:** New parser for cgroup info
```python
@dataclass
class CgroupInfo:
    subsys_name: str
    hierarchy: int
    num_cgroups: int
    enabled: bool

def parse_proc_cgroups(content: str) -> List[CgroupInfo]:
    """Parse /proc/cgroups content."""
    cgroups = []
    lines = content.strip().split('\n')
    for line in lines[1:]:  # Skip header
        if line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 4:
            cgroups.append(CgroupInfo(
                subsys_name=parts[0],
                hierarchy=int(parts[1]),
                num_cgroups=int(parts[2]),
                enabled=parts[3] == '1'
            ))
    return cgroups
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Paramiko for SSH | subprocess + OpenSSH | Established | No dependency, simpler |
| Serial SSH execution | Parallel with ThreadPoolExecutor | Python 3.2+ | 10x+ faster for many hosts |
| Manual host key handling | `StrictHostKeyChecking=accept-new` | OpenSSH 7.6+ | Safer than `no`, accepts new keys only |

**Deprecated/outdated:**
- `asyncio.AbstractEventLoop.run_in_executor` - use `ThreadPoolExecutor` directly for simpler code
- `paramiko` for simple command execution - subprocess + OpenSSH is sufficient

## Open Questions

Things that couldn't be fully resolved:

1. **SSH username handling**
   - What we know: SSH allows `-l username` option; current validators don't use it
   - What's unclear: Should benchmark accept `--ssh-username` CLI arg, or use system default?
   - Recommendation: Add optional `--ssh-username` flag, default to current user

2. **Collection timing within benchmark**
   - What we know: Requirements say "start and end" collection
   - What's unclear: Exact timing - before/after command generation, or before/after execution?
   - Recommendation: Collect immediately before `_run()` starts and after it completes

3. **Error handling strategy**
   - What we know: Some hosts may fail to connect
   - What's unclear: Should partial failure stop benchmark or continue?
   - Recommendation: Continue with partial collection, log warnings, include errors in metadata

## Sources

### Primary (HIGH confidence)
- `mlpstorage/cluster_collector.py` - Existing MPI collection implementation
- `mlpstorage/environment/validators.py` - Existing SSH validation with BatchMode
- `mlpstorage/interfaces/collector.py` - ClusterCollectorInterface definition
- `mlpstorage/rules/models.py` - HostInfo, ClusterInformation data models
- [proc(5) Linux manual page](https://man7.org/linux/man-pages/man5/proc.5.html) - /proc filesystem documentation

### Secondary (MEDIUM confidence)
- [Python concurrent.futures documentation](https://docs.python.org/3/library/concurrent.futures.html) - ThreadPoolExecutor API
- [Parallel-SSH documentation](https://parallel-ssh.readthedocs.io/en/latest/) - Parallel SSH patterns
- [PyProc library](https://github.com/cnamejj/PyProc) - /proc parsing reference

### Tertiary (LOW confidence)
- Web search results on SSH automation best practices

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All stdlib, already used in codebase
- Architecture: HIGH - Extends existing proven patterns
- Pitfalls: HIGH - Based on existing code analysis and test cases

**Research date:** 2026-01-24
**Valid until:** 2026-02-24 (30 days - stable domain)
