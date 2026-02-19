# Phase 7: Time-Series Host Data Collection - Research

**Researched:** 2026-01-24
**Domain:** Background time-series system metrics collection during benchmark execution
**Confidence:** HIGH

## Summary

This phase implements continuous time-series host data collection during benchmark execution. The goal is to capture system metrics (diskstats, vmstat, loadavg, network stats) at regular 10-second intervals throughout the benchmark run, without impacting benchmark performance.

Key findings:

1. **Existing infrastructure is comprehensive** - Phase 6 delivered all the /proc parsers (vmstat, mounts, cgroups, diskstats, meminfo, etc.) and the SSHClusterCollector with parallel collection. The time-series collector can reuse these parsers entirely.

2. **Threading is the correct approach** - Reading /proc files is I/O-bound, not CPU-bound. Python's threading module is ideal for this use case. The GIL does not impact I/O-bound operations, and thread overhead is lower than process overhead.

3. **Non-daemon threads with Event signaling** - The recommended pattern for background tasks that need graceful shutdown. Using `threading.Event` with `wait(timeout=interval)` provides both periodic execution and responsive shutdown.

**Primary recommendation:** Implement a `TimeSeriesCollector` class using a non-daemon background thread with `threading.Event` for signaling. Reuse existing `/proc` parsers from `cluster_collector.py`. Store samples in memory and write to JSON file on completion.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| threading | stdlib | Background collection thread | I/O-bound task, lower overhead than multiprocessing |
| threading.Event | stdlib | Thread signaling for graceful stop | Python-recommended pattern for background thread control |
| time | stdlib | Timestamps and interval calculation | Standard for timing operations |
| dataclasses | stdlib | Data structures for samples | Already used throughout codebase |
| json | stdlib | Output serialization | Already used for metadata output |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| concurrent.futures.ThreadPoolExecutor | stdlib | Parallel multi-host collection | When collecting from multiple hosts simultaneously |
| socket | stdlib | Hostname identification | Already used in cluster_collector.py |
| os | stdlib | File operations | Writing output files |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| threading | multiprocessing | multiprocessing has higher overhead for I/O-bound tasks, no advantage for /proc reading |
| threading.Event | threading.Timer | Timer requires repeated scheduling; Event with wait(timeout) is cleaner for periodic tasks |
| threading | asyncio | asyncio adds complexity; blocking /proc reads don't benefit from async |
| In-memory buffer | Direct file writes | Frequent file I/O during benchmark could impact performance; batch write at end is safer |

**Installation:**
```bash
# No additional packages required - all stdlib
pip install -e .
```

## Architecture Patterns

### Recommended Project Structure
```
mlpstorage/
  cluster_collector.py          # Add TimeSeriesCollector class
  interfaces/
    collector.py                # TimeSeriesCollectorInterface (new)
  rules/
    models.py                   # TimeSeriesSample, TimeSeriesData dataclasses
  benchmarks/
    base.py                     # Integration with time-series collection
```

### Pattern 1: Background Thread with Event Signaling
**What:** Non-daemon thread using Event for graceful shutdown
**When to use:** Any long-running background task that needs clean termination
**Example:**
```python
# Source: Python threading documentation (https://docs.python.org/3/library/threading.html)
import threading
import time

class TimeSeriesCollector:
    """Collects time-series metrics in background thread."""

    def __init__(self, interval_seconds: float = 10.0):
        self.interval = interval_seconds
        self._stop_event = threading.Event()
        self._samples: List[Dict[str, Any]] = []
        self._thread = threading.Thread(
            target=self._collection_loop,
            daemon=False,  # Non-daemon for graceful shutdown
            name="TimeSeriesCollector"
        )

    def _collection_loop(self):
        """Run periodic collection until stop signal."""
        while not self._stop_event.is_set():
            sample = self._collect_sample()
            self._samples.append(sample)
            # Use wait(timeout) instead of sleep() for quick stop response
            self._stop_event.wait(timeout=self.interval)

    def start(self):
        """Start background collection."""
        self._thread.start()

    def stop(self) -> List[Dict[str, Any]]:
        """Stop collection and return all samples."""
        self._stop_event.set()
        self._thread.join(timeout=self.interval + 5)
        return self._samples
```

### Pattern 2: Lightweight /proc Collection for Time-Series
**What:** Collect only dynamic metrics that change over time (not static info)
**When to use:** Time-series collection where we want minimal overhead
**Example:**
```python
# Source: Existing pattern in mlpstorage/cluster_collector.py
def collect_timeseries_sample() -> Dict[str, Any]:
    """Collect time-varying metrics only - lightweight for periodic collection."""
    sample = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'hostname': socket.gethostname(),
    }

    # Dynamic metrics only - skip static info like cpuinfo, os_release
    try:
        with open('/proc/diskstats', 'r') as f:
            sample['diskstats'] = parse_proc_diskstats(f.read())
    except Exception:
        pass

    try:
        with open('/proc/vmstat', 'r') as f:
            sample['vmstat'] = parse_proc_vmstat(f.read())
    except Exception:
        pass

    try:
        with open('/proc/loadavg', 'r') as f:
            sample['loadavg'] = parse_proc_loadavg(f.read())
    except Exception:
        pass

    try:
        with open('/proc/meminfo', 'r') as f:
            sample['meminfo'] = parse_proc_meminfo(f.read())
    except Exception:
        pass

    try:
        with open('/proc/net/dev', 'r') as f:
            sample['netdev'] = parse_proc_net_dev(f.read())
    except Exception:
        pass

    return sample
```

### Pattern 3: Multi-Host Time-Series Collection
**What:** Collect from all hosts in parallel at each interval
**When to use:** When benchmark runs across multiple nodes
**Example:**
```python
# Source: Extends existing SSHClusterCollector pattern
class MultiHostTimeSeriesCollector:
    """Collect time-series from multiple hosts."""

    def __init__(self, hosts: List[str], interval: float = 10.0, max_workers: int = 10):
        self.hosts = hosts
        self.interval = interval
        self.max_workers = max_workers
        self._samples_by_host: Dict[str, List[Dict]] = {h: [] for h in hosts}

    def _collect_from_host(self, host: str) -> Dict[str, Any]:
        """Collect single sample from host."""
        if _is_localhost(host):
            return collect_timeseries_sample()
        else:
            # Use SSH to collect (reuse SSHClusterCollector pattern)
            return self._collect_via_ssh(host)

    def _collect_all_hosts(self):
        """Collect from all hosts in parallel."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._collect_from_host, host): host
                for host in self.hosts
            }
            for future in as_completed(futures):
                host = futures[future]
                try:
                    sample = future.result(timeout=self.interval / 2)
                    self._samples_by_host[host].append(sample)
                except Exception as e:
                    # Log error but continue - don't fail collection
                    pass
```

### Pattern 4: Benchmark Integration Hook
**What:** Start time-series collection before _run(), stop after
**When to use:** In Benchmark.run() method
**Example:**
```python
# Source: Extends existing pattern in mlpstorage/benchmarks/base.py
def run(self) -> int:
    """Execute the benchmark with time-series collection."""
    self._validate_environment()

    # Collect cluster info at start (HOST-03)
    self._collect_cluster_start()

    # Start time-series collection (HOST-04, HOST-05)
    self._start_timeseries_collection()

    start_time = time.time()
    try:
        result = self._run()
    finally:
        self.runtime = time.time() - start_time

        # Stop time-series collection and get samples
        self._stop_timeseries_collection()

        # Collect cluster info at end (HOST-03)
        self._collect_cluster_end()

    return result
```

### Anti-Patterns to Avoid
- **Using daemon=True:** Daemon threads are killed abruptly without cleanup, may leave partial data
- **Using sleep() instead of Event.wait():** sleep() doesn't respond to stop signals quickly
- **Collecting static data every interval:** cpuinfo, os_release don't change - waste of resources
- **Writing to file during collection:** Frequent I/O could impact benchmark performance
- **Ignoring collection errors:** Log errors but don't fail the benchmark; collection is secondary
- **Large in-memory buffer without limits:** Add max_samples limit to prevent memory issues in long runs

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| /proc parsing | New parsers | Existing `parse_proc_*` functions | Already tested, handle edge cases |
| Localhost detection | String comparison | `_is_localhost()` function | Handles hostname, FQDN, IP variations |
| SSH collection | New SSH code | SSHClusterCollector pattern | Already handles BatchMode, timeouts |
| Thread-safe stop | Flags with locks | threading.Event | Designed for this purpose, thread-safe |
| Parallel collection | Manual threading | ThreadPoolExecutor | Simpler, handles exceptions properly |
| JSON serialization | Custom encoder | MLPSJsonEncoder | Already handles enums, sets, dataclasses |

**Key insight:** Phase 6 already implemented all the hard parts (parsers, SSH, parallel collection). Time-series collection is primarily about adding a background thread wrapper around existing collection functions.

## Common Pitfalls

### Pitfall 1: Daemon Thread Data Loss
**What goes wrong:** Using daemon=True causes thread to be killed without saving collected data
**Why it happens:** Daemon threads are convenient but terminate abruptly at program exit
**How to avoid:** Use daemon=False with Event signaling; always call stop() before program exit
**Warning signs:** Missing or truncated time-series files, data only from beginning of run

### Pitfall 2: Blocking the Stop Signal
**What goes wrong:** Collection takes longer than interval, stop signal delayed
**Why it happens:** SSH collection to slow/unresponsive host blocks the collection thread
**How to avoid:** Set timeouts on SSH operations; use timeout on ThreadPoolExecutor.result()
**Warning signs:** Benchmark completes but program hangs waiting for time-series thread

### Pitfall 3: Memory Growth in Long Runs
**What goes wrong:** Collecting samples every 10 seconds for hours fills memory
**Why it happens:** Unbounded list of samples grows indefinitely
**How to avoid:** Add max_samples limit; consider streaming to file for very long runs
**Warning signs:** Memory usage grows linearly with runtime, OOM on long benchmarks

### Pitfall 4: Inconsistent Sample Timing
**What goes wrong:** Samples drift from 10-second intervals due to collection time
**Why it happens:** Using sleep(10) instead of calculating next sample time
**How to avoid:** Calculate next sample time as start + n*interval, wait for that time
**Warning signs:** Timestamp deltas vary significantly from 10 seconds

### Pitfall 5: Impacting Benchmark Performance
**What goes wrong:** Time-series collection causes measurable benchmark slowdown
**Why it happens:** Collection happening on same CPU cores as benchmark
**How to avoid:** Keep collection lightweight; use single thread; avoid excessive I/O
**Warning signs:** Benchmark metrics differ significantly between collection enabled/disabled

### Pitfall 6: SSH Collection Failures Breaking Collection
**What goes wrong:** One failed SSH causes entire collection interval to fail
**Why it happens:** Not catching exceptions per-host
**How to avoid:** Catch exceptions per-host, continue with available data
**Warning signs:** Missing data from all hosts when one host is unreachable

## Code Examples

Verified patterns from official sources and existing codebase:

### Background Thread with Event (Python Docs)
```python
# Source: https://docs.python.org/3/library/threading.html
import threading

class BackgroundCollector:
    def __init__(self, interval: float):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=False)

    def _run(self):
        while not self._stop_event.is_set():
            self._do_collection()
            self._stop_event.wait(timeout=self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
```

### Existing /proc Parsers (Reuse)
```python
# Source: mlpstorage/cluster_collector.py
from mlpstorage.cluster_collector import (
    parse_proc_diskstats,
    parse_proc_vmstat,
    parse_proc_loadavg,
    parse_proc_meminfo,
    parse_proc_net_dev,
    _is_localhost,
)

# These functions are already tested and handle edge cases
# Time-series collection should reuse them directly
```

### ThreadPoolExecutor for Parallel Collection (Existing Pattern)
```python
# Source: mlpstorage/cluster_collector.py SSHClusterCollector.collect()
from concurrent.futures import ThreadPoolExecutor, as_completed

def collect_from_hosts(hosts: List[str], timeout: float) -> Dict[str, Any]:
    results = {}
    with ThreadPoolExecutor(max_workers=min(10, len(hosts))) as executor:
        future_to_host = {
            executor.submit(collect_from_single_host, host): host
            for host in hosts
        }
        for future in as_completed(future_to_host):
            host = future_to_host[future]
            try:
                results[host] = future.result(timeout=timeout)
            except Exception as e:
                results[host] = {'error': str(e)}
    return results
```

### JSON Output with Existing Encoder
```python
# Source: mlpstorage/utils.py and mlpstorage/benchmarks/base.py
import json
from mlpstorage.utils import MLPSJsonEncoder

def write_timeseries_data(data: Dict[str, Any], output_path: str):
    """Write time-series data to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, cls=MLPSJsonEncoder)
```

## Data Models

### TimeSeriesSample Dataclass
```python
@dataclass
class TimeSeriesSample:
    """Single time-series sample from one host."""
    timestamp: str  # ISO format
    hostname: str
    diskstats: Optional[List[Dict[str, Any]]] = None
    vmstat: Optional[Dict[str, int]] = None
    meminfo: Optional[Dict[str, int]] = None
    loadavg: Optional[Dict[str, float]] = None
    netdev: Optional[List[Dict[str, Any]]] = None
    errors: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
```

### TimeSeriesData Dataclass
```python
@dataclass
class TimeSeriesData:
    """Complete time-series collection for a benchmark run."""
    collection_interval_seconds: float
    start_time: str  # ISO format
    end_time: str  # ISO format
    num_samples: int
    samples_by_host: Dict[str, List[TimeSeriesSample]]
    collection_method: str  # 'local', 'ssh', 'mpi'
    hosts_requested: List[str]
    hosts_collected: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'collection_interval_seconds': self.collection_interval_seconds,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'num_samples': self.num_samples,
            'samples_by_host': {
                host: [s.to_dict() for s in samples]
                for host, samples in self.samples_by_host.items()
            },
            'collection_method': self.collection_method,
            'hosts_requested': self.hosts_requested,
            'hosts_collected': self.hosts_collected,
        }
```

## CLI Arguments

New arguments needed for time-series control:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--timeseries-interval` | int | 10 | Collection interval in seconds |
| `--skip-timeseries` | flag | False | Disable time-series collection |
| `--max-timeseries-samples` | int | 3600 | Maximum samples to keep (10 hours at 10s interval) |

## Output File Format

Time-series data should be written to a separate file in the results directory:

```
{results_dir}/
  {benchmark_type}_{datetime}_metadata.json     # Existing
  {benchmark_type}_{datetime}_cluster_info.json # Existing
  {benchmark_type}_{datetime}_timeseries.json   # NEW
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| daemon=True for background | daemon=False with Event | Always best practice | Prevents data loss |
| sleep() for intervals | Event.wait(timeout) | Python 3.0+ | Quick response to stop |
| Lock for thread coordination | Event for stop signals | Context-dependent | Simpler, less error-prone |
| multiprocessing for background | threading for I/O-bound | Always for I/O | Lower overhead |

**Deprecated/outdated:**
- `thread` module: Use `threading` instead (thread is deprecated since Python 3)
- Creating new thread per sample: Use single thread with loop

## Open Questions

Things that couldn't be fully resolved:

1. **Exact timing precision**
   - What we know: Event.wait(timeout) may not be exactly 10 seconds
   - What's unclear: How much drift is acceptable?
   - Recommendation: Document actual intervals in output; calculate delta from timestamps

2. **Very long benchmark handling**
   - What we know: 10-hour run = 3600 samples = ~10MB in memory
   - What's unclear: Should we support even longer runs?
   - Recommendation: Default max_samples=3600; document limitation; offer --max-timeseries-samples

3. **Host failure mid-collection**
   - What we know: Host may become unreachable during run
   - What's unclear: How to represent partial host data?
   - Recommendation: Include samples collected before failure; mark host as 'partial' in metadata

## Sources

### Primary (HIGH confidence)
- [Python threading documentation](https://docs.python.org/3/library/threading.html) - Thread, Event, Timer classes
- `mlpstorage/cluster_collector.py` - Existing /proc parsers, SSHClusterCollector, localhost detection
- `mlpstorage/benchmarks/base.py` - Existing collection integration patterns
- `mlpstorage/rules/models.py` - ClusterSnapshots pattern for dataclasses

### Secondary (MEDIUM confidence)
- [Super Fast Python: Periodic Background Tasks](https://superfastpython.com/thread-periodic-background/) - Background thread patterns
- [Super Fast Python: Daemon Threads](https://superfastpython.com/daemon-threads-in-python/) - Daemon vs non-daemon

### Tertiary (LOW confidence)
- Web search on threading vs multiprocessing for I/O-bound tasks - General guidance

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All stdlib, matches existing codebase patterns
- Architecture: HIGH - Extends proven patterns from Phase 6
- Pitfalls: HIGH - Based on Python documentation and existing code analysis
- Performance impact: MEDIUM - Theoretical analysis; should validate with actual benchmarks

**Research date:** 2026-01-24
**Valid until:** 2026-02-24 (30 days - stable domain)
