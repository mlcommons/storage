---
phase: 07-time-series-host-data-collection
plan: 01
subsystem: cluster-collection
tags:
  - time-series
  - dataclasses
  - threading
  - proc-parsers
  - background-collection
requires:
  - proc-vmstat-parser
  - proc-meminfo-parser
  - proc-diskstats-parser
  - proc-loadavg-parser
  - proc-netdev-parser
provides:
  - TimeSeriesSample dataclass
  - TimeSeriesData dataclass
  - collect_timeseries_sample function
  - TimeSeriesCollector class
affects:
  - HOST-04 requirement support
  - Benchmark time-series data collection
  - System resource monitoring during benchmark runs
tech-stack:
  added: []
  patterns:
    - Dataclass with to_dict/from_dict for serialization
    - Background thread with Event-based graceful shutdown
    - Dynamic /proc metric collection
decisions:
  - id: thread-event-pattern
    choice: Use threading.Event with wait(timeout) for graceful shutdown
    rationale: Allows quick response to stop signal without busy-waiting
  - id: max-samples-limit
    choice: Enforce max_samples limit (default 3600) to prevent memory issues
    rationale: Long-running benchmarks should not exhaust memory with samples
  - id: check-stopped-first
    choice: Check _stopped before _started in start() method
    rationale: Provides more accurate error messages when restarting a stopped collector
key-files:
  created: []
  modified:
    - mlpstorage/rules/models.py
    - mlpstorage/cluster_collector.py
    - tests/unit/test_cluster_collector.py
metrics:
  duration: 202 seconds
  completed: 2026-01-24
---

# Phase 07 Plan 01: Core Time-Series Infrastructure Summary

**One-liner:** TimeSeriesSample/TimeSeriesData dataclasses and background-threaded TimeSeriesCollector for 10-second interval system metric collection during benchmarks

## What Was Built

1. **TimeSeriesSample Dataclass**: Structured representation of a single time-series sample with timestamp, hostname, and optional dynamic metrics (diskstats, vmstat, meminfo, loadavg, netdev, errors)

2. **TimeSeriesData Dataclass**: Aggregation container for all samples collected during a benchmark run, organized by host with collection metadata

3. **collect_timeseries_sample() Function**: Collects dynamic /proc metrics (diskstats, vmstat, loadavg, meminfo, netdev) into a single sample dict

4. **TimeSeriesCollector Class**: Background thread-based collector with start()/stop() lifecycle, configurable interval, and max_samples limit

5. **Comprehensive Unit Tests**: 23 tests covering sample collection, collector lifecycle, and dataclass serialization

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Add TimeSeriesSample and TimeSeriesData dataclasses | 57f645c | mlpstorage/rules/models.py |
| 2 | Add collect_timeseries_sample and TimeSeriesCollector | 053a0b8 | mlpstorage/cluster_collector.py |
| 3 | Add unit tests for time-series collection | 4588be0 | tests/unit/test_cluster_collector.py, mlpstorage/cluster_collector.py |

## Technical Details

### New Dataclasses (models.py)

```python
@dataclass
class TimeSeriesSample:
    """Single time-series sample from one host."""
    timestamp: str  # ISO format YYYY-MM-DDTHH:MM:SSZ
    hostname: str
    diskstats: Optional[List[Dict[str, Any]]] = None
    vmstat: Optional[Dict[str, int]] = None
    meminfo: Optional[Dict[str, int]] = None
    loadavg: Optional[Dict[str, float]] = None
    netdev: Optional[List[Dict[str, Any]]] = None
    errors: Optional[Dict[str, str]] = None

@dataclass
class TimeSeriesData:
    """Complete time-series collection for a benchmark run."""
    collection_interval_seconds: float
    start_time: str
    end_time: str
    num_samples: int
    samples_by_host: Dict[str, List[TimeSeriesSample]]
    collection_method: str
    hosts_requested: List[str]
    hosts_collected: List[str]
```

### Collector Class (cluster_collector.py)

```python
class TimeSeriesCollector:
    def __init__(self, interval_seconds=10.0, max_samples=3600, logger=None)
    def start(self) -> None
    def stop(self) -> List[Dict[str, Any]]

    @property
    def samples(self) -> List[Dict[str, Any]]
    @property
    def start_time(self) -> Optional[str]
    @property
    def end_time(self) -> Optional[str]
    @property
    def is_running(self) -> bool
```

### Sample Collection Output

Running `collect_timeseries_sample()` returns:
```python
{
    'timestamp': '2026-01-24T22:20:00Z',
    'hostname': 'myhost',
    'diskstats': [...],  # List of disk stats
    'vmstat': {...},     # Dict of vmstat key-value pairs
    'loadavg': {         # Load average data
        'load_1min': 0.5,
        'load_5min': 0.6,
        'load_15min': 0.7,
        'running_processes': 2,
        'total_processes': 500
    },
    'meminfo': {...},    # Memory info
    'netdev': [...]      # Network interface stats
}
```

## Verification Results

All verification criteria met:

1. All imports work:
   ```
   from mlpstorage.rules.models import TimeSeriesSample, TimeSeriesData
   from mlpstorage.cluster_collector import collect_timeseries_sample, TimeSeriesCollector
   ```

2. Functional test passed:
   - Collector at 0.5s interval collected 3 samples in 1.5 seconds

3. Unit tests pass: 23 passed (total 85 in test_cluster_collector.py)

### Must-Haves Verification

**Truths:**
- TimeSeriesSample dataclass can be instantiated with all dynamic metric fields: VERIFIED
- TimeSeriesData dataclass aggregates samples by host with metadata: VERIFIED
- collect_timeseries_sample() returns dict with diskstats, vmstat, loadavg, meminfo, netdev: VERIFIED
- TimeSeriesCollector collects samples at specified intervals in background thread: VERIFIED
- TimeSeriesCollector.stop() returns all collected samples: VERIFIED

**Artifacts:**
- mlpstorage/rules/models.py provides TimeSeriesSample and TimeSeriesData: VERIFIED (contains "class TimeSeriesSample")
- mlpstorage/cluster_collector.py provides collector: VERIFIED (contains "class TimeSeriesCollector")
- tests/unit/test_cluster_collector.py provides unit tests: VERIFIED (contains "test_timeseries")

**Key Links:**
- cluster_collector.py calls parse_proc_diskstats, parse_proc_vmstat, parse_proc_loadavg, parse_proc_meminfo, parse_proc_net_dev: VERIFIED
- cluster_collector.py uses threading.Event for stop signal: VERIFIED

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed check order in start() method**
- **Found during:** Task 3 (test_reuse_after_stop_raises_error failed)
- **Issue:** Checking `_started` before `_stopped` gave wrong error message when restarting stopped collector
- **Fix:** Reordered checks to check `_stopped` first
- **Files modified:** mlpstorage/cluster_collector.py
- **Commit:** 4588be0

## Decisions Made

**Decision 1: Use threading.Event with wait(timeout) for graceful shutdown**
- **Context:** Need to stop collection thread quickly when benchmark ends
- **Choice:** Use Event.wait(timeout) instead of time.sleep()
- **Rationale:** Allows thread to respond immediately to stop signal instead of waiting for sleep to complete
- **Impact:** Collector stops within milliseconds of stop() being called

**Decision 2: Enforce max_samples limit**
- **Context:** Long benchmarks could collect thousands of samples
- **Choice:** Default max_samples=3600 (~10 hours at 10s interval)
- **Rationale:** Prevents memory exhaustion, logs warning when limit reached
- **Impact:** Memory usage is bounded regardless of benchmark duration

**Decision 3: Check stopped before started in start() method**
- **Context:** Test expected "already stopped" but got "already started"
- **Choice:** Check `_stopped` flag before `_started` flag
- **Rationale:** More accurate error message when trying to restart stopped collector
- **Impact:** Users get helpful error message about creating new instance

## Integration Points

**Upstream Dependencies:**
- Uses parse_proc_diskstats, parse_proc_vmstat, parse_proc_loadavg, parse_proc_meminfo, parse_proc_net_dev from cluster_collector.py (Phase 6)

**Downstream Consumers:**
- 07-02 will integrate TimeSeriesCollector into Benchmark base class
- 07-03 will add multi-host time-series collection via SSH
- Result files will include time-series data for analysis

## Files Changed

### Modified

**mlpstorage/rules/models.py**
- Added TimeSeriesSample dataclass with to_dict/from_dict methods
- Added TimeSeriesData dataclass with to_dict/from_dict methods
- Added asdict import from dataclasses

**mlpstorage/cluster_collector.py**
- Added threading import
- Added collect_timeseries_sample() function
- Added TimeSeriesCollector class with background thread collection
- Fixed check order in start() method

**tests/unit/test_cluster_collector.py**
- Added time import
- Added collect_timeseries_sample, TimeSeriesCollector imports
- Added TestCollectTimeseriesSample class (7 tests)
- Added TestTimeSeriesCollector class (10 tests)
- Added TestTimeSeriesSampleDataclass class (3 tests)
- Added TestTimeSeriesDataDataclass class (3 tests)

## Testing Notes

Test execution results:
```
TestCollectTimeseriesSample::test_returns_dict_with_required_fields PASSED
TestCollectTimeseriesSample::test_contains_diskstats PASSED
TestCollectTimeseriesSample::test_contains_vmstat PASSED
TestCollectTimeseriesSample::test_contains_loadavg PASSED
TestCollectTimeseriesSample::test_contains_meminfo PASSED
TestCollectTimeseriesSample::test_contains_netdev PASSED
TestCollectTimeseriesSample::test_no_errors_key_when_successful PASSED
TestTimeSeriesCollector::test_init_sets_defaults PASSED
TestTimeSeriesCollector::test_init_custom_values PASSED
TestTimeSeriesCollector::test_start_sets_running PASSED
TestTimeSeriesCollector::test_stop_returns_samples PASSED
TestTimeSeriesCollector::test_collects_samples_at_interval PASSED
TestTimeSeriesCollector::test_max_samples_limit_enforced PASSED
TestTimeSeriesCollector::test_start_twice_raises_error PASSED
TestTimeSeriesCollector::test_stop_without_start_raises_error PASSED
TestTimeSeriesCollector::test_reuse_after_stop_raises_error PASSED
TestTimeSeriesCollector::test_samples_contain_expected_fields PASSED
TestTimeSeriesSampleDataclass::test_create_with_required_fields PASSED
TestTimeSeriesSampleDataclass::test_to_dict_excludes_none PASSED
TestTimeSeriesSampleDataclass::test_from_dict_roundtrip PASSED
TestTimeSeriesDataDataclass::test_create_with_fields PASSED
TestTimeSeriesDataDataclass::test_to_dict_serializes_samples PASSED
TestTimeSeriesDataDataclass::test_from_dict_roundtrip PASSED
```

All 23 new tests pass, total 85 tests in test_cluster_collector.py.

## Lessons Learned

**What Went Well:**
- Existing dataclass patterns (ClusterSnapshots, HostDiskInfo) provided clear templates
- Existing /proc parsers provided foundation for collect_timeseries_sample
- Thread-based collection is straightforward with threading.Event

**For Future Plans:**
- 07-02 will integrate TimeSeriesCollector into Benchmark base class
- 07-03 will need SSH-based collection for multi-host time-series
- May want to add filtering for specific disk devices or network interfaces

## Performance Notes

Execution time: ~202 seconds (~3.4 minutes)

Tasks: 3 completed in 3 commits

Commits:
- 57f645c: feat(07-01): add TimeSeriesSample and TimeSeriesData dataclasses
- 053a0b8: feat(07-01): add collect_timeseries_sample and TimeSeriesCollector
- 4588be0: test(07-01): add unit tests for time-series collection

---

**Summary created:** 2026-01-24
**Executor:** Claude Opus 4.5
**Status:** Complete
