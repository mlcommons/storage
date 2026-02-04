---
phase: 07-time-series-host-data-collection
plan: 02
subsystem: cluster-collection
tags:
  - time-series
  - multi-host
  - ssh
  - parallel-collection
  - threading
requires:
  - TimeSeriesCollector
  - collect_timeseries_sample
  - _is_localhost
  - SSH patterns from SSHClusterCollector
provides:
  - MultiHostTimeSeriesCollector class
  - TIMESERIES_SSH_SCRIPT for remote collection
affects:
  - HOST-05 requirement support (parallel collection without performance impact)
  - Multi-host time-series data collection
  - Benchmark time-series data organized by hostname
tech-stack:
  added: []
  patterns:
    - Parallel SSH collection via ThreadPoolExecutor
    - Localhost optimization for direct collection
    - Background thread with Event-based graceful shutdown
    - Samples organized by hostname
decisions:
  - id: parallel-ssh-collection
    choice: Use ThreadPoolExecutor for parallel host collection
    rationale: Collects from all hosts simultaneously without waiting for each SSH
  - id: localhost-direct-collection
    choice: Use _is_localhost to detect local hosts and skip SSH
    rationale: Avoids SSH overhead and configuration requirements for localhost
  - id: graceful-host-failure
    choice: Continue collection even when some hosts fail
    rationale: Partial data is better than no data for debugging
key-files:
  created: []
  modified:
    - mlpstorage/cluster_collector.py
    - tests/unit/test_cluster_collector.py
metrics:
  duration: 184 seconds
  completed: 2026-01-24
---

# Phase 07 Plan 02: Multi-Host Time-Series Collection Summary

**One-liner:** MultiHostTimeSeriesCollector with parallel SSH collection for multiple hosts, localhost optimization, and graceful failure handling

## What Was Built

1. **TIMESERIES_SSH_SCRIPT**: Lightweight Python script for remote time-series metric collection via SSH, collecting only dynamic /proc metrics

2. **MultiHostTimeSeriesCollector Class**: Parallel multi-host collector with SSH for remote hosts and direct collection for localhost

3. **Parallel Collection**: Uses ThreadPoolExecutor to collect from all hosts simultaneously at each interval

4. **Localhost Optimization**: Detects localhost via _is_localhost() and uses direct collection instead of SSH

5. **Graceful Failure Handling**: Collection continues even when some hosts are unreachable, with error samples recorded

6. **Comprehensive Unit Tests**: 13 tests covering initialization, lifecycle, collection, and failure handling

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Add MultiHostTimeSeriesCollector class | 5f7dd38 | mlpstorage/cluster_collector.py |
| 2 | Add unit tests for MultiHostTimeSeriesCollector | 4a515a0 | tests/unit/test_cluster_collector.py |

## Technical Details

### TIMESERIES_SSH_SCRIPT

Lightweight script for remote collection (collects only dynamic metrics):
```python
# Collects: diskstats, vmstat, loadavg, meminfo, netdev
# Outputs: JSON to stdout
# Errors: Included in 'errors' dict if any
```

### MultiHostTimeSeriesCollector Class

```python
class MultiHostTimeSeriesCollector:
    def __init__(
        self,
        hosts: List[str],
        interval_seconds: float = 10.0,
        max_samples: int = 3600,
        ssh_username: Optional[str] = None,
        ssh_timeout: int = 30,
        max_workers: int = 10,
        logger=None
    )
    def start(self) -> None
    def stop(self) -> Dict[str, List[Dict[str, Any]]]

    @property
    def samples_by_host(self) -> Dict[str, List[Dict[str, Any]]]
    @property
    def start_time(self) -> Optional[str]
    @property
    def end_time(self) -> Optional[str]
    @property
    def is_running(self) -> bool
    def get_hosts_with_data(self) -> List[str]
```

### Collection Flow

1. **Initialization**: Parse hosts list, deduplicate, remove slot counts
2. **Start**: Begin background thread, set start_time
3. **Collection Loop**: At each interval:
   - Submit collection tasks to ThreadPoolExecutor (one per host)
   - For each host:
     - If localhost: call collect_timeseries_sample() directly
     - If remote: SSH + TIMESERIES_SSH_SCRIPT + parse response
   - Store samples in _samples_by_host dict
4. **Stop**: Signal stop event, wait for thread, return samples_by_host

### Sample Output Structure

```python
{
    'localhost': [
        {
            'timestamp': '2026-01-24T22:26:00Z',
            'hostname': 'localhost',
            'diskstats': [...],
            'vmstat': {...},
            'loadavg': {...},
            'meminfo': {...},
            'netdev': [...]
        },
        ...
    ],
    'remote-host-1': [
        {...},
        ...
    ]
}
```

## Verification Results

All verification criteria met:

1. Import check passed:
   ```
   from mlpstorage.cluster_collector import MultiHostTimeSeriesCollector, TIMESERIES_SSH_SCRIPT
   from concurrent.futures import as_completed
   ```

2. Functional test passed:
   - Collector with localhost at 0.5s interval collected 3 samples in 1.5 seconds

3. Unit tests pass: 13 passed (total 98 in test_cluster_collector.py)

### Must-Haves Verification

**Truths:**
- MultiHostTimeSeriesCollector collects from multiple hosts in parallel: VERIFIED (ThreadPoolExecutor pattern)
- Localhost hosts use direct collection, remote hosts use SSH: VERIFIED (_is_localhost check)
- Collection continues even if some hosts fail: VERIFIED (error samples recorded, collection continues)
- Samples are organized by hostname: VERIFIED (samples_by_host dict structure)

**Artifacts:**
- mlpstorage/cluster_collector.py provides MultiHostTimeSeriesCollector: VERIFIED
- tests/unit/test_cluster_collector.py provides TestMultiHostTimeSeriesCollector: VERIFIED

**Key Links:**
- Uses _stop_event.wait pattern from TimeSeriesCollector: VERIFIED
- Uses ThreadPoolExecutor for parallel collection: VERIFIED
- Uses _is_localhost for localhost detection: VERIFIED
- Uses as_completed for future handling: VERIFIED

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

**Decision 1: Use ThreadPoolExecutor for parallel host collection**
- **Context:** Need to collect from all hosts at each interval without blocking
- **Choice:** Use ThreadPoolExecutor with max_workers limit
- **Rationale:** Collects from all hosts simultaneously, limited by max_workers
- **Impact:** Collection time is dominated by slowest host, not cumulative

**Decision 2: Use _is_localhost for localhost detection**
- **Context:** Localhost collection should not require SSH
- **Choice:** Reuse existing _is_localhost function from SSHClusterCollector
- **Rationale:** Consistent localhost detection, avoids SSH overhead
- **Impact:** Faster collection for localhost, no SSH configuration needed

**Decision 3: Continue collection when hosts fail**
- **Context:** Some hosts may be temporarily unreachable
- **Choice:** Record error sample and continue collection
- **Rationale:** Partial data is better than no data for analysis
- **Impact:** Collection does not abort on individual host failures

## Integration Points

**Upstream Dependencies:**
- TimeSeriesCollector (07-01) - collection loop pattern
- collect_timeseries_sample (07-01) - local sample collection
- _is_localhost (06-02) - localhost detection
- parse_proc_* functions (06-01) - parsing remote collection output

**Downstream Consumers:**
- 07-03 will integrate MultiHostTimeSeriesCollector into Benchmark base class
- Result files will include time-series data organized by host

## Files Changed

### Modified

**mlpstorage/cluster_collector.py**
- Added TIMESERIES_SSH_SCRIPT constant for remote collection
- Added MultiHostTimeSeriesCollector class with:
  - __init__ with hosts, interval, max_samples, ssh_username, ssh_timeout, max_workers
  - _get_unique_hosts for host deduplication
  - _build_ssh_command for SSH command construction
  - _parse_remote_sample for parsing SSH collection output
  - _collect_from_host for single host collection
  - _collect_all_hosts for parallel collection
  - _collection_loop for background thread
  - start(), stop() lifecycle methods
  - Properties: samples_by_host, start_time, end_time, is_running
  - get_hosts_with_data helper method

**tests/unit/test_cluster_collector.py**
- Added MultiHostTimeSeriesCollector import
- Added TestMultiHostTimeSeriesCollector class with 13 tests:
  - test_init_sets_defaults
  - test_init_custom_values
  - test_deduplicates_hosts
  - test_removes_slot_counts
  - test_start_sets_running
  - test_stop_returns_samples_by_host
  - test_collects_from_localhost
  - test_samples_have_expected_structure
  - test_max_samples_per_host_enforced
  - test_start_twice_raises_error
  - test_stop_without_start_raises_error
  - test_get_hosts_with_data
  - test_handles_unreachable_host_gracefully

## Testing Notes

Test execution results:
```
TestMultiHostTimeSeriesCollector::test_init_sets_defaults PASSED
TestMultiHostTimeSeriesCollector::test_init_custom_values PASSED
TestMultiHostTimeSeriesCollector::test_deduplicates_hosts PASSED
TestMultiHostTimeSeriesCollector::test_removes_slot_counts PASSED
TestMultiHostTimeSeriesCollector::test_start_sets_running PASSED
TestMultiHostTimeSeriesCollector::test_stop_returns_samples_by_host PASSED
TestMultiHostTimeSeriesCollector::test_collects_from_localhost PASSED
TestMultiHostTimeSeriesCollector::test_samples_have_expected_structure PASSED
TestMultiHostTimeSeriesCollector::test_max_samples_per_host_enforced PASSED
TestMultiHostTimeSeriesCollector::test_start_twice_raises_error PASSED
TestMultiHostTimeSeriesCollector::test_stop_without_start_raises_error PASSED
TestMultiHostTimeSeriesCollector::test_get_hosts_with_data PASSED
TestMultiHostTimeSeriesCollector::test_handles_unreachable_host_gracefully PASSED
```

All 13 new tests pass, total 98 tests in test_cluster_collector.py.

## Lessons Learned

**What Went Well:**
- Existing TimeSeriesCollector provided clear template for collection loop
- SSHClusterCollector patterns (localhost detection, SSH command building) were reusable
- ThreadPoolExecutor integration straightforward with as_completed

**For Future Plans:**
- 07-03 will integrate MultiHostTimeSeriesCollector into Benchmark base class
- May want to add configurable collection timeout per host
- May want to add host-specific SSH options

## Performance Notes

Execution time: ~184 seconds (~3.1 minutes)

Tasks: 2 completed in 2 commits

Commits:
- 5f7dd38: feat(07-02): add MultiHostTimeSeriesCollector for parallel multi-host collection
- 4a515a0: test(07-02): add unit tests for MultiHostTimeSeriesCollector

---

**Summary created:** 2026-01-24
**Executor:** Claude Opus 4.5
**Status:** Complete
