---
phase: 06-ssh-based-host-collection
plan: 03
subsystem: benchmark-integration
tags:
  - ssh-collection
  - cluster-snapshots
  - benchmark-base
  - host-03
requires:
  - 06-02 (SSHClusterCollector)
provides:
  - benchmark-ssh-integration
  - cluster-snapshots
  - collection-method-selection
affects:
  - KV Cache benchmark now uses SSH collection automatically
  - VectorDB benchmark now uses SSH collection automatically
  - HOST-03 requirement fulfilled (start/end collection)
tech-stack:
  added: []
  patterns:
    - Collection method selection based on exec_type
    - Start/end cluster snapshots for state comparison
    - Backward-compatible cluster_information attribute
decisions:
  - id: ssh-for-non-mpi
    choice: Use SSH collection when exec_type is not MPI
    rationale: Non-MPI benchmarks have no MPI infrastructure for collection
  - id: start-end-snapshots
    choice: Collect cluster info at start and end of benchmark
    rationale: Enables analysis of system state changes (HOST-03 requirement)
  - id: backward-compatible
    choice: Set cluster_information from start snapshot
    rationale: Existing code expecting cluster_information continues to work
key-files:
  created: []
  modified:
    - mlpstorage/rules/models.py
    - mlpstorage/benchmarks/base.py
    - tests/unit/test_benchmarks_base.py
metrics:
  duration: ~6 minutes
  completed: 2026-01-24
---

# Phase 06 Plan 03: Benchmark Base Integration Summary

**One-liner:** SSH collection integrated into Benchmark base class with ClusterSnapshots for HOST-03 start/end state capture

## What Was Built

1. **ClusterSnapshots Dataclass**: New dataclass in models.py to hold start and end ClusterInformation with collection method tracking

2. **SSH Collection Integration**: Updated Benchmark base class with:
   - `_should_use_ssh_collection()`: Determines when to use SSH vs MPI collection
   - `_collect_via_ssh()`: Collects using SSHClusterCollector
   - `_collect_cluster_start()`: Collects at benchmark start
   - `_collect_cluster_end()`: Collects at benchmark end and creates ClusterSnapshots

3. **run() Method Update**: Now calls start/end collection to satisfy HOST-03 requirement

4. **Metadata Enhancement**: cluster_snapshots included in metadata when available

5. **Comprehensive Unit Tests**: 15 new tests for collection selection and snapshot functionality

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Add ClusterSnapshots dataclass to models.py | 85e8bc1 | mlpstorage/rules/models.py |
| 2 | Update Benchmark base class with SSH collection and snapshots | e8a00f9 | mlpstorage/benchmarks/base.py |
| 3 | Add unit tests for collection method selection | 2551bdf | tests/unit/test_benchmarks_base.py |

## Technical Details

### ClusterSnapshots Dataclass

```python
@dataclass
class ClusterSnapshots:
    """Cluster information snapshots from benchmark start and end."""
    start: 'ClusterInformation'
    end: Optional['ClusterInformation'] = None
    collection_method: str = 'unknown'

    def as_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any], logger) -> Optional['ClusterSnapshots']: ...
```

### Collection Method Selection

```python
def _should_use_ssh_collection(self) -> bool:
    """Use SSH when:
    - hosts are specified
    - exec_type is NOT MPI
    - command is 'run' (not datagen/configview)
    - skip_cluster_collection is not set
    """
```

### Updated run() Flow

```python
def run(self) -> int:
    self._validate_environment()
    self._collect_cluster_start()  # HOST-03: start snapshot
    start_time = time.time()
    result = self._run()
    self.runtime = time.time() - start_time
    self._collect_cluster_end()    # HOST-03: end snapshot
    return result
```

## Verification Results

All verification criteria met:

1. **ClusterSnapshots imports correctly**:
   ```python
   from mlpstorage.rules.models import ClusterSnapshots
   # Success
   ```

2. **Benchmark base class imports successfully**:
   ```python
   from mlpstorage.benchmarks.base import Benchmark
   # Success
   ```

3. **All unit tests pass**: 54 tests passed

4. **SSH collection selected for non-MPI benchmarks**:
   ```python
   benchmark._should_use_ssh_collection()  # True when hosts+non-MPI
   ```

### Must-Haves Verification

**Truths:**
- Benchmark base class supports SSH-based collection for non-MPI benchmarks: VERIFIED
- Collection happens at benchmark start and end (HOST-03): VERIFIED
- Metadata includes both start and end cluster snapshots: VERIFIED
- Non-MPI benchmarks with --hosts use SSH collection automatically: VERIFIED

**Artifacts:**
- mlpstorage/benchmarks/base.py provides "SSH collection integration and start/end snapshots": VERIFIED
- mlpstorage/rules/models.py provides "ClusterSnapshots dataclass": VERIFIED
- tests/unit/test_benchmarks_base.py provides "Tests for collection method selection": VERIFIED

**Key Links:**
- SSHClusterCollector imported and instantiated in base.py: VERIFIED
- _collect_cluster_start and _collect_cluster_end called in run(): VERIFIED

## Deviations from Plan

**Test Fix: test_tracks_runtime**
- **Rule Applied:** Rule 1 - Bug
- **Found during:** Task 3 test execution
- **Issue:** Existing test test_tracks_runtime was failing because run() now calls more methods that might use time.time()
- **Fix:** Updated test to mock _collect_cluster_start and _collect_cluster_end
- **Files modified:** tests/unit/test_benchmarks_base.py
- **Impact:** Test now properly isolates time tracking behavior

**Test Fix: EXEC_TYPE.NONE**
- **Rule Applied:** Rule 1 - Bug
- **Found during:** Task 3 test execution
- **Issue:** Test used EXEC_TYPE.NONE which doesn't exist (only MPI and DOCKER)
- **Fix:** Changed test to use EXEC_TYPE.DOCKER which is also non-MPI
- **Files modified:** tests/unit/test_benchmarks_base.py
- **Impact:** Test correctly validates SSH collection for non-MPI exec types

## Decisions Made

**Decision 1: SSH for non-MPI execution**
- **Context:** Need to determine when to use SSH vs MPI collection
- **Choice:** Use SSH when exec_type is not EXEC_TYPE.MPI
- **Rationale:** Non-MPI benchmarks cannot use MPI infrastructure for collection
- **Impact:** KV Cache and VectorDB benchmarks automatically use SSH collection

**Decision 2: Start/end cluster snapshots**
- **Context:** HOST-03 requires capturing cluster state at both start and end
- **Choice:** Create ClusterSnapshots dataclass with start and end fields
- **Rationale:** Enables analysis of system state changes during benchmark
- **Impact:** Metadata now includes comprehensive cluster state history

**Decision 3: Backward-compatible cluster_information**
- **Context:** Existing code uses self.cluster_information
- **Choice:** Set cluster_information from start snapshot
- **Rationale:** Maintains backward compatibility with verification and reporting code
- **Impact:** Existing code continues to work unchanged

## Integration Points

**Upstream Dependencies:**
- 06-02: SSHClusterCollector class
- 06-01: /proc parsers used by collector

**Downstream Consumers:**
- KV Cache benchmark: Automatically uses SSH collection
- VectorDB benchmark: Automatically uses SSH collection
- Metadata system: Includes cluster_snapshots in output
- Result validation: Can analyze start/end state differences

## Files Changed

### Modified
- `mlpstorage/rules/models.py`
  - Added ClusterSnapshots dataclass (~55 lines)
  - Includes as_dict() and from_dict() serialization methods

- `mlpstorage/benchmarks/base.py`
  - Added imports for SSHClusterCollector and ClusterSnapshots
  - Added _should_use_ssh_collection() method
  - Added _collect_via_ssh() method
  - Added _collect_cluster_start() method
  - Added _collect_cluster_end() method
  - Updated run() to call start/end collection
  - Updated metadata property to include cluster_snapshots

- `tests/unit/test_benchmarks_base.py`
  - Added EXEC_TYPE import
  - Added TestBenchmarkCollectionSelection class (10 tests)
  - Added TestBenchmarkClusterSnapshots class (5 tests)
  - Fixed test_tracks_runtime to mock collection methods

## Testing Notes

Test execution results:
```
tests/unit/test_benchmarks_base.py::TestBenchmarkCollectionSelection::test_should_use_ssh_collection_no_hosts PASSED
tests/unit/test_benchmarks_base.py::TestBenchmarkCollectionSelection::test_should_use_ssh_collection_empty_hosts PASSED
tests/unit/test_benchmarks_base.py::TestBenchmarkCollectionSelection::test_selects_ssh_collection_with_hosts_no_exec_type PASSED
tests/unit/test_benchmarks_base.py::TestBenchmarkCollectionSelection::test_selects_ssh_collection_docker_exec_type PASSED
tests/unit/test_benchmarks_base.py::TestBenchmarkCollectionSelection::test_should_not_use_ssh_collection_mpi_exec_type PASSED
tests/unit/test_benchmarks_base.py::TestBenchmarkCollectionSelection::test_should_not_use_ssh_collection_for_datagen PASSED
tests/unit/test_benchmarks_base.py::TestBenchmarkCollectionSelection::test_should_not_use_ssh_collection_for_configview PASSED
tests/unit/test_benchmarks_base.py::TestBenchmarkCollectionSelection::test_should_not_use_ssh_collection_when_disabled PASSED
tests/unit/test_benchmarks_base.py::TestBenchmarkCollectionSelection::test_should_collect_cluster_info_no_hosts PASSED
tests/unit/test_benchmarks_base.py::TestBenchmarkCollectionSelection::test_should_collect_cluster_info_with_hosts_mpi PASSED
tests/unit/test_benchmarks_base.py::TestBenchmarkClusterSnapshots::test_collect_cluster_start_uses_ssh PASSED
tests/unit/test_benchmarks_base.py::TestBenchmarkClusterSnapshots::test_collect_cluster_end_creates_snapshots PASSED
tests/unit/test_benchmarks_base.py::TestBenchmarkClusterSnapshots::test_run_calls_start_and_end_collection PASSED
tests/unit/test_benchmarks_base.py::TestBenchmarkClusterSnapshots::test_metadata_includes_cluster_snapshots PASSED
tests/unit/test_benchmarks_base.py::TestBenchmarkClusterSnapshots::test_skips_end_collection_without_start PASSED
```

All 54 tests pass with no failures or warnings.

## Lessons Learned

**What Went Well:**
- SSHClusterCollector from 06-02 integrated cleanly
- ClusterSnapshots dataclass follows established patterns
- Start/end collection is transparent to subclasses

**For Future Plans:**
- May want to add configuration for collection timeout
- Consider adding delta computation between start/end snapshots
- Could add collection time tracking for performance analysis

## Performance Notes

Execution time: ~6 minutes

Tasks: 3 completed in 3 commits

Commits:
- 85e8bc1: feat(06-03): add ClusterSnapshots dataclass for start/end collection
- e8a00f9: feat(06-03): integrate SSH collection into benchmark base class
- 2551bdf: test(06-03): add tests for collection method selection and snapshots

---

**Summary created:** 2026-01-24
**Executor:** Claude Opus 4.5
**Status:** Complete
