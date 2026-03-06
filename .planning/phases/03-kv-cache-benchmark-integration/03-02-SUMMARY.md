---
phase: 03-kv-cache-benchmark-integration
plan: 02
subsystem: benchmark
tags:
  - kvcache
  - mpi-execution
  - distributed-execution
  - multi-host
requires:
  - 03-01 (KV Cache Distributed CLI Arguments)
provides:
  - kvcache-mpi-execution
  - kvcache-cluster-collection
affects:
  - KV cache benchmark distributed runs
  - mlpstorage kvcache run command
tech-stack:
  added: []
  patterns:
    - MPI command wrapping (from DLIOBenchmark.generate_dlio_command)
    - Cluster information collection for distributed runs
decisions:
  - id: mpi-wrapper-pattern
    choice: Follow DLIOBenchmark pattern for MPI command wrapping
    rationale: Consistent behavior across all DLIO-based and non-DLIO benchmarks
  - id: num-processes-default
    choice: Default num_processes to len(hosts) when not specified
    rationale: Sensible default - one process per host
  - id: cluster-collection-run-only
    choice: Collect cluster information only for 'run' command, not 'datasize'
    rationale: Datasize is a calculation, doesn't need cluster info
key-files:
  created:
    - tests/unit/test_benchmarks_kvcache.py
  modified:
    - mlpstorage/benchmarks/kvcache.py
metrics:
  duration: 171 seconds
  completed: 2026-01-24
---

# Phase 03 Plan 02: KV Cache MPI Execution Support Summary

**One-liner:** KVCacheBenchmark now wraps kv-cache.py with MPI prefix when exec_type=MPI, following DLIOBenchmark pattern

## What Was Built

1. **MPI Command Wrapping**:
   - Detects when `exec_type == EXEC_TYPE.MPI`
   - Generates MPI prefix using `generate_mpi_prefix_cmd`
   - Wraps kv-cache.py command with MPI prefix
   - Passes through all MPI flags (oversubscribe, allow-run-as-root, mpi-params)

2. **Cluster Information Collection**:
   - Collects cluster information for 'run' command
   - Stores in `self.cluster_information`
   - Skipped for 'datasize' command

3. **Test Suite** (12 tests):
   - Local execution tests (no MPI wrapper)
   - Docker execution tests (no MPI wrapper)
   - MPI execution tests (wrapper added)
   - Empty hosts list handling
   - num_processes default behavior
   - MPI flag passthrough tests
   - mpiexec support test
   - Cluster collection tests

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Add MPI command wrapping to KVCacheBenchmark | 41519f0 | mlpstorage/benchmarks/kvcache.py |
| 2 | Add unit tests for MPI execution | 44f6b53 | tests/unit/test_benchmarks_kvcache.py |

## Technical Details

### Import Changes

```python
from mlpstorage.config import (
    BENCHMARK_TYPES,
    EXEC_TYPE,  # NEW
    KVCACHE_MODELS,
    ...
)
from mlpstorage.utils import generate_mpi_prefix_cmd  # NEW
```

### __init__ Changes

```python
def __init__(self, args, ...):
    super().__init__(args, ...)

    # Store num_processes for MPI execution
    self.num_processes = getattr(args, 'num_processes', None)

    # Collect cluster information for distributed runs
    if getattr(args, 'command', '') == 'run':
        self.cluster_information = self._collect_cluster_information()
```

### _build_kvcache_command Changes

```python
def _build_kvcache_command(self) -> str:
    # ... build cmd_parts ...

    # Build the base command
    cmd = " ".join(cmd_parts)

    # Add MPI wrapper if distributed execution requested
    exec_type = getattr(self.args, 'exec_type', None)
    if exec_type == EXEC_TYPE.MPI:
        hosts = getattr(self.args, 'hosts', None)
        if hosts and len(hosts) > 0:
            # Default num_processes to number of hosts if not specified
            num_procs = self.num_processes or len(hosts)
            mpi_prefix = generate_mpi_prefix_cmd(
                mpi_cmd=getattr(self.args, 'mpi_bin', 'mpirun'),
                hosts=hosts,
                num_processes=num_procs,
                oversubscribe=getattr(self.args, 'oversubscribe', False),
                allow_run_as_root=getattr(self.args, 'allow_run_as_root', False),
                params=getattr(self.args, 'mpi_params', None),
                logger=self.logger
            )
            cmd = f"{mpi_prefix} {cmd}"

    return cmd
```

### Generated Command Examples

**Local execution (no MPI):**
```bash
/usr/bin/python3 kv-cache.py --model llama3.1-8b --num-users 100 ...
```

**Distributed execution with MPI:**
```bash
mpirun -n 4 -host host1:2,host2:2 --bind-to none --map-by node /usr/bin/python3 kv-cache.py --model llama3.1-8b ...
```

## Verification Results

All verification criteria met:

1. KVCacheBenchmark._build_kvcache_command() generates MPI-wrapped command when:
   - exec_type == EXEC_TYPE.MPI: VERIFIED
   - hosts list is provided and non-empty: VERIFIED

2. MPI command includes:
   - Correct mpi_bin (mpirun or mpiexec): VERIFIED
   - Host list: VERIFIED
   - Process count (-n): VERIFIED
   - Optional flags (--oversubscribe, --allow-run-as-root): VERIFIED

3. Local execution still works when exec_type is not MPI: VERIFIED

4. All unit tests pass:
```
tests/unit/test_benchmarks_kvcache.py ... 12 passed in 0.21s
tests/unit/test_cli_kvcache.py ... 38 passed
Total: 50 passed in 0.28s
```

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

**Decision 1: Follow DLIOBenchmark pattern**
- **Context:** Need to add MPI wrapping to kv-cache.py command
- **Choice:** Use same pattern as DLIOBenchmark.generate_dlio_command()
- **Rationale:** Consistent behavior, proven pattern, reuses generate_mpi_prefix_cmd
- **Impact:** KV cache and training/checkpointing benchmarks have identical MPI behavior

**Decision 2: num_processes defaults to len(hosts)**
- **Context:** What happens when --num-processes is not specified?
- **Choice:** Default to number of hosts
- **Rationale:** One process per host is a sensible default
- **Impact:** Users can omit --num-processes for simple distributed runs

**Decision 3: Cluster collection for run only**
- **Context:** When to collect cluster information?
- **Choice:** Only for 'run' command, not 'datasize'
- **Rationale:** Datasize is a calculation, doesn't need cluster info
- **Impact:** Faster datasize execution, no unnecessary network calls

## Integration Points

**Upstream Dependencies:**
- `mlpstorage.config.EXEC_TYPE` (enum for mpi/docker)
- `mlpstorage.utils.generate_mpi_prefix_cmd`
- `mlpstorage.benchmarks.base.Benchmark._collect_cluster_information`

**Downstream Consumers:**
- KVCacheBenchmark._execute_run (uses generated command)
- Future result collection and metrics

**API Contract:**
```python
# KVCacheBenchmark instance attributes
benchmark.num_processes      # int or None
benchmark.cluster_information  # ClusterInformation or None (for run command)

# Command generation
benchmark._build_kvcache_command()
# Returns: str with or without MPI prefix based on args
```

## Next Phase Readiness

**Blockers:** None

**Concerns:**
- kvcache not yet wired into cli_parser.py (out of scope, noted in 03-01)
- Need to test actual MPI execution in integration tests

**Ready for 03-03:** Result collection and metrics

## Files Created/Modified

### Created
- `tests/unit/test_benchmarks_kvcache.py` (+371 lines)
  - TestKVCacheMPIExecution (8 tests)
  - TestKVCacheClusterCollection (2 tests)
  - TestKVCacheNumProcessesStorage (2 tests)

### Modified
- `mlpstorage/benchmarks/kvcache.py` (+31 lines)
  - Added EXEC_TYPE import
  - Added generate_mpi_prefix_cmd import
  - Added num_processes storage in __init__
  - Added cluster collection for run command
  - Added MPI wrapping in _build_kvcache_command

## Testing Notes

All 12 tests pass:

```
TestKVCacheMPIExecution::test_local_execution_no_mpi_wrapper PASSED
TestKVCacheMPIExecution::test_docker_execution_no_mpi_wrapper PASSED
TestKVCacheMPIExecution::test_mpi_execution_adds_wrapper PASSED
TestKVCacheMPIExecution::test_mpi_execution_empty_hosts_no_wrapper PASSED
TestKVCacheMPIExecution::test_mpi_execution_defaults_num_processes_to_host_count PASSED
TestKVCacheMPIExecution::test_mpi_execution_oversubscribe_flag PASSED
TestKVCacheMPIExecution::test_mpi_execution_allow_run_as_root_flag PASSED
TestKVCacheMPIExecution::test_mpi_execution_uses_mpiexec PASSED
TestKVCacheClusterCollection::test_cluster_collection_called_for_run_command PASSED
TestKVCacheClusterCollection::test_cluster_collection_not_called_for_datasize_command PASSED
TestKVCacheNumProcessesStorage::test_num_processes_stored_from_args PASSED
TestKVCacheNumProcessesStorage::test_num_processes_none_when_not_provided PASSED
```

Test approach:
- Mock `generate_output_location` to control output directory
- Mock `_collect_cluster_information` to avoid actual MPI calls
- Verify command string contents for MPI wrapper presence/absence
- Verify MPI flags are included when specified

## Lessons Learned

**What Went Well:**
- DLIOBenchmark pattern was easy to follow
- generate_mpi_prefix_cmd already handles all the complexity
- Test fixtures from test_benchmarks_base.py provided good template

**For Future Plans:**
- VectorDB benchmark may need similar MPI support if distributed
- Pattern is now well-established for non-DLIO benchmarks
- Consider adding integration tests for actual MPI execution

## Performance Notes

Execution time: ~3 minutes (171 seconds)

Tasks: 2 completed in 2 commits

Commits:
- 41519f0: feat(03-02): add MPI execution support to KVCacheBenchmark
- 44f6b53: test(03-02): add unit tests for KVCacheBenchmark MPI execution

---

**Summary created:** 2026-01-24T03:34:54Z
**Executor:** Claude Opus 4.5
**Status:** Complete
