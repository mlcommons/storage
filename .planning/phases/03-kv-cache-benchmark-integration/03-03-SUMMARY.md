---
phase: 03-kv-cache-benchmark-integration
plan: 03
subsystem: benchmark
tags:
  - kvcache
  - metadata
  - history-integration
  - distributed-execution
requires:
  - 03-01 (KV Cache Distributed CLI Arguments)
  - 03-02 (MPI Execution Support)
provides:
  - kvcache-metadata-complete
  - kvcache-history-integration
affects:
  - mlpstorage history list (KV cache runs now appear)
  - Metadata JSON files for KV cache benchmark runs
tech-stack:
  added: []
  patterns:
    - Consistent metadata structure across benchmark types
    - Model field duplication for compatibility (kvcache_model + model)
decisions:
  - id: model-field-consistency
    choice: Add 'model' field in addition to 'kvcache_model' for history compatibility
    rationale: History module and other benchmarks use 'model' field, KV cache retains specific name for clarity
  - id: num-processes-in-metadata
    choice: Include num_processes in metadata even when None
    rationale: Consistent field presence across local and distributed runs
  - id: conditional-distributed-fields
    choice: hosts and exec_type only appear in metadata when set
    rationale: Reduces noise in metadata for local runs
key-files:
  created: []
  modified:
    - mlpstorage/benchmarks/kvcache.py
    - tests/unit/test_benchmarks_kvcache.py
metrics:
  duration: 180 seconds
  completed: 2026-01-24
---

# Phase 03 Plan 03: KV Cache Metadata and History Integration Summary

**One-liner:** KVCacheBenchmark metadata now includes all required fields for history integration and distributed execution tracking

## What Was Built

1. **Enhanced Metadata Structure**:
   - Added `model` field for consistency with other benchmarks
   - Added `num_processes` for distributed run tracking
   - Added `exec_type` when distributed execution is configured
   - Added `hosts` when hosts are specified
   - Retained `kvcache_model` for KV cache specific identification

2. **Metadata Test Suite** (5 tests):
   - Required fields for history module verification
   - KV cache specific fields verification
   - Distributed execution info verification
   - Model field consistency verification
   - Non-distributed metadata verification

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Enhance KV cache metadata structure | 5c75413 | mlpstorage/benchmarks/kvcache.py |
| 2 | Add metadata tests for history integration | e81f864 | tests/unit/test_benchmarks_kvcache.py |
| 3 | End-to-end verification | (no commit) | Verification only |

## Technical Details

### Metadata Property Changes

```python
@property
def metadata(self) -> Dict[str, Any]:
    """Generate metadata for the KV cache benchmark run."""
    base_metadata = super().metadata

    # Add KV cache specific metadata
    base_metadata.update({
        'kvcache_model': self.model,
        'model': self.model,  # Add for consistency with other benchmarks
        'num_users': self.num_users,
        'duration': self.duration,
        'gpu_mem_gb': self.gpu_mem_gb,
        'cpu_mem_gb': self.cpu_mem_gb,
        'cache_dir': self.cache_dir,
        'generation_mode': self.generation_mode,
        'performance_profile': self.performance_profile,
        'num_processes': self.num_processes,  # Include for distributed runs
    })

    # Add execution info for distributed runs
    exec_type = getattr(self.args, 'exec_type', None)
    if exec_type:
        base_metadata['exec_type'] = exec_type.value if hasattr(exec_type, 'value') else str(exec_type)

    hosts = getattr(self.args, 'hosts', None)
    if hosts:
        base_metadata['hosts'] = hosts

    # Add metrics if available
    if hasattr(self, 'metrics'):
        base_metadata['kvcache_metrics'] = self.metrics

    return base_metadata
```

### Example Metadata Output

**Local execution:**
```json
{
  "benchmark_type": "kv_cache",
  "model": "llama3.1-8b",
  "kvcache_model": "llama3.1-8b",
  "command": "run",
  "run_datetime": "20260124_120000",
  "result_dir": "/results/kvcache_llama3.1-8b_run_20260124_120000",
  "num_processes": null,
  "num_users": 100,
  "duration": 60,
  "gpu_mem_gb": 16.0,
  "cpu_mem_gb": 32.0,
  "generation_mode": "realistic",
  "performance_profile": "latency"
}
```

**Distributed execution:**
```json
{
  "benchmark_type": "kv_cache",
  "model": "llama3.1-8b",
  "kvcache_model": "llama3.1-8b",
  "command": "run",
  "run_datetime": "20260124_120000",
  "result_dir": "/results/kvcache_llama3.1-8b_run_20260124_120000",
  "num_processes": 4,
  "num_users": 100,
  "duration": 60,
  "gpu_mem_gb": 16.0,
  "cpu_mem_gb": 32.0,
  "generation_mode": "realistic",
  "performance_profile": "latency",
  "exec_type": "mpi",
  "hosts": ["host1", "host2"]
}
```

## Verification Results

All verification criteria met:

1. KVCacheBenchmark.metadata includes all required fields:
   - benchmark_type, model, command, run_datetime, result_dir: VERIFIED
   - num_processes, hosts, exec_type for distributed runs: VERIFIED
   - kvcache_model, num_users, duration, gpu_mem_gb, cpu_mem_gb: VERIFIED

2. All unit tests pass:
```
tests/unit/test_benchmarks_kvcache.py ... 17 passed in 0.22s
tests/unit/test_cli_kvcache.py ... 38 passed in 0.29s
Total: 55 passed
```

3. Help output shows all distributed execution arguments: VERIFIED
   - --hosts, --exec-type, --num-processes
   - --mpi-bin, --oversubscribe, --allow-run-as-root, --mpi-params

4. CLI integration note: kvcache not yet wired into cli_parser.py (tracked separately)

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

**Decision 1: Model field duplication**
- **Context:** KV cache uses 'kvcache_model' but history module expects 'model'
- **Choice:** Include both fields
- **Rationale:** History compatibility while preserving KV cache specific naming
- **Impact:** Metadata has slight redundancy but works with all consumers

**Decision 2: num_processes always present**
- **Context:** Should num_processes be in metadata only for distributed runs?
- **Choice:** Always include it, value is None for local runs
- **Rationale:** Consistent field presence makes parsing easier
- **Impact:** Field always exists, consumers don't need to check for presence

**Decision 3: Conditional distributed fields**
- **Context:** Should hosts/exec_type be present when not set?
- **Choice:** Only include when values are set
- **Rationale:** Reduces metadata noise for local runs
- **Impact:** Consumers need to check for presence of these fields

## Integration Points

**Upstream Dependencies:**
- `mlpstorage.benchmarks.base.Benchmark.metadata` (base metadata property)
- `mlpstorage.config.EXEC_TYPE` (for exec_type value serialization)

**Downstream Consumers:**
- `mlpstorage.history.HistoryTracker` (reads metadata from result directories)
- Future reporting and analysis tools

**API Contract:**
```python
# KVCacheBenchmark metadata fields
metadata['benchmark_type']      # str: "kv_cache"
metadata['model']               # str: model name (for history compatibility)
metadata['kvcache_model']       # str: model name (KV cache specific)
metadata['command']             # str: "run" or "datasize"
metadata['run_datetime']        # str: YYYYMMDD_HHMMSS
metadata['result_dir']          # str: path to results directory
metadata['num_processes']       # int or None
metadata['num_users']           # int
metadata['duration']            # int (seconds)
metadata['gpu_mem_gb']          # float
metadata['cpu_mem_gb']          # float
metadata['generation_mode']     # str
metadata['performance_profile'] # str

# Optional fields (only for distributed runs)
metadata.get('exec_type')       # str or None
metadata.get('hosts')           # list[str] or None
```

## Next Phase Readiness

**Blockers:** None

**Concerns:**
- kvcache not yet wired into cli_parser.py (out of scope, tracked in STATE.md)
- Full history list verification requires actual runs (verified via metadata structure)

**Ready for 03-04:** Validation and error handling

## Files Created/Modified

### Modified
- `mlpstorage/benchmarks/kvcache.py` (+11 lines)
  - Added 'model' field for consistency
  - Added 'num_processes' to metadata
  - Added conditional 'exec_type' and 'hosts' fields

- `tests/unit/test_benchmarks_kvcache.py` (+153 lines)
  - TestKVCacheMetadata class (5 tests)
    - test_metadata_has_required_fields
    - test_metadata_includes_kvcache_specific_fields
    - test_metadata_includes_distributed_info
    - test_metadata_model_consistency
    - test_metadata_without_distributed_info

## Testing Notes

All 17 KV cache benchmark tests pass:

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
TestKVCacheMetadata::test_metadata_has_required_fields PASSED
TestKVCacheMetadata::test_metadata_includes_kvcache_specific_fields PASSED
TestKVCacheMetadata::test_metadata_includes_distributed_info PASSED
TestKVCacheMetadata::test_metadata_model_consistency PASSED
TestKVCacheMetadata::test_metadata_without_distributed_info PASSED
```

Test approach:
- Mock `generate_output_location` to control output directory
- Mock `_collect_cluster_information` to avoid actual MPI calls
- Verify metadata dictionary contents

## Lessons Learned

**What Went Well:**
- Base class metadata property already included many required fields
- Existing test patterns made adding new tests straightforward
- Clear separation between local and distributed metadata fields

**For Future Plans:**
- VectorDB benchmark may need similar metadata enhancement
- Consider standardizing metadata structure across all benchmark types
- History module integration could be verified with integration tests

## Performance Notes

Execution time: ~3 minutes (180 seconds)

Tasks: 3 completed in 2 commits (Task 3 was verification only)

Commits:
- 5c75413: feat(03-03): enhance KV cache metadata for history integration
- e81f864: test(03-03): add metadata tests for history integration

---

**Summary created:** 2026-01-24
**Executor:** Claude Opus 4.5
**Status:** Complete
