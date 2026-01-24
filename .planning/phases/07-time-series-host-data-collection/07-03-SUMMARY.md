---
phase: 07-time-series-host-data-collection
plan: 03
subsystem: benchmark-base
tags:
  - time-series
  - cli-arguments
  - benchmark-integration
  - file-output
  - host-04
  - host-05
requires:
  - TimeSeriesCollector
  - MultiHostTimeSeriesCollector
  - TimeSeriesData dataclass
  - TimeSeriesSample dataclass
provides:
  - CLI arguments: --timeseries-interval, --skip-timeseries, --max-timeseries-samples
  - Benchmark.run() time-series integration
  - Time-series JSON file output
  - Metadata time-series reference
affects:
  - HOST-04 requirement (time-series data files)
  - HOST-05 requirement (background collection)
  - All benchmark run commands
tech-stack:
  added: []
  patterns:
    - CLI argument builder pattern for time-series
    - Benchmark.run() lifecycle integration
    - Background thread collection with try/finally cleanup
    - JSON file output with MLPSJsonEncoder
decisions:
  - id: timeseries-default-interval
    choice: Default 10-second interval, configurable via --timeseries-interval
    rationale: Balances granularity with collection overhead
  - id: timeseries-run-only
    choice: Only collect time-series for 'run' command, not datagen/configview
    rationale: Time-series only meaningful during actual benchmark execution
  - id: timeseries-skip-whatif
    choice: Skip time-series collection in what-if mode
    rationale: No actual execution happens, so no point collecting metrics
  - id: timeseries-try-finally
    choice: Use try/finally to ensure cleanup even if _run() fails
    rationale: Always stop collector and write data regardless of benchmark outcome
key-files:
  created: []
  modified:
    - mlpstorage/cli/common_args.py
    - mlpstorage/cli/training_args.py
    - mlpstorage/cli/checkpointing_args.py
    - mlpstorage/cli/vectordb_args.py
    - mlpstorage/cli/kvcache_args.py
    - mlpstorage/benchmarks/base.py
    - tests/unit/test_benchmarks_base.py
metrics:
  duration: ~12 minutes
  completed: 2026-01-24
---

# Phase 07 Plan 03: Benchmark Time-Series Integration Summary

**One-liner:** CLI arguments for time-series control, Benchmark.run() integration with background collection, and JSON file output following {benchmark_type}_{datetime}_timeseries.json naming convention

## What Was Built

1. **CLI Arguments**: Added --timeseries-interval, --skip-timeseries, --max-timeseries-samples to common_args.py

2. **CLI Wiring**: Integrated add_timeseries_arguments into all benchmark run parsers (training, checkpointing, vectordb, kvcache)

3. **Benchmark Base Integration**: Added _should_collect_timeseries(), _start_timeseries_collection(), _stop_timeseries_collection(), write_timeseries_data() methods

4. **Run Method Updates**: Integrated time-series lifecycle around _run() with try/finally for guaranteed cleanup

5. **Metadata Integration**: Added timeseries_data reference to metadata property with file path, sample count, interval, and hosts

6. **Comprehensive Unit Tests**: 19 tests covering all time-series integration scenarios

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Add time-series CLI arguments and wire into all benchmark parsers | 27484b9 | common_args.py, training_args.py, checkpointing_args.py, vectordb_args.py, kvcache_args.py |
| 2 | Integrate time-series collection into Benchmark base class | 9726915 | mlpstorage/benchmarks/base.py |
| 3 | Add unit tests for time-series benchmark integration | c0c33f5 | tests/unit/test_benchmarks_base.py |

## Technical Details

### CLI Arguments

```python
def add_timeseries_arguments(parser):
    """Add time-series collection arguments."""
    timeseries_group = parser.add_argument_group("Time-Series Collection")
    timeseries_group.add_argument(
        '--timeseries-interval',
        type=float,
        default=10.0,
        help="Interval in seconds between samples"
    )
    timeseries_group.add_argument(
        '--skip-timeseries',
        action='store_true',
        help="Disable time-series collection"
    )
    timeseries_group.add_argument(
        '--max-timeseries-samples',
        type=int,
        default=3600,
        help="Maximum samples per host"
    )
```

### Benchmark Base Methods

```python
class Benchmark:
    # Instance variables
    _timeseries_collector = None
    _timeseries_data = None
    timeseries_filename = "{type}_{datetime}_timeseries.json"
    timeseries_file_path = "{output_dir}/{filename}"

    def _should_collect_timeseries(self) -> bool
    def _start_timeseries_collection(self) -> None
    def _stop_timeseries_collection(self) -> None
    def write_timeseries_data(self) -> None
```

### Updated run() Method

```python
def run(self) -> int:
    self._validate_environment()
    self._collect_cluster_start()
    self._start_timeseries_collection()  # Start background thread

    start_time = time.time()
    try:
        result = self._run()
    finally:
        self.runtime = time.time() - start_time
        self._stop_timeseries_collection()  # Stop and store results
        self._collect_cluster_end()
        self.write_timeseries_data()  # Write JSON file

    return result
```

### Metadata Time-Series Reference

```python
metadata['timeseries_data'] = {
    'file': self.timeseries_filename,
    'num_samples': self._timeseries_data.num_samples,
    'interval_seconds': self._timeseries_data.collection_interval_seconds,
    'hosts_collected': self._timeseries_data.hosts_collected,
}
```

## Verification Results

All verification criteria met:

1. CLI args check: `'timeseries_interval' in HELP_MESSAGES` = True
2. CLI wiring check: 5 files contain add_timeseries_arguments
3. Benchmark integration check: All methods exist on Benchmark class
4. Unit tests: 19 passed
5. Full test suite: 171 passed (98 cluster_collector + 73 benchmark_base)

### Must-Haves Verification

**Truths:**
- User can specify --timeseries-interval: VERIFIED (in add_timeseries_arguments)
- User can specify --skip-timeseries: VERIFIED (in add_timeseries_arguments)
- User can specify --max-timeseries-samples: VERIFIED (in add_timeseries_arguments)
- Time-series starts before _run() and stops after: VERIFIED (run() method)
- Time-series file written to results directory: VERIFIED (write_timeseries_data)
- Metadata includes timeseries_data reference: VERIFIED (metadata property)
- File follows naming convention: VERIFIED (training_20260124_120000_timeseries.json)

**Artifacts:**
- common_args.py provides --timeseries-interval: VERIFIED
- training_args.py contains add_timeseries_arguments: VERIFIED
- checkpointing_args.py contains add_timeseries_arguments: VERIFIED
- vectordb_args.py contains add_timeseries_arguments: VERIFIED
- kvcache_args.py contains add_timeseries_arguments: VERIFIED
- base.py provides _start_timeseries_collection: VERIFIED
- test_benchmarks_base.py contains test_timeseries tests: VERIFIED

**Key Links:**
- base.py uses MultiHostTimeSeriesCollector: VERIFIED
- base.py uses TimeSeriesData: VERIFIED
- base.py writes timeseries.json: VERIFIED
- All *_args.py call add_timeseries_arguments for run parser: VERIFIED

## Deviations from Plan

None - plan executed exactly as written.

## Decisions Made

**Decision 1: Default 10-second interval**
- **Context:** Need a sensible default for collection frequency
- **Choice:** 10.0 seconds as default, configurable via CLI
- **Rationale:** Balances data granularity with minimal collection overhead
- **Impact:** Typical 1-hour benchmark produces 360 samples

**Decision 2: Only collect for 'run' command**
- **Context:** datagen and configview don't execute actual benchmarks
- **Choice:** Check command in _should_collect_timeseries()
- **Rationale:** Time-series only meaningful during actual benchmark execution
- **Impact:** No time-series overhead for non-benchmark commands

**Decision 3: Skip in what-if mode**
- **Context:** what-if mode doesn't actually execute commands
- **Choice:** Check what_if flag in _should_collect_timeseries()
- **Rationale:** No point collecting metrics when nothing is running
- **Impact:** Faster what-if preview

**Decision 4: Use try/finally for cleanup**
- **Context:** Need to ensure collector stops even if _run() fails
- **Choice:** Wrap _run() in try/finally block
- **Rationale:** Always stop collector, collect end cluster info, write data
- **Impact:** Robust cleanup regardless of benchmark outcome

## Integration Points

**Upstream Dependencies:**
- TimeSeriesCollector (07-01) - single-host collection
- MultiHostTimeSeriesCollector (07-02) - multi-host collection
- TimeSeriesData dataclass (07-01) - data structure
- TimeSeriesSample dataclass (07-01) - sample structure

**Downstream Consumers:**
- All benchmark run commands get time-series collection automatically
- Results directories include time-series JSON files
- Metadata includes time-series data reference for result processing

## Files Changed

### Modified

**mlpstorage/cli/common_args.py**
- Added time-series help messages to HELP_MESSAGES dict
- Added add_timeseries_arguments function

**mlpstorage/cli/training_args.py**
- Added add_timeseries_arguments import
- Called add_timeseries_arguments(run_benchmark) for run parser

**mlpstorage/cli/checkpointing_args.py**
- Added add_timeseries_arguments import
- Called add_timeseries_arguments(run_benchmark) for run parser

**mlpstorage/cli/vectordb_args.py**
- Added add_timeseries_arguments import
- Called add_timeseries_arguments(run_benchmark) for run parser

**mlpstorage/cli/kvcache_args.py**
- Added add_timeseries_arguments import
- Called add_timeseries_arguments in _add_kvcache_distributed_arguments

**mlpstorage/benchmarks/base.py**
- Added TimeSeriesData, TimeSeriesSample, TimeSeriesCollector, MultiHostTimeSeriesCollector imports
- Added _timeseries_collector, _timeseries_data instance variables
- Added timeseries_filename, timeseries_file_path instance variables
- Added _should_collect_timeseries() method
- Added _start_timeseries_collection() method
- Added _stop_timeseries_collection() method
- Added write_timeseries_data() method
- Updated metadata property to include timeseries_data reference
- Updated run() method with try/finally and time-series lifecycle

**tests/unit/test_benchmarks_base.py**
- Added time import
- Added TestTimeSeriesCollectionIntegration class with 19 tests
- Updated test_tracks_runtime to mock new time-series methods

## Testing Notes

Test execution results:
```
TestTimeSeriesCollectionIntegration::test_should_collect_timeseries_default_true PASSED
TestTimeSeriesCollectionIntegration::test_should_collect_timeseries_skip_flag PASSED
TestTimeSeriesCollectionIntegration::test_should_collect_timeseries_datagen_disabled PASSED
TestTimeSeriesCollectionIntegration::test_should_collect_timeseries_whatif_disabled PASSED
TestTimeSeriesCollectionIntegration::test_start_timeseries_creates_collector PASSED
TestTimeSeriesCollectionIntegration::test_start_timeseries_multihost_with_hosts PASSED
TestTimeSeriesCollectionIntegration::test_start_timeseries_singlehost_without_hosts PASSED
TestTimeSeriesCollectionIntegration::test_stop_timeseries_creates_data PASSED
TestTimeSeriesCollectionIntegration::test_stop_timeseries_multihost_creates_data PASSED
TestTimeSeriesCollectionIntegration::test_write_timeseries_creates_file PASSED
TestTimeSeriesCollectionIntegration::test_timeseries_file_follows_naming_convention PASSED
TestTimeSeriesCollectionIntegration::test_metadata_includes_timeseries_reference PASSED
TestTimeSeriesCollectionIntegration::test_run_integrates_timeseries_collection PASSED
TestTimeSeriesCollectionIntegration::test_timeseries_uses_background_thread PASSED
TestTimeSeriesCollectionIntegration::test_timeseries_multihost_uses_correct_thread_name PASSED
TestTimeSeriesCollectionIntegration::test_timeseries_skipped_for_datagen_command PASSED
TestTimeSeriesCollectionIntegration::test_timeseries_skipped_for_configview_command PASSED
TestTimeSeriesCollectionIntegration::test_timeseries_stop_without_start_noop PASSED
TestTimeSeriesCollectionIntegration::test_write_timeseries_without_data_noop PASSED
```

All 19 new tests pass, total 73 tests in test_benchmarks_base.py.

Full suite: 171 tests pass (98 cluster_collector + 73 benchmark_base).

## Lessons Learned

**What Went Well:**
- MultiHostTimeSeriesCollector from 07-02 integrated cleanly
- try/finally pattern ensures robust cleanup
- Existing test patterns made new tests straightforward

**For Future Reference:**
- Time-series integration adds ~187 lines to base.py
- Default 10-second interval appropriate for most benchmarks
- max_timeseries_samples=3600 allows 10 hours at default interval

## Phase 07 Complete

This plan completes Phase 07 (Time-Series Host Data Collection):
- 07-01: Core time-series infrastructure (TimeSeriesSample, TimeSeriesData, TimeSeriesCollector)
- 07-02: Multi-host collection (MultiHostTimeSeriesCollector, parallel SSH)
- 07-03: Benchmark integration (CLI arguments, run() lifecycle, file output)

**HOST-04 Requirement:** Time-series data files with proper naming convention - COMPLETE
**HOST-05 Requirement:** Background collection with minimal performance impact - COMPLETE

---

**Summary created:** 2026-01-24
**Executor:** Claude Opus 4.5
**Status:** Complete
