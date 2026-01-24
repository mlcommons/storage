# MLPerf Storage v3.0 - Project State

## Project Reference

**Core Value:** Orchestrate multiple benchmark types (training, checkpointing, kv-cache, vectordb) across distributed systems and produce verified, rules-compliant results.

**Current Focus:** Phase 6 COMPLETE - SSH-Based Host Collection

## Current Position

**Phase:** 6 of 10 - SSH-Based Host Collection
**Plan:** 06-03 of 3 (COMPLETE)
**Status:** Phase complete
**Last activity:** 2026-01-24 - Completed 06-03-PLAN.md (Benchmark Base Integration)

**Progress:**
```
Phase 1:  [##########] 100% (5/5 plans) COMPLETE
Phase 2:  [##########] 100% (5/5 plans) COMPLETE
Phase 3:  [##########] 100% (3/3 plans) COMPLETE
Phase 4:  [##########] 100% (3/3 plans) COMPLETE
Phase 5:  [##########] 100% (3/3 plans) COMPLETE
Phase 6:  [##########] 100% (3/3 plans) COMPLETE
Phase 7:  [----------] 0%
Phase 8:  [----------] 0%
Phase 9:  [----------] 0%
Phase 10: [----------] 0%
Overall:  [########--] 85% (22/26 plans complete)
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases completed | 6/10 |
| Requirements delivered | 11/21 (PKG-01, PKG-02, PKG-03, UX-01, UX-02, UX-03, BENCH-01, BENCH-02, BENCH-03, BENCH-04, BENCH-05, HOST-03) |
| Plans executed | 22 |
| Avg tasks per plan | 2.5 |

## Accumulated Context

### Key Decisions

| Decision | Rationale | Date |
|----------|-----------|------|
| 10 phases for comprehensive depth | Requirements naturally cluster into 10 delivery boundaries given dependencies | 2026-01-23 |
| Package management first | Foundation for reproducibility, enables fail-fast validation | 2026-01-23 |
| UX validation before benchmarks | Fail-fast pattern reduces wasted time on misconfigured environments | 2026-01-23 |
| KV/VDB benchmarks parallel tracks | Both can develop after fail-fast validation, no interdependency | 2026-01-23 |
| Progress indication last | Applies broadly, benefits from all other work being done | 2026-01-23 |
| Use dataclasses for lockfile models | Python dataclasses for LockedPackage, ValidationResult, LockfileMetadata | 2026-01-23 |
| Parse requirements.txt format | Support pip-compile/uv requirements.txt format for lockfiles | 2026-01-23 |
| Normalize package names | Store package names as lowercase in dict keys for case-insensitive lookups | 2026-01-23 |
| Wrap uv via subprocess | Use subprocess to invoke uv pip compile for lockfile generation | 2026-01-23 |
| GenerationOptions dataclass | Type-safe configuration for lockfile generation parameters | 2026-01-23 |
| Check uv availability first | Verify uv installed before generation, provide helpful error messages | 2026-01-23 |
| Use importlib.metadata for validation | Runtime version checking via importlib.metadata.version() | 2026-01-23 |
| Skip mpi4py validation | mpi4py must match system MPI, version validation doesn't apply | 2026-01-23 |
| Skip VCS dependencies | git+, hg+, svn+ URLs don't have comparable versions | 2026-01-23 |
| Nested subcommands for lockfile | Use mlpstorage lockfile generate/verify pattern | 2026-01-23 |
| Universal arguments for --verify-lockfile | Add flag to add_universal_arguments for all benchmarks | 2026-01-23 |
| Validate before benchmark instantiation | Fail fast before collecting cluster info | 2026-01-23 |
| Show copy-paste commands in errors | Include pip install and uv pip sync in error messages | 2026-01-23 |
| OSInfo dataclass with optional distro | Simple, type-safe OS info with None for non-Linux distro fields | 2026-01-24 |
| distro package with fallback | Try distro package first, fall back to platform.freedesktop_os_release (3.10+) | 2026-01-24 |
| Tuple key lookup for install hints | Use (dependency, system, distro) tuples with specificity-based fallback | 2026-01-24 |
| Separate enhanced functions for hints | Add new *_with_hints functions alongside legacy functions for backward compatibility | 2026-01-24 |
| Multi-line error templates | Use multi-line templates with clear sections for readability and copy-pasteability | 2026-01-24 |
| ValidationIssue as Exception | Make ValidationIssue inherit from Exception to allow raising directly | 2026-01-24 |
| Localhost skip in SSH checks | Skip SSH checks for localhost/127.0.0.1 (auto-success) | 2026-01-24 |
| BatchMode SSH | Use SSH BatchMode to avoid password prompts in automated checks | 2026-01-24 |
| Collect-all-then-report validation | Collect ALL issues before raising, so users see complete picture | 2026-01-24 |
| First error preserves exception type | Raise first collected error to preserve specific exception type | 2026-01-24 |
| Helper functions for conditional checks | _requires_mpi, _is_distributed_run, _requires_dlio for testable logic | 2026-01-24 |
| Single validation entry point | validate_benchmark_environment replaces validate_pre_run | 2026-01-24 |
| Benchmark validation hook | _validate_environment() in base class for benchmark-specific validation | 2026-01-24 |
| Skip validation flag | --skip-validation for debugging when validation is blocking | 2026-01-24 |
| Distributed args for run only | KV cache run command gets --hosts, --exec-type, --num-processes; datasize does not | 2026-01-24 |
| Reuse common CLI args | KV cache uses add_host_arguments, add_mpi_arguments from common_args | 2026-01-24 |
| EXEC_TYPE.MPI default | KV cache --exec-type defaults to MPI like training/checkpointing | 2026-01-24 |
| MPI wrapper pattern for KV cache | Follow DLIOBenchmark pattern for MPI command wrapping | 2026-01-24 |
| num_processes defaults to len(hosts) | Sensible default - one process per host when not specified | 2026-01-24 |
| Cluster collection for run only | Collect cluster information only for 'run' command, not 'datasize' | 2026-01-24 |
| Model field consistency | Add 'model' field in addition to 'kvcache_model' for history compatibility | 2026-01-24 |
| num_processes always in metadata | Include num_processes in metadata even when None for consistency | 2026-01-24 |
| Conditional distributed fields | hosts and exec_type only appear in metadata when set | 2026-01-24 |
| Rename run-search to run for vectordb | All benchmarks use 'run' subcommand for consistency | 2026-01-24 |
| Config name as model (VectorDB) | Use config_name as 'model' field since VectorDB doesn't have ML models | 2026-01-24 |
| Command-specific metadata (VectorDB) | Include different fields for datagen vs run commands | 2026-01-24 |
| Write metadata for both commands | Both run and datagen write metadata for history tracking | 2026-01-24 |
| Test patterns from KVCache | Follow KVCache test patterns for VectorDB test consistency | 2026-01-24 |
| Preview status always returns OPEN | VectorDB is preview, check_preview_status always returns OPEN | 2026-01-24 |
| Minimum runtime 30 seconds for VectorDB | Matches VECTORDB_DEFAULT_RUNTIME, prevents trivially short runs | 2026-01-24 |
| Base MultiRunRulesChecker for preview benchmarks | KV Cache and VectorDB use base checker for multi-run (no specific submission rules yet) | 2026-01-24 |
| VECTORDB_REQUIREMENTS follows KVCACHE pattern | Consistent preview benchmark documentation structure | 2026-01-24 |
| Follow existing dataclass pattern for /proc parsers | Match HostDiskInfo/HostNetworkInfo pattern with to_dict/from_dict | 2026-01-24 |
| Localhost skip for SSH collection | Use direct local collection for localhost/127.0.0.1/::1 | 2026-01-24 |
| Parallel SSH collection | ThreadPoolExecutor with configurable max_workers for parallel host collection | 2026-01-24 |
| BatchMode SSH for collection | Use BatchMode=yes for non-interactive automated collection | 2026-01-24 |
| SSH for non-MPI benchmarks | Use SSH collection when exec_type is not MPI | 2026-01-24 |
| Start/end cluster snapshots | ClusterSnapshots dataclass for HOST-03 requirement | 2026-01-24 |
| Backward-compatible cluster_information | Set cluster_information from start snapshot for compatibility | 2026-01-24 |

### Technical Patterns Established

- Benchmark base class pattern: Subclass, set BENCHMARK_TYPE, implement _run()
- Registry pattern for CLI construction
- MPI-based cluster collection existing, SSH collection now added
- DLIO as underlying engine for training/checkpointing
- Dataclass-based models for structured data
- Regex-based parsing for lockfile requirements.txt format
- Subprocess wrapping for external CLI tools
- Options dataclass pattern for configurable functions
- Runtime package inspection via importlib.metadata
- Structured validation results with metrics
- Modular CLI argument builder pattern
- Universal arguments for cross-cutting concerns
- Command handler pattern in main.py
- Fail-fast validation before execution
- OS detection with distro fallback pattern
- Specificity-based lookup for OS-specific instructions
- OS-aware error messaging with copy-pasteable commands
- Exception-inheriting dataclass for structured errors
- BatchMode SSH for non-interactive connectivity validation
- Collect-all-then-report validation pattern
- Union type for heterogeneous error collection
- Validation hook pattern in base class
- Distributed argument builder pattern (reuse common args)
- MPI command wrapping pattern (generate_mpi_prefix_cmd)
- Consistent metadata structure across benchmark types
- Consistent CLI subcommand naming (all use 'run' for execution)
- CLI test pattern with fixture-based parser
- Benchmark test pattern with mocked dependencies
- RunRulesChecker inheritance for benchmark-specific validation
- check_* method pattern for auto-discovered validation rules
- Benchmark type routing in BenchmarkVerifier for all 4 types
- Preview benchmark requirements formatting pattern
- RunRulesChecker test pattern with mock logger and valid run fixtures
- Line-by-line /proc file parsing pattern with graceful error handling
- ClusterCollectorInterface implementation pattern (SSHClusterCollector)
- ThreadPoolExecutor for parallel operations
- Localhost detection and optimization pattern
- Collection method selection based on exec_type
- Start/end cluster snapshots for state comparison

### Open TODOs

- [x] Complete Phase 1: Package Management Foundation
- [x] Complete Phase 2: Environment Validation and Fail-Fast
- [x] Complete Phase 3: KV Cache Benchmark Integration
- [x] Complete Phase 4: VectorDB Benchmark Integration
- [x] Complete Phase 5: Benchmark Validation Pipeline Integration
- [x] Complete Phase 6: SSH-Based Host Collection
- [ ] Review external KV cache code in `kv_cache_benchmark/`
- [ ] Review VectorDB scripts from external branch
- [ ] Verify DLIO parquet support requirements
- [ ] Wire kvcache into cli_parser.py (noticed during 03-01)
- [ ] Wire vectordb into cli_parser.py

### Active Blockers

None currently.

### Notes

- 6-week feature freeze timeline
- Existing KVCacheBenchmark class exists but needs full integration
- VectorDBBenchmark class now has metadata property, write_metadata integration, and comprehensive tests
- MPI collection works, SSH collection now integrated into benchmark base class
- Environment module now provides OS-aware install hints
- Dependency checking now uses environment module for OS-specific error messages
- validate_benchmark_environment now collects all issues before reporting
- Fail-fast validation integrated into main.py before benchmark instantiation
- KV cache CLI arguments now support distributed execution (run command only)
- KVCacheBenchmark now wraps commands with MPI prefix when exec_type=MPI
- KV cache metadata includes all required fields for history integration
- VectorDB metadata includes all required fields for history integration
- VectorDB has comprehensive unit tests: 53 CLI tests + 15 benchmark tests = 68 total
- VectorDB rules checker has 12 unit tests for complete validation coverage
- cluster_collector.py now has parsers for /proc/vmstat, /proc/mounts, /proc/cgroups
- 29 unit tests for cluster_collector parsers (from 06-01)
- SSHClusterCollector implemented with parallel collection and localhost optimization
- 33 new unit tests for SSHClusterCollector and _is_localhost (from 06-02)
- SSH collection integrated into Benchmark base class with start/end snapshots (06-03)
- ClusterSnapshots dataclass for HOST-03 requirement (start/end state)
- 15 new tests for collection selection and cluster snapshots (from 06-03)
- Total cluster_collector tests: 62, Total benchmark_base tests: 54

## Session Continuity

### Last Session
- **Date:** 2026-01-24
- **Accomplished:** Completed 06-03-PLAN.md (Benchmark Base Integration)
- **Next:** Ready for Phase 7

### Context for Next Session
- Phase 6 COMPLETE: SSH-Based Host Collection
  - 06-01: /proc Parsers for HOST-02 COMPLETE
    - MountInfo and CgroupInfo dataclasses added
    - parse_proc_vmstat, parse_proc_mounts, parse_proc_cgroups functions added
    - collect_local_system_info updated to include vmstat, mounts, cgroups
    - 29 unit tests added to tests/unit/test_cluster_collector.py
  - 06-02: SSHClusterCollector Implementation COMPLETE
    - _is_localhost helper for localhost detection
    - SSH_COLLECTOR_SCRIPT for remote execution
    - SSHClusterCollector implementing ClusterCollectorInterface
    - Parallel collection via ThreadPoolExecutor
    - 33 new unit tests for SSHClusterCollector
  - 06-03: Benchmark Base Integration COMPLETE
    - ClusterSnapshots dataclass in models.py
    - SSH collection integrated into Benchmark base class
    - _should_use_ssh_collection, _collect_via_ssh, _collect_cluster_start, _collect_cluster_end methods
    - run() calls start/end collection (HOST-03)
    - 15 new unit tests for collection selection
- Available for downstream use:
  - Non-MPI benchmarks automatically use SSH collection
  - ClusterSnapshots captures start/end state
  - Metadata includes cluster_snapshots
- Note: vectordb and kvcache not yet wired into cli_parser.py
- Note: 2 pre-existing test failures in test_rules_calculations.py (unrelated to Phase 6)
- No blockers
- Ready to proceed with Phase 7

---

*State updated: 2026-01-24*
