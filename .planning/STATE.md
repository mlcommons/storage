# MLPerf Storage v3.0 - Project State

## Project Reference

**Core Value:** Orchestrate multiple benchmark types (training, checkpointing, kv-cache, vectordb) across distributed systems and produce verified, rules-compliant results.

**Current Focus:** Phase 4 - VectorDB Benchmark Integration (Plan 01 Complete)

## Current Position

**Phase:** 4 of 10 - VectorDB Benchmark Integration
**Plan:** 04-01 of 3 (COMPLETE)
**Status:** In progress
**Last activity:** 2026-01-24 - Completed 04-01-PLAN.md (VectorDB CLI rename)

**Progress:**
```
Phase 1:  [##########] 100% (5/5 plans) COMPLETE
Phase 2:  [##########] 100% (5/5 plans) COMPLETE
Phase 3:  [##########] 100% (3/3 plans) COMPLETE
Phase 4:  [###-------] 33% (1/3 plans)
Phase 5:  [----------] 0%
Phase 6:  [----------] 0%
Phase 7:  [----------] 0%
Phase 8:  [----------] 0%
Phase 9:  [----------] 0%
Phase 10: [----------] 0%
Overall:  [######----] 58% (14/24 plans complete)
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases completed | 3/10 |
| Requirements delivered | 7/21 (PKG-01, PKG-02, PKG-03, UX-01, UX-02, UX-03, BENCH-01, BENCH-02) |
| Plans executed | 14 |
| Avg tasks per plan | 2.4 |

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

### Technical Patterns Established

- Benchmark base class pattern: Subclass, set BENCHMARK_TYPE, implement _run()
- Registry pattern for CLI construction
- MPI-based cluster collection existing, SSH to be added
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

### Open TODOs

- [x] Complete Phase 1: Package Management Foundation
- [x] Complete Phase 2: Environment Validation and Fail-Fast
- [x] Complete Phase 3: KV Cache Benchmark Integration
- [ ] Review external KV cache code in `kv_cache_benchmark/`
- [ ] Review VectorDB scripts from external branch
- [ ] Verify DLIO parquet support requirements
- [ ] Wire kvcache into cli_parser.py (noticed during 03-01)

### Active Blockers

None currently.

### Notes

- 6-week feature freeze timeline
- Existing KVCacheBenchmark class exists but needs full integration
- VectorDBBenchmark class exists as stub, needs implementation
- MPI collection works, SSH collection needs to be added
- Environment module now provides OS-aware install hints
- Dependency checking now uses environment module for OS-specific error messages
- validate_benchmark_environment now collects all issues before reporting
- Fail-fast validation integrated into main.py before benchmark instantiation
- KV cache CLI arguments now support distributed execution (run command only)
- KVCacheBenchmark now wraps commands with MPI prefix when exec_type=MPI
- KV cache metadata includes all required fields for history integration

## Session Continuity

### Last Session
- **Date:** 2026-01-24
- **Accomplished:** Completed 04-01-PLAN.md (VectorDB CLI rename from run-search to run)
- **Next:** Continue Phase 4 with 04-02-PLAN.md

### Context for Next Session
- Phase 4 IN PROGRESS: VectorDB Benchmark Integration
  - 04-01: VectorDB CLI Command Rename COMPLETE
    - Renamed 'run-search' to 'run' for CLI consistency
    - Updated CLI args, help messages, command_method_map
    - Updated tests to match new command name
    - 6 VectorDB tests pass
  - 04-02: Metadata and History Integration PENDING
  - 04-03: [TBD] PENDING
- Available for downstream use:
  - `mlpstorage vectordb run` command (consistent with other benchmarks)
  - `mlpstorage vectordb datagen` command
  - VectorDBBenchmark routes 'run' to execute_run method
- No blockers

---

*State updated: 2026-01-24*
