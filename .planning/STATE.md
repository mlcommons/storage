# MLPerf Storage v3.0 - Project State

## Project Reference

**Core Value:** Orchestrate multiple benchmark types (training, checkpointing, kv-cache, vectordb) across distributed systems and produce verified, rules-compliant results.

**Current Focus:** Phase 3 In Progress - KV Cache Benchmark Integration

## Current Position

**Phase:** 3 of 10 - KV Cache Benchmark Integration
**Plan:** 03-02 of 5 (COMPLETE)
**Status:** In Progress
**Last activity:** 2026-01-24 - Completed 03-02-PLAN.md

**Progress:**
```
Phase 1:  [##########] 100% (5/5 plans) COMPLETE
Phase 2:  [##########] 100% (5/5 plans) COMPLETE
Phase 3:  [####------] 40% (2/5 plans)
Phase 4:  [----------] 0%
Phase 5:  [----------] 0%
Phase 6:  [----------] 0%
Phase 7:  [----------] 0%
Phase 8:  [----------] 0%
Phase 9:  [----------] 0%
Phase 10: [----------] 0%
Overall:  [####------] 44% (12/27 plans complete)
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases completed | 2/10 |
| Requirements delivered | 5/21 (PKG-01, PKG-02, PKG-03, CLI integration, Fail-fast validation) |
| Plans executed | 12 |
| Avg tasks per plan | 2.3 |

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

### Open TODOs

- [x] Complete Phase 1: Package Management Foundation
- [x] Complete Phase 2: Environment Validation and Fail-Fast
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

## Session Continuity

### Last Session
- **Date:** 2026-01-24
- **Accomplished:** Completed 03-02-PLAN.md execution (MPI Execution Support)
- **Next:** Execute 03-03-PLAN.md (Result collection and metrics)

### Context for Next Session
- Phase 3 IN PROGRESS: KV Cache Benchmark Integration
  - 03-01: KV Cache Distributed CLI Arguments COMPLETE
    - Added --hosts, --exec-type, --num-processes, MPI args to run command
    - 38 unit tests in tests/unit/test_cli_kvcache.py
    - Datasize command correctly lacks distributed args
  - 03-02: Multi-host orchestration COMPLETE
    - KVCacheBenchmark._build_kvcache_command() now wraps with MPI prefix
    - Cluster information collected for 'run' command
    - 12 unit tests in tests/unit/test_benchmarks_kvcache.py
  - 03-03: Result collection and metrics (NEXT)
  - 03-04: Validation and error handling
  - 03-05: Integration testing
- Available for downstream use:
  - KV cache run command now accepts: --hosts, --exec-type, --num-processes, --mpi-bin, --oversubscribe, --allow-run-as-root, --mpi-params
  - KVCacheBenchmark generates MPI-wrapped commands when exec_type=MPI
  - num_processes defaults to len(hosts) when not specified
- Note: kvcache not yet wired into cli_parser.py (out of scope)
- No blockers

---

*State updated: 2026-01-24*
