# MLPerf Storage v3.0 - Project State

## Project Reference

**Core Value:** Orchestrate multiple benchmark types (training, checkpointing, kv-cache, vectordb) across distributed systems and produce verified, rules-compliant results.

**Current Focus:** Phase 2 - Environment Validation and Fail-Fast

## Current Position

**Phase:** 2 of 10 - Environment Validation and Fail-Fast
**Plan:** 02-03 of 5
**Status:** In progress
**Last activity:** 2026-01-24 - Completed 02-03-PLAN.md

**Progress:**
```
Phase 1:  [##########] 100% (5/5 plans) COMPLETE
Phase 2:  [######----] 60% (3/5 plans)
Phase 3:  [----------] 0%
Phase 4:  [----------] 0%
Phase 5:  [----------] 0%
Phase 6:  [----------] 0%
Phase 7:  [----------] 0%
Phase 8:  [----------] 0%
Phase 9:  [----------] 0%
Phase 10: [----------] 0%
Overall:  [###-------] 30% (8/27 plans complete)
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases completed | 1/10 |
| Requirements delivered | 4/21 (PKG-01, PKG-02, PKG-03, CLI integration) |
| Plans executed | 8 |
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

### Open TODOs

- [x] Complete Phase 1: Package Management Foundation
- [x] Complete 02-01: Environment Detection Module
- [x] Complete 02-02: Executable Checking Module
- [x] Complete 02-03: SSH Validation and Issue Collection
- [ ] Complete 02-04: Pre-Run Validation Orchestration
- [ ] Complete 02-05: Integration Tests
- [ ] Review external KV cache code in `kv_cache_benchmark/`
- [ ] Review VectorDB scripts from external branch
- [ ] Verify DLIO parquet support requirements

### Active Blockers

None currently.

### Notes

- 6-week feature freeze timeline
- Existing KVCacheBenchmark class exists but needs full integration
- VectorDBBenchmark class exists as stub, needs implementation
- MPI collection works, SSH collection needs to be added
- Environment module now provides OS-aware install hints
- Dependency checking now uses environment module for OS-specific error messages

## Session Continuity

### Last Session
- **Date:** 2026-01-24
- **Accomplished:** Completed 02-03-PLAN.md execution (SSH Validation and Issue Collection)
- **Next:** Execute 02-04-PLAN.md (Pre-Run Validation Orchestration)

### Context for Next Session
- Phase 2 in progress: Environment Validation and Fail-Fast
  - 02-01: Environment detection module COMPLETE
    - `mlpstorage/environment/os_detect.py` - OSInfo dataclass, detect_os()
    - `mlpstorage/environment/install_hints.py` - INSTALL_INSTRUCTIONS, get_install_instruction()
  - 02-02: Executable checking with OS-aware hints COMPLETE
    - `check_mpi_with_hints()`, `check_dlio_with_hints()`, `check_ssh_available()`
    - Error templates: DEPENDENCY_MPI_MISSING, DEPENDENCY_DLIO_MISSING, DEPENDENCY_SSH_MISSING
  - 02-03: SSH validation and issue collection COMPLETE
    - `mlpstorage/environment/validators.py` - ValidationIssue, validate_ssh_connectivity, collect_validation_issues
    - ValidationIssue is both a dataclass and an Exception (can be raised directly)
    - SSH validation checks binary first, uses BatchMode, skips localhost
  - 02-04: Next up - Pre-run validation orchestration
  - 02-05: Integration tests
- Available for downstream use:
  - `from mlpstorage.environment import detect_os, get_install_instruction, OSInfo`
  - `from mlpstorage.environment import ValidationIssue, validate_ssh_connectivity, collect_validation_issues`
  - `from mlpstorage.dependency_check import check_mpi_with_hints, check_dlio_with_hints, check_ssh_available`
  - Supports Ubuntu, Debian, RHEL, CentOS, Fedora, Arch, macOS, Windows
- No blockers

---

*State updated: 2026-01-24*
