# MLPerf Storage v3.0 - Project State

## Project Reference

**Core Value:** Orchestrate multiple benchmark types (training, checkpointing, kv-cache, vectordb) across distributed systems and produce verified, rules-compliant results.

**Current Focus:** Phase 1 - Package Management Foundation

## Current Position

**Phase:** 1 of 10 - Package Management Foundation
**Plan:** 01-05 of 5 (COMPLETE)
**Status:** Phase complete
**Last activity:** 2026-01-23 - Completed 01-05-PLAN.md

**Progress:**
```
Phase 1:  [██████████] 100% (5/5 plans) ✓ COMPLETE
Phase 2:  [----------] 0%
Phase 3:  [----------] 0%
Phase 4:  [----------] 0%
Phase 5:  [----------] 0%
Phase 6:  [----------] 0%
Phase 7:  [----------] 0%
Phase 8:  [----------] 0%
Phase 9:  [----------] 0%
Phase 10: [----------] 0%
Overall:  [██--------] 19% (4/21 requirements - PKG-01, PKG-02, PKG-03 complete)
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Phases completed | 1/10 |
| Requirements delivered | 4/21 (PKG-01, PKG-02, PKG-03, CLI integration) |
| Plans executed | 5 |
| Avg tasks per plan | 2.0 |

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

### Open TODOs

- [x] Complete Phase 1: Package Management Foundation
- [ ] Plan Phase 2 with `/gsd:plan-phase 2`
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

## Session Continuity

### Last Session
- **Date:** 2026-01-23
- **Accomplished:** Completed 01-05-PLAN.md execution (CLI Integration) - Phase 1 COMPLETE
- **Next:** Plan and execute Phase 2 (Fail-Fast Dependency Validation)

### Context for Next Session
- Phase 1 complete: Package Management Foundation
  - Lockfile generation with uv (PKG-01) ✓
  - CPU-only PyTorch config (PKG-02) ✓
  - Runtime version validation (PKG-03) ✓
  - Full CLI integration ✓
- Available commands:
  - `mlpstorage lockfile generate` - Create lockfiles from pyproject.toml
  - `mlpstorage lockfile verify` - Validate installed packages
  - `--verify-lockfile PATH` - Available on all benchmark commands
- Ready for Phase 2: Fail-fast dependency validation can leverage lockfile infrastructure
- No blockers

---

*State updated: 2026-01-23*
