# MLPerf Storage v3.0 Roadmap

## Overview

This roadmap delivers MLPerf Storage v3.0 with fully integrated KV cache and VectorDB benchmarks, new training models, SSH-based host collection with time-series data, and improved package management. Phases are ordered for the 6-week feature freeze timeline, with foundational work first to unblock parallel development.

---

## Phase 1: Package Management Foundation

**Goal:** Users can install and verify reproducible package configurations without GPU dependencies.

**Dependencies:** None (foundational)

**Requirements:**
- PKG-01: Lockfile for Python dependencies with pinned versions
- PKG-02: Remove GPU package dependencies from default install
- PKG-03: Validate package versions match lockfile before benchmark execution

**Success Criteria:**
1. User can generate a lockfile from current environment (`mlpstorage lockfile generate`)
2. User can install exact versions from lockfile and see matching verification pass
3. User installing default package does not pull GPU-related dependencies (torch-cuda, etc.)
4. Benchmark execution fails with clear message when package versions differ from lockfile

**Plans:** 5 plans

Plans:
- [x] 01-01-PLAN.md - Lockfile module structure and data models
- [x] 01-02-PLAN.md - CPU-only pyproject.toml configuration
- [x] 01-03-PLAN.md - Lockfile generation with uv pip compile
- [x] 01-04-PLAN.md - Runtime version validation
- [x] 01-05-PLAN.md - CLI integration and benchmark hookup

---

## Phase 2: Environment Validation and Fail-Fast

**Goal:** Users receive clear, actionable guidance when environment is misconfigured.

**Dependencies:** Phase 1 (package validation integrates with lockfile)

**Requirements:**
- UX-01: Detect missing commands/packages with actionable error messages
- UX-02: Suggest installation steps for missing dependencies
- UX-03: Validate environment before benchmark execution (fail-fast)

**Success Criteria:**
1. User running benchmark without MPI installed sees error with installation instructions for their detected OS
2. User running benchmark without DLIO sees error with `pip install` command to resolve
3. User with missing host tools (SSH, etc.) sees specific error before benchmark starts, not cryptic failure mid-run
4. All validation occurs before any benchmark execution begins (fail-fast pattern)

**Plans:** 5 plans

Plans:
- [x] 02-01-PLAN.md - OS detection module and install hints
- [x] 02-02-PLAN.md - Enhanced dependency checking with OS-aware errors
- [x] 02-03-PLAN.md - SSH connectivity validation and issue collection
- [x] 02-04-PLAN.md - Comprehensive fail-fast environment validator
- [x] 02-05-PLAN.md - Integration into benchmark execution path

---

## Phase 3: KV Cache Benchmark Integration

**Goal:** Users can run KV cache benchmarks through the unified CLI with standard reporting.

**Dependencies:** Phase 2 (fail-fast validation)

**Requirements:**
- BENCH-01: KVCacheBenchmark class extending Benchmark base (wraps kv-cache.py)
- BENCH-02: KV cache MPI execution across multiple hosts

**Success Criteria:**
1. User can run `mlpstorage kvcache run` and see benchmark execute with standard result directory structure
2. User can run KV cache benchmark across multiple hosts using MPI with `--hosts` argument
3. KV cache benchmark generates metadata JSON file consistent with training/checkpointing benchmarks
4. User can view KV cache benchmark in `mlpstorage history list` output

**Plans:** 3 plans

Plans:
- [x] 03-01-PLAN.md - Add distributed execution CLI arguments
- [x] 03-02-PLAN.md - MPI command wrapping in KVCacheBenchmark
- [x] 03-03-PLAN.md - Metadata verification and history integration

---

## Phase 4: VectorDB Benchmark Integration

**Goal:** Users can run VectorDB benchmarks through the unified CLI with data generation support.

**Dependencies:** Phase 2 (fail-fast validation)

**Requirements:**
- BENCH-03: VectorDBBenchmark class extending Benchmark base (wraps VDB scripts)
- BENCH-04: VectorDB CLI commands (run, datagen operations)

**Success Criteria:**
1. User can run `mlpstorage vectordb run` and see benchmark execute with standard result directory structure
2. User can run `mlpstorage vectordb datagen` to generate test data for VectorDB benchmarks
3. VectorDB benchmark generates metadata JSON file consistent with other benchmark types
4. User can view VectorDB benchmark in `mlpstorage history list` output

**Plans:** 3 plans

Plans:
- [x] 04-01-PLAN.md - CLI command naming consistency (run-search to run)
- [x] 04-02-PLAN.md - Metadata enhancement for history integration
- [x] 04-03-PLAN.md - Unit tests for VectorDB CLI and benchmark

---

## Phase 5: Benchmark Validation Pipeline Integration

**Goal:** Users can validate and report on KV cache and VectorDB results using standard tooling.

**Dependencies:** Phase 3 (KV Cache), Phase 4 (VectorDB)

**Requirements:**
- BENCH-05: Integration with existing validation/reporting pipeline

**Success Criteria:**
1. User can run `mlpstorage reports reportgen` on KV cache results and see validation output
2. User can run `mlpstorage reports reportgen` on VectorDB results and see validation output
3. Validation rules for KV cache and VectorDB benchmarks produce CLOSED/OPEN/INVALID categories
4. Combined reports can include all benchmark types (training, checkpointing, kvcache, vectordb)

**Plans:** 3 plans

Plans:
- [x] 05-01-PLAN.md - Create VectorDBRunRulesChecker
- [x] 05-02-PLAN.md - Update BenchmarkVerifier routing and formatters
- [x] 05-03-PLAN.md - Unit tests for VectorDB validation

---

## Phase 6: SSH-Based Host Collection

**Goal:** Users can collect host information for non-MPI benchmarks via SSH.

**Dependencies:** Phase 3 (KV Cache), Phase 4 (VectorDB) - SSH collection needed for these benchmarks

**Requirements:**
- HOST-01: SSH-based host collection for non-MPI benchmarks
- HOST-02: Collect /proc/ data (diskstats, vmstat, cpuinfo, filesystems, cgroups)
- HOST-03: Collection at benchmark start and end

**Success Criteria:**
1. User running non-MPI benchmark with `--hosts` sees cluster information collected via SSH
2. User sees /proc/ data (diskstats, vmstat, cpuinfo) in cluster info JSON output
3. User sees filesystem and cgroup information in cluster info for storage analysis
4. Benchmark metadata includes host collection snapshots from start and end of execution

**Plans:** 3 plans

Plans:
- [x] 06-01-PLAN.md - New /proc parsers (vmstat, mounts, cgroups)
- [x] 06-02-PLAN.md - SSHClusterCollector implementation
- [x] 06-03-PLAN.md - Benchmark integration with start/end snapshots

---

## Phase 7: Time-Series Host Data Collection

**Goal:** Users can collect time-series host metrics during benchmark execution without performance impact.

**Dependencies:** Phase 6 (SSH collection infrastructure)

**Requirements:**
- HOST-04: Time-series collection (10 sec intervals) during execution
- HOST-05: Parallel collection process without benchmark performance impact

**Success Criteria:**
1. User sees time-series data files in result directory with samples at 10-second intervals
2. Time-series data includes diskstats, vmstat metrics evolving over benchmark duration
3. User can verify collection process runs in parallel (separate process/thread)
4. Benchmark performance metrics show no measurable degradation with collection enabled vs disabled

**Plans:** 3 plans

Plans:
- [ ] 07-01-PLAN.md - Core time-series dataclasses and single-host collector
- [ ] 07-02-PLAN.md - Multi-host time-series collection with SSH
- [ ] 07-03-PLAN.md - Benchmark integration and CLI arguments

---

## Phase 8: New Training Models

**Goal:** Users can run dlrm, retinanet, and flux training benchmarks with full validation support.

**Dependencies:** Phase 2 (fail-fast validation for new dependencies)

**Requirements:**
- TRAIN-01: Add dlrm model configuration
- TRAIN-02: Add retinanet model configuration
- TRAIN-03: Add flux model configuration

**Success Criteria:**
1. User can run `mlpstorage training run --model dlrm` with appropriate accelerator configs
2. User can run `mlpstorage training run --model retinanet` with appropriate accelerator configs
3. User can run `mlpstorage training run --model flux` with appropriate accelerator configs
4. All three new models have validation rules for CLOSED/OPEN submission categories
5. User can generate data for all three models via `mlpstorage training datagen`

---

## Phase 9: DLIO Parquet Support

**Goal:** Users can use parquet format for training data with full DLIO integration.

**Dependencies:** Phase 8 (new models may benefit from parquet support)

**Requirements:**
- TRAIN-04: Update DLIO to support parquet for data loaders, readers, data generation

**Success Criteria:**
1. User can specify `--format parquet` for training data generation
2. User can run training benchmarks reading parquet-format datasets
3. Data generation for parquet produces valid files readable by DLIO
4. Configuration files support parquet format specification alongside existing formats

---

## Phase 10: Progress Indication

**Goal:** Users see clear progress feedback during long-running operations.

**Dependencies:** All phases (applies broadly to CLI)

**Requirements:**
- UX-04: Clear progress indication during long operations

**Success Criteria:**
1. User sees progress bar or percentage during data generation operations
2. User sees elapsed time and estimated remaining time for benchmark execution
3. User sees clear stage indicators (e.g., "Collecting cluster info...", "Running benchmark...", "Writing results...")
4. Progress indication works in both interactive terminal and redirected output modes

---

## Progress

| Phase | Name | Status | Requirements |
|-------|------|--------|--------------|
| 1 | Package Management Foundation | Complete | PKG-01, PKG-02, PKG-03 |
| 2 | Environment Validation and Fail-Fast | Complete | UX-01, UX-02, UX-03 |
| 3 | KV Cache Benchmark Integration | Complete | BENCH-01, BENCH-02 |
| 4 | VectorDB Benchmark Integration | Complete | BENCH-03, BENCH-04 |
| 5 | Benchmark Validation Pipeline Integration | Complete | BENCH-05 |
| 6 | SSH-Based Host Collection | Complete | HOST-01, HOST-02, HOST-03 |
| 7 | Time-Series Host Data Collection | Not Started | HOST-04, HOST-05 |
| 8 | New Training Models | Not Started | TRAIN-01, TRAIN-02, TRAIN-03 |
| 9 | DLIO Parquet Support | Not Started | TRAIN-04 |
| 10 | Progress Indication | Not Started | UX-04 |

---

*Roadmap created: 2026-01-23*
