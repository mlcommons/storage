# MLPerf Storage Benchmark Suite v3.0

## What This Is

A benchmark orchestration framework for the MLCommons MLPerf Storage working group. The suite runs storage benchmarks aligned with MLPerf rules and reports results with verification of rules compliance.

## Core Value

**The ONE thing that must work:** Orchestrate multiple benchmark types (training, checkpointing, kv-cache, vectordb) across distributed systems and produce verified, rules-compliant results.

## Context

### Current State
- v2.0 release with Claude Code enhancements
- Training and checkpointing benchmarks use DLIO as underlying engine
- KV cache benchmark exists in separate directory (`kv_cache_benchmark/`)
- VectorDB benchmark code exists in external branch
- MPI-based execution and host collection for DLIO benchmarks
- Existing error handling and validation pipeline

### Target State (v3.0)
- Fully integrated KV cache and VectorDB benchmarks as Benchmark subclasses
- New training models (dlrm, retinanet, flux)
- Package version management with lockfiles
- SSH-based host collection for non-MPI benchmarks
- Time-series /proc/ data collection during benchmark execution
- Improved error messaging and user guidance

### Timeline
- **Feature freeze:** 6 weeks
- **Bugfix period:** 6 weeks
- **Code freeze:** 12 weeks total

## Requirements

### Validated (Existing)

- ✓ Training benchmark orchestration via DLIO — existing
- ✓ Checkpointing benchmark orchestration via DLIO — existing
- ✓ MPI-based distributed execution — existing
- ✓ Rules validation pipeline — existing
- ✓ Report generation — existing
- ✓ CLI with nested subcommands — existing
- ✓ Benchmark registry pattern — existing

### Active

- [ ] Package version lockfile management
- [ ] Remove GPU package dependencies (not used)
- [ ] KV cache Benchmark class (wraps kv-cache.py)
- [ ] KV cache MPI execution across hosts
- [ ] VectorDB Benchmark class (wraps load_vdb.py, compact_and_watch.py, simple_bench.py)
- [ ] SSH-based host collection for non-MPI benchmarks
- [ ] New training models: dlrm, retinanet, flux
- [ ] Improved error messaging for missing commands/packages
- [ ] Clear user guidance for resolving dependency issues
- [ ] Time-series /proc/ collection (diskstats, vmstat, cpuinfo, etc.)
- [ ] Parallel collection process (10 sec intervals) without impacting benchmark

### Out of Scope

- GPU support — deliberately not supporting GPU execution
- Rewriting KV/VDB as native benchmarks — v3.0 wraps existing scripts
- Real-time monitoring UI — collection only, no visualization
- Cloud provider integrations — on-premise/bare-metal focus

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Lockfile for package versions | Reproducibility across systems, MPI version issues | Pending |
| Benchmark subclasses for KV/VDB | Minimal integration, reuse CLI and reporting infrastructure | Pending |
| SSH for non-MPI host collection | KV cache and VectorDB don't require MPI execution | Pending |
| Parallel process for time-series | Must not impact benchmark performance | Pending |

## Constraints

- **No GPU dependencies** — storage benchmark, not compute
- **MPI compatibility** — must work with various MPI implementations
- **Cross-platform** — Linux primarily, various distributions
- **Minimal dependencies** — reduce version conflict surface area

## External Code References

| Component | Location | Notes |
|-----------|----------|-------|
| KV cache benchmark | `kv_cache_benchmark/` (local) | Also: `mlcommons/storage/TF_KVCache` branch |
| VectorDB benchmark | `mlcommons/storage/TF_VDBBench` branch | Scripts: load_vdb.py, compact_and_watch.py, simple_bench.py |
| DLIO benchmark | External package | Upstream dependency for training/checkpointing |

## Success Metrics

- All 4 benchmark types (training, checkpointing, kv-cache, vectordb) runnable from unified CLI
- Package lockfile prevents version conflicts in CI
- Error messages guide users to resolution for common issues
- Host data collected for all benchmark types (MPI or SSH)
- Time-series collection runs without measurable benchmark impact

---
*Last updated: 2026-01-23 after initialization*
