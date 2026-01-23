# MLPerf Storage v3.0 Requirements

## v1 Requirements

### Package Management

- [ ] **PKG-01**: Lockfile for Python dependencies with pinned versions
- [ ] **PKG-02**: Remove GPU package dependencies from default install
- [ ] **PKG-03**: Validate package versions match lockfile before benchmark execution

### Benchmark Integration

- [ ] **BENCH-01**: KVCacheBenchmark class extending Benchmark base (wraps kv-cache.py)
- [ ] **BENCH-02**: KV cache MPI execution across multiple hosts
- [ ] **BENCH-03**: VectorDBBenchmark class extending Benchmark base (wraps VDB scripts)
- [ ] **BENCH-04**: VectorDB CLI commands (run, datagen operations)
- [ ] **BENCH-05**: Integration with existing validation/reporting pipeline

### Training Updates

- [ ] **TRAIN-01**: Add dlrm model configuration
- [ ] **TRAIN-02**: Add retinanet model configuration
- [ ] **TRAIN-03**: Add flux model configuration
- [ ] **TRAIN-04**: Update DLIO to support parquet for data loaders, readers, data generation

### Host Collection

- [ ] **HOST-01**: SSH-based host collection for non-MPI benchmarks
- [ ] **HOST-02**: Collect /proc/ data (diskstats, vmstat, cpuinfo, filesystems, cgroups)
- [ ] **HOST-03**: Collection at benchmark start and end
- [ ] **HOST-04**: Time-series collection (10 sec intervals) during execution
- [ ] **HOST-05**: Parallel collection process without benchmark performance impact

### Error Handling & UX

- [ ] **UX-01**: Detect missing commands/packages with actionable error messages
- [ ] **UX-02**: Suggest installation steps for missing dependencies
- [ ] **UX-03**: Validate environment before benchmark execution (fail-fast)
- [ ] **UX-04**: Clear progress indication during long operations

---

## v2 Requirements (Deferred)

- [ ] Deeper KV cache integration (native implementation vs wrapper)
- [ ] Deeper VectorDB integration (native implementation vs wrapper)
- [ ] Real-time monitoring dashboard for time-series data
- [ ] Cloud provider integrations (AWS, GCP, Azure)

---

## Out of Scope

- **GPU support** — Storage benchmark, deliberately not supporting GPU execution
- **Rewriting KV/VDB as native benchmarks** — v3.0 wraps existing scripts
- **Real-time visualization** — Collection only, no visualization in v3.0
- **Windows support** — Linux-only target

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PKG-01 | TBD | Pending |
| PKG-02 | TBD | Pending |
| PKG-03 | TBD | Pending |
| BENCH-01 | TBD | Pending |
| BENCH-02 | TBD | Pending |
| BENCH-03 | TBD | Pending |
| BENCH-04 | TBD | Pending |
| BENCH-05 | TBD | Pending |
| TRAIN-01 | TBD | Pending |
| TRAIN-02 | TBD | Pending |
| TRAIN-03 | TBD | Pending |
| TRAIN-04 | TBD | Pending |
| HOST-01 | TBD | Pending |
| HOST-02 | TBD | Pending |
| HOST-03 | TBD | Pending |
| HOST-04 | TBD | Pending |
| HOST-05 | TBD | Pending |
| UX-01 | TBD | Pending |
| UX-02 | TBD | Pending |
| UX-03 | TBD | Pending |
| UX-04 | TBD | Pending |

---
*Last updated: 2026-01-23*
