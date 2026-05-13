# Test Suite

This directory contains the full test suite for **mlp-storage v3.0**, covering all
supported workload types: training, checkpointing, KV-cache, and vector-database
benchmarks — on all storage backends (local filesystem, NFS/Lustre, and S3-compatible
object storage via s3dlio, minio, or s3torchconnector).

> **New to the project?** Read [docs/README.md](../docs/README.md) first — it
> explains all four benchmark workloads, the object storage library layer, and
> the full document reference. Then come back here to run tests.

---

## ⚡ Recent Benchmark Results — April 26, 2026

Full end-to-end MLPerf Storage benchmark results on all four supported training and
checkpointing workloads, tested on both POSIX NVMe and S3-compatible object storage
(via [s3-ultra](../s3-ultra/) fake S3 server over loopback).

**Host:** loki-russ · **MPI ranks:** 4 · **Accelerator profile:** B200

| Workload | POSIX NVMe | AU% | S3 Object | AU% | Details |
|----------|-----------|:---:|-----------|:---:|---------|
| **RetinaNet** (250K × 323 KB JPEG, batch 24) | 1,866 s/s | **92.8%** ✅ | 1,919 s/s | **95.4%** ✅ | [RetinaNet_test_results.md](RetinaNet_test_results.md) |
| **Flux** (130 Parquet × 256 samples) | 141 s/s | **99.7%** ✅ | 121 s/s | **85.4%** ⚠️ | [Flux_test_results.md](Flux_test_results.md) |
| **DLRM** (64 Parquet × 1M samples) | 389K s/s | **0.48%** ❌ | 106K s/s | **0.11%** ❌ | [DLRM_test_results.md](DLRM_test_results.md) |
| **Checkpointing** (llama3-8b, NP=4) | write **1.416 GiB/s** | — | write **2.213 GiB/s** | — | [Checkpoint_test_results.md](Checkpoint_test_results.md) |

> **RetinaNet:** b200 AU target ≥ 85% — both POSIX and S3 pass comfortably. Results include
> O_DIRECT verification; see full file for T1–T4 breakdown and bug-fix history.
>
> **Flux:** POSIX meets the ≥ 90% AU target; S3 at 85.4% falls slightly short due to
> per-request latency overhead on loopback HTTP. Real object storage would close this gap.
>
> **DLRM:** AU target is 70%, but both runs fail due to near-zero compute time (0.375 ms/step) —
> the workload is overwhelmingly I/O bound. A production submission requires high-bandwidth
> parallel storage to sustain the ~9 MB/step demand at accelerator speed.
>
> **Checkpointing:** No AU metric; throughput shows S3 multipart write (32 MB parts, 16 in-flight)
> **exceeds** local NVMe write speed thanks to pipelining. Read is network-limited at 8.4 GiB/s.

---

## Quick Start for New Users

### Step 1 — Clone and set up the environment

```bash
git clone https://github.com/mlcommons/storage.git mlp-storage
cd mlp-storage
uv sync
```

[`uv`](https://docs.astral.sh/uv/) creates and manages the virtual environment
automatically — no manual `venv` or `pip` steps required. If `uv` is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To include test and full extras:

```bash
uv sync --all-extras
```

> **Already cloned / returning user?** Just run `uv sync` again after pulling — it
> is idempotent and fast. It updates the environment to match `uv.lock` automatically.
>
> Confirm the installed version:
> ```bash
> uv run python -c "import mlpstorage_py; print(mlpstorage_py.VERSION)"
> # Should print: 3.0.0
> ```

### Step 2 — Run the unit tests (no infrastructure required)

```bash
uv run pytest tests/unit/
```

Expected output: all tests pass in a few seconds. No MinIO, no MPI, no GPU required.
These tests mock all external dependencies.

```
==================== XX passed in X.XXs ====================
```

If you see import errors, run `uv sync --all-extras` and retry.

### Step 3 — (Optional) Run integration tests with object storage

Integration and object-store tests require a running S3-compatible endpoint (MinIO,
Ceph, Vast, etc.). Set credentials in a `.env` file at the **project root** — this
file is gitignored and should never be committed:

```bash
# mlp-storage/.env  (copy from .env.example if present, or create manually)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_ENDPOINT_URL=http://your-host:9000
AWS_REGION=us-east-1
```

Shell environment variables take precedence over `.env` values if both are set.

Then run:

```bash
# Confirm endpoint is reachable (standalone script — use python, not pytest)
python tests/integration/test_s3_connectivity.py \
    --libraries s3dlio minio \
    --s3dlio-bucket mlp-s3dlio \
    --minio-bucket mlp-minio

pytest tests/integration/ -v                          # full integration suite
```

### Step 4 — (Optional) Run object-store performance tests

```bash
# Quick functional test — 3 NPZ files, all three libraries
./tests/object-store/test_mlp_s3dlio.sh
./tests/object-store/test_mlp_minio.sh
./tests/object-store/test_mlp_s3torch.sh

# Cross-library throughput benchmark
python tests/object-store/test_direct_write_comparison.py
```

---

## How pytest is configured

pytest is pre-configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short --setup-show"
```

This means **`pytest` (with no arguments) from the project root runs all tests** in
`tests/` that match `test_*.py`. The `-v --tb=short --setup-show` flags are always
active, so you will see fixture setup/teardown and short tracebacks on failure.

Useful overrides:

```bash
pytest tests/unit/                          # unit tests only
pytest tests/integration/                  # integration tests only
pytest tests/unit/test_benchmarks_kvcache.py -v   # single file
pytest tests/unit/ -k "kvcache"            # tests matching a keyword
pytest tests/unit/ --tb=long              # full tracebacks on failure
pytest tests/unit/ -x                      # stop at first failure
pytest tests/unit/ --cov=mlpstorage --cov-report=term-missing  # coverage
```

---

## Shared test infrastructure

### `tests/__init__.py`

Empty file that makes `tests/` a Python package. This is required so that
`conftest.py` can do `from tests.fixtures import ...`. Do not delete it.

### `tests/conftest.py`

Defines shared pytest fixtures available to **all** test files automatically — no
import needed. pytest discovers and injects these by name.

Key fixtures provided:

| Fixture | Type | What it provides |
|---------|------|-----------------|
| `mock_logger` | `MagicMock` | Captures log calls (`info`, `warning`, `error`, etc.) |
| `capturing_logger` | `MockLogger` | Returns `(logger, messages_dict)` for assertion |
| `test_logger` | `MockLogger` | Full `MockLogger` instance from fixtures package |
| `mock_executor` | `MockCommandExecutor` | Replaces subprocess calls — no real commands run |
| `mock_executor_with_dlio` | `MockCommandExecutor` | Pre-configured with DLIO success responses |
| `mock_executor_failure` | `MockCommandExecutor` | Pre-configured to simulate failures |
| `mock_collector` | `MockClusterCollector` | Replaces MPI cluster collection |
| `mock_collector_multi_host` | `MockClusterCollector` | 4-host, 64-core, 256 GB config |
| `mock_collector_failure` | `MockClusterCollector` | Simulates MPI unavailable |
| `base_args` | `Namespace` | Minimal CLI args shared by all commands |
| `training_run_args` | `Namespace` | Full args for a training `run` command |
| `checkpointing_args` | `Namespace` | Full args for a checkpointing `run` command |
| `fixtures_dir` | `Path` | Path to `tests/fixtures/` |
| `sample_results_dir` | `Path` | Path to `tests/fixtures/sample_results/` |

### `tests/fixtures/`

Python package (`__init__.py` + 4 modules) with the underlying mock classes:

| Module | Provides |
|--------|---------|
| `mock_logger.py` | `MockLogger`, `create_mock_logger()` |
| `mock_executor.py` | `MockCommandExecutor` — intercepts subprocess/shell calls |
| `mock_collector.py` | `MockClusterCollector` — fake MPI cluster info |
| `sample_data.py` | `SAMPLE_MEMINFO`, `SAMPLE_CPUINFO`, `SAMPLE_DISKSTATS`, `SAMPLE_HOSTS`, factory functions |

These are imported by `conftest.py` and re-exported as fixtures. Individual test files
can also import them directly if they need finer control.

---

## Directory Structure

```
tests/
├── unit/            # Fast, no-infrastructure pytest unit tests
├── integration/     # Integration tests (may need live storage / MPI)
├── object-store/    # Object storage performance tests and demos
├── checkpointing/   # Streaming checkpoint tests and demos
├── configs/         # YAML configs and S3 testing guides
├── fixtures/        # Shared mock classes and sample data (used by conftest.py)
│   ├── __init__.py
│   ├── mock_logger.py
│   ├── mock_executor.py
│   ├── mock_collector.py
│   ├── sample_data.py
│   └── sample_results/   # Sample JSON result files for rules/reporting tests
├── __init__.py      # Makes tests/ a package (required — do not delete)
├── conftest.py      # Shared pytest fixtures (auto-loaded by pytest)
└── README.md        # This file
```

---

## 1. Unit Tests (`tests/unit/`)

Fast, self-contained tests requiring no external infrastructure. Run with pytest.

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run a specific module
pytest tests/unit/test_benchmarks_kvcache.py -v
```

### Coverage by module

| File | What it tests |
|------|--------------|
| `test_benchmark_run.py` | `BenchmarkRun` construction (from_benchmark, from_result_dir, from_data) |
| `test_benchmarks_base.py` | `Benchmark` base class initialization |
| `test_benchmarks_kvcache.py` | `KVCacheBenchmark` — MPI command generation, distributed execution |
| `test_benchmarks_vectordb.py` | `VectorDBBenchmark` — command method map, subcommands |
| `test_cli.py` | CLI argument parsing — training commands |
| `test_cli_kvcache.py` | CLI argument parsing — KV cache model and cache configuration |
| `test_cli_vectordb.py` | CLI argument parsing — VectorDB run/datagen subcommands |
| `test_cluster_collector.py` | Cluster metric collection |
| `test_config.py` | Config module, env var handling, `DEFAULT_RESULTS_DIR` env-var override |
| `test_dlio_object_storage.py` | `DLIOBenchmark._apply_object_storage_params()` — `.env` loading, param injection, error cases |
| `test_main_warnings.py` | `run_benchmark()` tempdir warning — fires/suppresses correctly |
| `test_dependency_check.py` | Dependency checking logic |
| `test_environment.py` | Environment detection and validation |
| `test_history.py` | `HistoryTracker` — run history file management |
| `test_imports.py` | Package import sanity checks |
| `test_progress.py` | Progress reporting |
| `test_reporting.py` | `ReportGenerator` — result dataclasses and output formatting |
| `test_rules_calculations.py` | Rules calculations — training data size, memory/step math |
| `test_rules_checkers.py` | `RulesChecker` base class |
| `test_rules_dataclasses.py` | Rules dataclasses |
| `test_rules_extractors.py` | Rules result extractors |
| `test_rules_vectordb.py` | `VectorDBRunRulesChecker` — benchmark type validation |
| `test_utils.py` | Utility function tests |
| `test_validation_helpers.py` | Validation helper functions |

---

## 2. Integration Tests (`tests/integration/`)

End-to-end tests that exercise real storage backends, DLIO, and MPI. Most require
the virtual environment and may require a running object store or MPI installation.

```bash
# Benchmark execution flow (mock dependencies — no live storage needed)
pytest tests/integration/test_benchmark_flow.py -v

# Full submission validation
pytest tests/integration/test_full_submission.py -v

# S3 connectivity (standalone script — use python, not pytest; requires object storage endpoint)
python tests/integration/test_s3_connectivity.py \
    --libraries s3dlio minio \
    --s3dlio-bucket mlp-s3dlio \
    --minio-bucket mlp-minio

# Multi-endpoint selection logic (no live storage needed)
python tests/integration/test_multi_endpoint.py

# Multi-endpoint integration (requires object storage)
pytest tests/integration/test_multi_endpoint_integration.py -v

# DLIO storage layer with file:// URIs (verifies zero-copy)
python tests/integration/test_dlio_storage.py

# s3dlio compatibility layer
python tests/integration/test_compat.py
python tests/integration/test_compat_runtime.py

# MPI basic smoke test
python tests/integration/test_mpi_basic.py

# DLIO + MPI together
python tests/integration/test_dlio_mpi.py

# A/B comparison: MLP vs dpsi implementations
python tests/integration/test_ab_comparison.py
```

### Benchmark scripts (non-pytest)

```bash
# Raw write throughput: compare s3dlio, minio, s3torchconnector side-by-side
python tests/integration/benchmark_write_comparison.py

# Raw read throughput comparison
python tests/integration/benchmark_read_comparison.py

# s3dlio-specific write and read benchmarks
python tests/integration/benchmark_s3dlio_write.py
python tests/integration/benchmark_s3dlio_read.py

# Parquet byte-range read example
python tests/integration/parquet_byte_range_example.py

# Generate test data (NPZ/HDF5/TFRecord)
python tests/integration/generate_test_data.py

# Verify s3dlio installation and basic operation
python tests/integration/verify_s3dlio.py
```

---

## 3. Object Storage Tests (`tests/object-store/`)

Performance and correctness tests for the three supported object storage libraries:
**s3dlio**, **minio**, and **s3torchconnector**. See
[tests/object-store/README.md](object-store/README.md) for full documentation and
benchmark results.

### Cross-library throughput comparison

```bash
cd /home/eval/Documents/Code/mlp-storage && source .venv/bin/activate

# Native API write + read — all three libraries, default 100 × 128 MiB, 8 workers
python tests/object-store/test_direct_write_comparison.py

# 12-worker run (matches Object_Perf_Results.md baseline)
python tests/object-store/test_direct_write_comparison.py \
    --num-files 100 --size-mb 128 --write-workers 12 --read-workers 12

# Single library only
python tests/object-store/test_direct_write_comparison.py --library s3dlio
```

### DLIO-driven training and checkpoint workloads

```bash
# Training (100 × 128 MiB NPZ, 2 epochs, all libraries)
python tests/object-store/test_dlio_multilib_demo.py --workload training

# Streaming checkpoint (~105 GB, llama3-8b profile)
python tests/object-store/test_dlio_multilib_demo.py --workload checkpoint

# Single library
python tests/object-store/test_dlio_multilib_demo.py --workload training --library s3dlio
```

### MPI process-count sweep

```bash
# Sweep N=1,2,4 × all libraries for datagen + training throughput
python tests/object-store/test_training_mpi_sweep.py
```

### Per-library shell test scripts

```bash
# Quick end-to-end: generate 3 NPZ files and read them back
./tests/object-store/test_mlp_s3dlio.sh
./tests/object-store/test_mlp_minio.sh
./tests/object-store/test_mlp_s3torch.sh

# Multi-library demo via one script
./tests/object-store/test_s3dlio_multilib.sh
```

### Checkpoint-specific object store tests

```bash
python tests/object-store/test_s3dlio_checkpoint.py
python tests/object-store/test_minio_checkpoint.py
python tests/object-store/test_s3torch_checkpoint.py
python tests/object-store/test_s3dlio_direct.py    # zero-copy direct I/O path
```

### Reference

- **[Object_Perf_Results.md](object-store/Object_Perf_Results.md)** — Full benchmark
  results: native API throughput, DLIO streaming checkpoint (16 GB / 100 GB), MPI sweep
- **[bench-results-retinanet-20260425.md](object-store/bench-results-retinanet-20260425.md)** — April 25, 2026: write_threads sweep for RetinaNet on s3-ultra (loopback), NP=1
- **[s3ultra-test-results-20260425.md](object-store/s3ultra-test-results-20260425.md)** — April 25, 2026: s3-ultra end-to-end test results
- **[scaling-analysis-2026-04-25.md](object-store/scaling-analysis-2026-04-25.md)** — April 25, 2026: NP scaling analysis across storage backends
- **[NPZ-OPTIMIZATION-ANALYSIS.md](object-store/NPZ-OPTIMIZATION-ANALYSIS.md)** — NPZ read optimization analysis
- **[dlio_mpi_object_results.md](object-store/dlio_mpi_object_results.md)** — March 20, 2026: DLIO + MPI scaling results (UNet3D h100 profile, ~23.5 GB dataset, NP=1/2/4)
- **[s3dlio_performance_analysis.md](object-store/s3dlio_performance_analysis.md)** — March 20, 2026 HISTORICAL: root-cause analysis of s3dlio performance (6 findings; most resolved in v0.9.84)
- **[S3library_review_21-Mar.md](object-store/S3library_review_21-Mar.md)** — March 21, 2026: prefetch fairness review across all three libraries (analysis only; no code changes)

---

## 4. Checkpointing Tests (`tests/checkpointing/`)

Tests and demos for the `StreamingCheckpointing` feature — streaming checkpoint
writes with dramatically reduced memory overhead and multi-backend support.

### Streaming backend validation

```bash
# Validate all three backends: s3dlio, minio, s3torchconnector (default: 32 GB)
python tests/checkpointing/test_streaming_backends.py

# Quick validation (100 MB)
python tests/checkpointing/test_streaming_backends.py --size 0.1

# Specific backends only
python tests/checkpointing/test_streaming_backends.py --backends s3dlio minio

# Large-scale test
python tests/checkpointing/test_streaming_backends.py --size 64 --max-in-flight 32
```

### Demo scripts

```bash
# Demonstrate StreamingCheckpointing + dgen-py integration
# Shows: old vs new methods, file and object storage, multi-endpoint config
TEST_CHECKPOINT_DIR=/tmp/checkpoints ./tests/object-store/demo_streaming_checkpoint.sh

# 24 GB full comparison (matches PR testing)
TEST_SIZE_GB=24 TEST_CHECKPOINT_DIR=/tmp/checkpoints \
    ./tests/object-store/demo_streaming_checkpoint.sh

# Simple comparison of checkpoint optimization strategies
./tests/checkpointing/demo_checkpoint_methods.sh

# Custom size
OUTPUT_DIR=/data/test SIZE_GB=10 ./tests/checkpointing/demo_checkpoint_methods.sh
```

`tests/checkpointing/compare_methods.py` is the Python backend called by
`demo_checkpoint_methods.sh`.

---

## 5. Test Configs (`tests/configs/`)

YAML benchmark configurations for DLIO-driven S3 testing:

| File | Purpose |
|------|---------|
| `s3_test_mlp_s3dlio.yaml` | s3dlio backend config (unet3d dataset) |
| `s3_test_mlp_minio.yaml` | minio backend config |
| `s3_test_mlp_s3torchconnector.yaml` | s3torchconnector backend config |
| `s3_test_dpsi.yaml` | dpsi (bucket+key) baseline config |
| [S3_TESTING_GUIDE.md](configs/S3_TESTING_GUIDE.md) | Architecture comparison and setup guide |
| [S3_TEST_RESULTS.md](configs/S3_TEST_RESULTS.md) | Recorded test results |

---

## 6. Workload Reference

mlp-storage supports the following workload types, each exercised by the tests above:

| Workload | CLI command | Test files |
|----------|------------|------------|
| Training (DLIO) | `mlpstorage run training` | `unit/test_cli.py`, `integration/test_benchmark_flow.py`, `object-store/test_dlio_multilib_demo.py` |
| Checkpointing | `mlpstorage run checkpointing` | `checkpointing/test_streaming_backends.py`, `object-store/test_*_checkpoint.py` |
| KV Cache | `mlpstorage run kvcache` | `unit/test_benchmarks_kvcache.py`, `unit/test_cli_kvcache.py` |
| Vector DB | `mlpstorage run vectordb` | `unit/test_benchmarks_vectordb.py`, `unit/test_cli_vectordb.py`, `unit/test_rules_vectordb.py` |

Storage backends tested:

| Backend | Type | Notes |
|---------|------|-------|
| `file://` | Local / NFS / Lustre | Default; no extra config needed |
| `direct://` via **s3dlio** | Local (O_DIRECT) | Bypasses page cache entirely; use `storage_type=direct_fs` |
| `s3://` via **s3dlio** | Object storage | High-performance, multi-endpoint |
| `s3://` via **minio** | Object storage | Python minio client |
| `s3://` via **s3torchconnector** | Object storage | AWS reference implementation |

---

## 7. Checkpoint Performance Results

Full-stack checkpoint benchmark results using `mlpstorage checkpointing run` with the
**llama3-8b** model profile (`num_layers=24`), 8 MPI ranks, 1 checkpoint write + 1
checkpoint read.  Aggregate throughput reported by `[METRIC]` lines from the benchmark.

**Test date:** March 19, 2026

### Hardware / Network context

| Component | Details |
|-----------|---------|
| Network (object storage) | 10 Gbit Ethernet — **max ~1.25 GB/s** (network-limited) |
| Local storage (`local_fs`) | VMDK on remote vSAN — **max ~2 GB/s** |
| Checkpoint size | ~82 GB total (8 ranks × ~10.25 GB/rank: model + optimizer) |
| Page-cache bypass | `POSIX_FADV_DONTNEED` per chunk + `POSIX_FADV_RANDOM` at open — reads hit storage, not DRAM |

### Aggregate throughput

| Backend | Write (GB/s) | Read (GB/s) | Notes |
|---------|:-----------:|:-----------:|-------|
| `minio` | 1.04 | 1.09 | Network-limited (10 GbE cap ~1.25 GB/s) |
| `s3torchconnector` | 1.05 | 1.11 | Network-limited |
| `s3dlio` | 1.03 | 1.22 | Network-limited; best read (range-GET concurrency) |
| `local_fs` (fadvise) | **1.42** | **1.82** | vSAN-limited; fadvise(DONTNEED) page-cache bypass |
| `direct_fs` (O_DIRECT) | **1.36** | **1.48** | O_DIRECT via s3dlio `direct://`; hard page-cache bypass |

### Key observations

- **Object store backends** (minio, s3torchconnector, s3dlio) are all bottlenecked by the
  10 GbE network link (~1.25 GB/s ceiling). Their write results cluster tightly at
  1.03–1.05 GB/s. Read throughput varies slightly due to range-GET concurrency differences.
- **s3dlio** achieves the best read among object-store backends (1.22 GB/s) thanks to
  parallel chunk fetching via byte-range GETs.
- **local_fs** bypasses the network entirely, reaching 1.42 GB/s write and 1.82 GB/s read
  against the remote vSAN backing store (practical ceiling ~2 GB/s for that device).
- **Page-cache bypass** is critical for accurate storage benchmarking.  Without it, the
  kernel caches written checkpoint data in DRAM and subsequent reads are served from memory
  (~20 GB/s) rather than the storage device, invalidating the measurement.  Two approaches
  are provided:
  - `local_fs` — `POSIX_FADV_RANDOM` at open (disables readahead) + `POSIX_FADV_DONTNEED`
    after each chunk (soft hint; kernel reclaims asynchronously). Achieved 1.42 W / 1.82 R GB/s.
  - `direct_fs` — O_DIRECT via s3dlio's `direct://` URI; the kernel page cache is bypassed
    entirely at the syscall level, giving the most rigorous measurement. Achieved 1.36 W / 1.48 R GB/s.
    The ~6% write and ~19% read gap versus `local_fs` is expected: O_DIRECT forces synchronous,
    unbuffered I/O through the block layer, while fadvise still allows the kernel I/O scheduler
    to batch and merge requests efficiently.

### Reproducing the file-backend result

```bash
cd /home/eval/Documents/Code/mlp-storage
source .venv/bin/activate

mlpstorage checkpointing run \
  --model llama3-8b --num-processes 8 \
  --client-host-memory-in-gb 64 \
  --num-checkpoints-write 1 --num-checkpoints-read 1 \
  --checkpoint-folder /mnt/nvme_data/llama3-8b-file \
  --allow-run-as-root --oversubscribe --open --skip-timeseries \
  --params storage.storage_type=local_fs model.num_layers=24
```

Expected output (look for `[METRIC]` lines at the end):

```
[METRIC] Checkpoint save I/O Throughput (GB/second): 1.4152 (0.0000)
[METRIC] Checkpoint load I/O Throughput (GB/second): 1.8159 (0.0000)
```

### Reproducing the O_DIRECT result (`direct_fs`)

Uses s3dlio’s `direct://` URI to open files with `O_DIRECT`, completely bypassing
the kernel page cache at the syscall level — the most rigorous measurement of
raw storage throughput.

```bash
cd /home/eval/Documents/Code/mlp-storage
source .venv/bin/activate

mlpstorage checkpointing run \
  --model llama3-8b --num-processes 8 \
  --client-host-memory-in-gb 64 \
  --num-checkpoints-write 1 --num-checkpoints-read 1 \
  --checkpoint-folder /mnt/nvme_data/llama3-8b-direct \
  --allow-run-as-root --oversubscribe --open --skip-timeseries \
  --params storage.storage_type=direct_fs model.num_layers=24
```

> **Note:** `num_layers=24` reduces the checkpoint from the default ~105 GB to ~82 GB to
> fit on the 98 GB test partition. Adjust `--checkpoint-folder` to a location with
> sufficient free space before running.
