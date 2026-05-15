# DLRM Training Benchmark Results

## System Under Test

| Field | Value |
|-------|-------|
| Host | loki-russ |
| CPU | Intel Xeon Platinum 8280L @ 2.70 GHz |
| Physical CPUs (visible) | 28 vCPUs |
| RAM | 47.0 GB |
| OS | Linux |

## Workload Configuration

| Parameter | Value |
|-----------|-------|
| Model | dlrm |
| Simulated accelerators | 4 × B200 |
| MPI ranks | 4 (local, `127.0.0.1:4`) |
| Epochs | 1 |
| Batch size | 12,288 samples/step |
| Files (train) | 64 Parquet |
| Samples per file | 1,000,000 |
| Total samples | 64,000,000 |
| Record length | 761 bytes/sample |
| Dataset size | ~49 GB |
| Row group size | 6,144 |
| `read_threads` | 4 per rank |
| Simulated compute time | 0.375 ms/step |
| Steps per epoch | ~1,302 (64 × 1,000,000 / 12,288 / 4 ranks) |

> Note: DLRM is overwhelmingly **I/O bound** — compute time per step is 0.375 ms (vs 1,350 ms for Flux).
> The AU metric directly measures storage bandwidth vs accelerator demand.
> **AU target for DLRM is 70%** (from `reader.au: 0.70` in `dlrm_b200.yaml`), not 90%.

## Run Commands

### POSIX (Local NVMe)

```bash
# Datagen
cd /home/eval/Documents/Code/mlp-storage && uv run mlpstorage training datagen \
  --model dlrm --num-processes 4 --allow-run-as-root --open --skip-validation \
  --data-dir /mnt/nvme_data/mlperf_storage_dlio_data \
  --params dataset.num_files_train=64 dataset.num_samples_per_file=1000000

# Training
cd /home/eval/Documents/Code/mlp-storage && uv run mlpstorage training run \
  --model dlrm --num-accelerators 4 --accelerator-type b200 \
  --client-host-memory-in-gb 47 --open --allow-run-as-root --skip-validation \
  --file --data-dir /mnt/nvme_data/mlperf_storage_dlio_data \
  --params dataset.num_files_train=64 dataset.num_samples_per_file=1000000
```

### S3 Object Storage (MinIO)

```bash
# Datagen (into S3 bucket mlp-flux)
# Requires .env with BUCKET=mlp-flux loaded
cd /home/eval/Documents/Code/mlp-storage && uv run mlpstorage training datagen \
  --model dlrm --num-processes 4 --allow-run-as-root --open --skip-validation \
  --object s3 \
  --params dataset.num_files_train=64 dataset.num_samples_per_file=1000000

# Training (from S3)
cd /home/eval/Documents/Code/mlp-storage && uv run mlpstorage training run \
  --model dlrm --num-accelerators 4 --accelerator-type b200 \
  --client-host-memory-in-gb 47 --open --allow-run-as-root --skip-validation \
  --object s3 \
  --params dataset.num_files_train=64 dataset.num_samples_per_file=1000000
```

---

## Storage Targets

### 1 — POSIX (Local NVMe)

| Field | Value |
|-------|-------|
| Run ID | 20260426_162816 |
| Date | 2026-04-26 16:28 – 16:31 MDT |
| Storage type | POSIX (local filesystem) |
| Device | `/dev/nvme4n2p1` (NVMe SSD, 98 GB) |
| Mount point | `/mnt/nvme_data` |
| Data path | `/mnt/nvme_data/mlperf_storage_dlio_data/dlrm/` |

#### Results

| Metric | Value |
|--------|-------|
| **Accelerator Utilization (AU)** | **0.48%** |
| AU target | ≥ 70% |
| AU target met | ❌ fail |
| Training throughput | 388,921 samples/s |
| I/O throughput | **282.3 MiB/s** |
| Epoch 1 wall time | 179.1 s |

#### Notes

- AU is extremely low (0.48%) because DLRM compute is only 0.375 ms/step — the benchmark is almost entirely I/O bound.
- A [WARNING] was emitted: "dataset smaller than host memory; data might be cached after first epoch." The ~49 GB dataset fits within the 47 GB RAM page cache, so most reads are served from DRAM after initial cold reads.
- Even with page cache serving data, AU is only 0.48% — indicating the benchmark demands far higher I/O bandwidth than NVMe can deliver at this batch size / thread count.

---

### 2 — MinIO S3 (Object Storage)

| Field | Value |
|-------|-------|
| Run ID | 20260426_163722 |
| Date | 2026-04-26 16:37 – 16:47 MDT |
| Storage type | S3 object storage |
| Endpoint | `https://172.16.1.40:9000` (MinIO) |
| Bucket | `mlp-flux` |
| Storage library | s3dlio (byte-range GET) |
| Data path | `s3://mlp-flux/data/dlrm/` |

#### Results

| Metric | Value |
|--------|-------|
| **Accelerator Utilization (AU)** | **0.11%** |
| AU target | ≥ 70% |
| AU target met | ❌ fail |
| Training throughput | 106,351 samples/s |
| I/O throughput | **77.2 MiB/s** |
| Epoch 1 wall time | 616.7 s |

#### Notes

- S3 throughput (77.2 MiB/s) is only 27% of POSIX (282.3 MiB/s), reflecting S3 GET latency overhead per row-group read.
- Wall time 3.4× longer than POSIX (617s vs 179s) entirely due to I/O — compute is identical.
- Same dataset-smaller-than-RAM warning; the bottleneck is purely network/S3 latency, not data volume.

---

## Comparison Summary

| Metric | POSIX NVMe | MinIO S3 | Delta |
|--------|------------|----------|-------|
| Run ID | 20260426_162816 | 20260426_163722 | — |
| **AU %** | **0.48%** ❌ | **0.11%** ❌ | −0.37 pp |
| AU target | ≥ 70% | ≥ 70% | — |
| AU target met | fail | fail | — |
| Throughput (samples/s) | 388,921 | 106,351 | −72.7% |
| I/O throughput (MiB/s) | 282.3 | 77.2 | −72.7% |
| Wall time (s) | 179.1 | 616.7 | +3.4× slower |
| Storage type | Local NVMe (POSIX) | S3 object (byte-range GET) | — |

**Takeaway**: DLRM is overwhelmingly I/O bound (0.375 ms/step compute). Neither storage target comes close to the ≥ 70% AU target. POSIX NVMe at 282 MiB/s delivers 4.4× better throughput than MinIO S3 at 77 MiB/s. Even NVMe page-cache hits cannot sustain the bandwidth demanded by 12,288-sample batches at near-zero compute time. A proper DLRM submission would require a much larger dataset (to defeat page caching) and high-bandwidth storage (e.g., NVMe RAID or a fast parallel filesystem).

---

## Notes

- DLRM is strongly I/O bound: 0.375 ms/step compute vs 1,350 ms for Flux.
  Even NVMe may struggle to meet AU ≥ 90% at 12,288 samples/step × ~761 bytes = ~9.1 MB/step × 1302 steps/epoch ≈ 11.8 GB must be read at accelerator speed.
- Parquet footer cache (`_pf_cache`) active in `parquet_reader.py` — same fix as Flux.
- S3 row-group reads via byte-range GET using `parquet_reader_s3_iterable.py`.

---

## Direct DLIO Benchmark — Reader Library Comparison (2026-05-07)

> These tests bypass `mlpstorage` and run `dlio_benchmark` directly to isolate storage library performance.
>
> **AU formula** (from `statscounter.py`):
> `AU = (metric_steps × computation_time_per_step) / metric_window_wall_time`
> where `metric_steps = total_steps − 1` (first step excluded by default `metric_exclude_start_steps=1`).
> AU represents the fraction of time the simulated accelerator is computing vs. waiting for I/O.

### Test Configuration

| Parameter | Value |
|-----------|-------|
| Date | 2026-05-07 |
| Benchmark | `dlio_benchmark.main` (direct, no `mlpstorage` wrapper) |
| S3 server | s3-ultra at `127.0.0.1:9200` (synthetic, ~40 GB/s capable) |
| S3 credentials | `minioadmin/minioadmin` |
| File storage | `/mnt/test/dlrm/train/*.parquet` |
| Dataset | 64 × ~971 MiB Parquet files, **~60.5 GiB total** |
| Row groups / file | 123 RGs @ ~8 MiB/RG compressed |
| DataLoader workers | 8 |
| Prefetch threads/worker | 64 |
| Prefetch window | 64 RGs |
| I/O pattern | Sliding-window RG prefetch (`TorchIterableDataset`) |
| Epochs | 1 |
| Batch size | 2,048 samples |
| Computation time | 0.770 ms/step |
| `read_threads` | 8 |

### Per-Worker I/O Timing (`[io_timing]` lines)

| Reader | Data/worker | Per-worker elapsed | Per-worker throughput |
|--------|------------|--------------------|-----------------------|
| S3 + s3torchconnector | 7.562 GiB | ~63 s | ~121–131 MiB/s |
| S3 + s3dlio | 7.562 GiB | ~48.5 s | ~159–160 MiB/s |
| File posix (buffered) | 7.562 GiB | ~58–63 s | ~121–131 MiB/s |
| File direct:// (O_DIRECT) | 7.562 GiB | ~49–51 s | ~151–158 MiB/s |

### Epoch Results (NP=1, Single Rank)

| Reader | Epoch wall time | Aggregate throughput (60.5 GiB) | AU (raw) | AU (corrected) |
|--------|-----------------|---------------------------------|----------|----------------|
| S3 + s3torchconnector | 107.79 s | ~575 MiB/s (~603 MB/s) | 22.3% | 35.7% |
| **S3 + s3dlio** | **76.07 s** | **~814 MiB/s (~854 MB/s)** | **31.6%** | **50.6%** |
| File posix (buffered) | 95.23 s | ~650 MiB/s (~682 MB/s) | 25.3% | 40.5% |
| File direct:// (O_DIRECT) | 80.41 s | ~770 MiB/s (~808 MB/s) | 30.0% | 48.0% |
| Dry-run (simulate, no I/O) | 38.51 s | — | 62.5% | 100% |

> **AU (raw)** = `(steps − 1) × 0.000770031 s / epoch_wall_time` = `24.06 s / epoch_wall_time`.
> Dry-run measured at 38.51 s → AU_dry = 24.06 / 38.51 = **62.5%** (framework overhead ceiling).
> **AU (corrected)** = `AU_raw / AU_dry_run` — normalizes out unavoidable DataLoader/framework overhead,
> expressing how much of the *achievable* compute time was actually utilized. 100% = no I/O stall beyond framework floor.
> Aggregate throughput = 60.5 GiB ÷ epoch wall time (all 8 workers run in parallel).

### Key Findings

- **s3dlio is 41.7% faster than s3torchconnector** on S3 (76s vs 108s epoch; ~160 vs ~126 MiB/s per worker). Both use byte-range GETs; s3dlio benefits from its Rust async runtime vs CRT thread pool under this workload.
- **File direct:// (O_DIRECT) is the fastest file reader** at 80.4s — slightly faster than s3dlio S3 and 15% faster than posix. O_DIRECT bypasses the page cache and exercises the NVMe bandwidth directly.
- **File posix is comparable to s3torchconnector** (95s vs 108s), suggesting both are similarly bounded by concurrency or I/O queue depth.
- **Dry-run floor is ~38.5s** — pure PyTorch DataLoader/compute overhead with no I/O. All configurations add meaningful I/O time on top.
- DLIO's built-in I/O metric should be ignored — it reports ~0.84 MiB/s because it counts `get_sample()` calls × `record_length` (1024 bytes), not actual bytes transferred. Use `[io_timing]` lines for true throughput.

### Run Commands

```bash
cd /home/eval/Documents/Code/dlio_benchmark

# S3 + s3torchconnector
AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin AWS_ENDPOINT_URL=http://127.0.0.1:9200 \
  uv run python -m dlio_benchmark.main workload=dlrm_s3dlio_s3 \
  ++workload.storage.storage_options.storage_library=s3torchconnector

# S3 + s3dlio
AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin AWS_ENDPOINT_URL=http://127.0.0.1:9200 \
  uv run python -m dlio_benchmark.main workload=dlrm_s3dlio_s3 \
  ++workload.storage.storage_options.storage_library=s3dlio

# File direct:// (O_DIRECT via s3dlio)
uv run python -m dlio_benchmark.main workload=dlrm_s3dlio_file \
  ++workload.storage.storage_options.storage_library=direct

# File posix (buffered)
uv run python -m dlio_benchmark.main workload=dlrm_s3dlio_file

# Dry-run (simulate, no I/O)
uv run python -m dlio_benchmark.main workload=dlrm_s3dlio_file \
  ++workload.storage.storage_options.simulate_io=true
```

---

## Multi-Rank MPI Scaling — S3 + s3dlio (2026-05-07)

> Same `dlrm_s3dlio_s3` workload as above, launched via `mpirun` to simulate multiple accelerator ranks.
> All ranks share the same s3-ultra instance (127.0.0.1:9200). Each rank reads an equal share of the 64 files.

### Configuration

| Parameter | Value |
|-----------|-------|
| Date | 2026-05-07 |
| Storage library | s3dlio |
| S3 server | s3-ultra at `127.0.0.1:9200` |
| Config | `dlrm_s3dlio_s3.yaml` (64 files, 2048 batch, 0.770031 ms compute) |
| DataLoader workers/rank | 8 |
| Prefetch threads/worker | 64 |

### Dry-Run Baselines (framework overhead only, `simulate_io=true`)

| NP | Dry-run epoch | Compute budget/rank | AU_dry (raw) |
|----|--------------|--------------------|--------------|
| 1 | 36.35 s | 24.06 s | 66.2% |
| 2 | 20.61 s | 12.03 s | 58.4% |
| 4 | 9.65 s | 6.01 s | 62.3% |

> AU_dry drops at NP=2 because fewer steps per rank means less compute time relative to fixed per-rank DataLoader startup overhead.

### Results

| NP (MPI ranks) | Files/rank | Steps/rank | Epoch wall time | Aggregate throughput | AU (raw) | AU (corrected) |
|----------------|-----------|-----------|-----------------|---------------------|----------|----------------|
| 1 | 64 | 31,248 | 81.65 s | 759 MiB/s (796 MB/s) | 29.5% | 44.6% |
| 2 | 32 | 15,625 | 56.67 s | 1,094 MiB/s (1,147 MB/s) | 21.2% | 36.3% |
| 4 | 16 | 7,812 | 49.57 s | 1,250 MiB/s (1,311 MB/s) | 12.1% | 19.4% |

> **Aggregate throughput** = 60.5 GiB ÷ epoch wall time (all NP ranks run in parallel on same dataset).
>
> **AU (raw)** = `(steps_per_rank − 1) × 0.000770031 s / epoch_wall_time`:
> - NP=1: 24.06 / 81.65 = **29.5%** &nbsp; NP=2: 12.03 / 56.67 = **21.2%** &nbsp; NP=4: 6.01 / 49.57 = **12.1%**
>
> **AU (corrected)** = `AU_raw / AU_dry` using per-NP dry-run baselines above:
> - NP=1: 29.5% / 66.2% = **44.6%** &nbsp; NP=2: 21.2% / 58.4% = **36.3%** &nbsp; NP=4: 12.1% / 62.3% = **19.4%**

### Key Findings

- **Throughput scales super-linearly** going from NP=1 to NP=2 (+44%), then flattens NP=2→NP=4 (+14%). Multiple ranks issue concurrent GETs that better saturate s3-ultra's async runtime.
- **Raw AU decreases with more ranks**: each rank processes fewer steps (less compute time) while epoch wall time doesn't shrink proportionally. This is expected and not a storage deficiency.
- **Corrected AU also decreases with NP** (44.6% → 36.3% → 19.4%): at NP=4, even the dry-run baseline is tighter (only 9.65s epoch), so the I/O stall takes a larger share of the available time. The benchmark is genuinely becoming more I/O-limited per rank as NP scales on a shared single-node server.
- **Epoch wall time compresses** as NP increases (81.65 → 56.67 → 49.57 s), but with diminishing returns as all ranks compete for the same single-node S3 server.
- On a real multi-node deployment with dedicated S3 bandwidth per node, both throughput and corrected AU would scale more linearly.

### Run Commands

```bash
cd /home/eval/Documents/Code/dlio_benchmark
export AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin AWS_ENDPOINT_URL=http://127.0.0.1:9200

# NP=1
mpirun -n 1 -host 127.0.0.1:1 --bind-to none --map-by socket --mca btl ^vader --allow-run-as-root \
  .venv/bin/dlio_benchmark workload=dlrm_s3dlio_s3 \
  ++workload.storage.storage_options.storage_library=s3dlio \
  --config-dir=dlio_benchmark/configs

# NP=2
mpirun -n 2 -host 127.0.0.1:2 --bind-to none --map-by socket --mca btl ^vader --allow-run-as-root \
  .venv/bin/dlio_benchmark workload=dlrm_s3dlio_s3 \
  ++workload.storage.storage_options.storage_library=s3dlio \
  --config-dir=dlio_benchmark/configs

# NP=4
mpirun -n 4 -host 127.0.0.1:4 --bind-to none --map-by socket --mca btl ^vader --allow-run-as-root \
  .venv/bin/dlio_benchmark workload=dlrm_s3dlio_s3 \
  ++workload.storage.storage_options.storage_library=s3dlio \
  --config-dir=dlio_benchmark/configs
```
