# Flux Training Benchmark Results

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
| Model | flux |
| Simulated accelerators | 4 × B200 |
| MPI ranks | 4 (local, `127.0.0.1:4`) |
| Epochs | 1 |
| Batch size | 48 samples/step |
| Steps per epoch | 173 (256 × 130 / 48 / 4) |
| Files (train) | 130 Parquet |
| Samples per file | 256 |
| Total samples | 33,280 |
| Dataset size | ~67.1 GB |
| Simulated compute time | 1.35 s/step |
| `read_threads` | 2 per rank |

## Storage Targets

### 1 — MinIO S3 (Object Storage)

| Field | Value |
|-------|-------|
| Run ID | 20260426_155644 |
| Date | 2026-04-26 15:56 – 16:01 UTC |
| Storage type | S3 object storage |
| Endpoint | `https://172.16.1.40:9000` (MinIO) |
| Bucket | `mlp-flux` |
| Storage library | s3dlio 0.9.x (byte-range GET) |
| Data path | `s3://mlp-flux/data/flux/train/` |

#### Results

| Metric | Value |
|--------|-------|
| **Accelerator Utilization (AU)** | **85.39%** |
| AU target | ≥ 90% |
| AU target met | ❌ fail |
| Training throughput | 120.72 samples/s |
| I/O throughput | **249.2 MiB/s** |
| Epoch 1 wall time | 287.8 s |

#### Notes

- First successful run after fixing per-sample footer re-read bug in `parquet_reader_s3_iterable.py`.
- Root cause of prior hangs: `ON_DEMAND` mode calls `open()/close()` around every sample. Before fix, `open()` re-fetched the Parquet footer from S3 each call (33,280 extra GETs/epoch). Fix: `_pf_cache` caches `(ParquetFile, row-offsets)` for the full epoch; flushed at `finalize()`.
- I/O throughput of 249 MiB/s is well below the storage system's capable ~800 MiB/s. Likely bottleneck: byte-range GET latency per row-group × 2 `read_threads` per rank.

---

### 2 — POSIX (Local NVMe)

| Field | Value |
|-------|-------|
| Run ID | 20260426_160857 |
| Date | 2026-04-26 16:09 – 16:13 MDT |
| Storage type | POSIX (local filesystem) |
| Device | `/dev/nvme4n2p1` (NVMe SSD, 98 GB) |
| Mount point | `/mnt/nvme_data` |
| Data path | `/mnt/nvme_data/mlperf_storage_dlio_data/flux/` |

#### Results

| Metric | Value |
|--------|-------|
| **Accelerator Utilization (AU)** | **99.66%** |
| AU target | ≥ 90% |
| AU target met | ✅ success |
| Training throughput | 140.89 samples/s |
| I/O throughput | **290.9 MiB/s** |
| Epoch 1 wall time | 247.2 s |

#### Notes

- POSIX run significantly outperforms S3: AU 99.66% vs 85.39%, wall time 247s vs 288s.
- I/O throughput (290.9 MiB/s) only marginally higher than S3 (249.2 MiB/s); the data was largely served from the Linux page cache (~36 GB Inactive(file) cached after reads) rather than raw NVMe.
- The AU improvement from 85% → 99.7% shows the S3 bottleneck is network/latency, not CPU or computation.
- `parquet_reader.py` `_pf_cache` fix equally effective: footer reads cached per-epoch, row-group byte counts in `_rg_cache`.

---

## POSIX Run Commands

Data directory: `/mnt/nvme_data/mlperf_storage_dlio_data`

```bash
# Datagen
uv run mlpstorage training datagen --model flux --num-processes 4 \
  --allow-run-as-root --open --skip-validation \
  --data-dir /mnt/nvme_data/mlperf_storage_dlio_data \
  --params dataset.num_files_train=130 dataset.num_samples_per_file=256

# Training
uv run mlpstorage training run --model flux --num-accelerators 4 \
  --accelerator-type b200 --client-host-memory-in-gb 47 \
  --open --allow-run-as-root --skip-validation --file \
  --data-dir /mnt/nvme_data/mlperf_storage_dlio_data \
  --params dataset.num_files_train=130 dataset.num_samples_per_file=256
```

---

## Comparison Summary

| Metric | MinIO S3 | POSIX NVMe | Delta |
|--------|----------|------------|-------|
| Run ID | 20260426_155644 | 20260426_160857 | — |
| **AU %** | **85.39%** ❌ | **99.66%** ✅ | +14.3 pp |
| AU target met | fail | success | — |
| Throughput (samples/s) | 120.72 | 140.89 | +16.7% |
| I/O throughput (MiB/s) | 249.2 | 290.9 | +16.7% |
| Wall time (s) | 287.8 | 247.2 | −14.1% |
| Storage type | S3 object (byte-range GET) | Local NVMe (POSIX mmap) | — |

**Takeaway**: POSIX NVMe comfortably meets the ≥ 90% AU target (99.66%). The MinIO S3 target falls short at 85.4%, indicating the storage system or network is the bottleneck rather than compute. The `_pf_cache` fix (epoch-scoped Parquet footer cache) was required to achieve these results on both storage paths — without it, per-sample footer re-reads would have caused hangs or severe performance degradation.
