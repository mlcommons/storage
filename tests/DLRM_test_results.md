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
