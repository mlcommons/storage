# MLPerf Storage — Checkpointing Benchmark Results

**Workload**: LLaMA3-8B ZeRO-3 checkpoint (fp16 model + fp32 optimizer states)  
**Config**: `configs/dlio/workload/llama3_8b_checkpoint.yaml`  
**MPI ranks**: 4 (each rank = 1 ZeRO-3 shard)  
**Checkpoints**: 2 write + 2 read per run  

---

## Checkpoint Layout

| File type | Per-rank size | 4-rank total |
|-----------|--------------|-------------|
| `model_states.pt` (fp16) | 3.74 GB (4,015,130,624 B) | 14.96 GB |
| `optim_states.pt` (fp32) | 22.44 GB (24,091,029,504 B) | 89.74 GB |
| **Total per checkpoint** | **26.18 GB** | **104.70 GB** |

---

## S3 Object Storage — s3-ultra (localhost:9500)

**Storage target**: `s3://checkpoint-test/s3dlio/llama3-8b/` via s3-ultra fake-S3 server (in-memory, no disk)  
**Library**: s3dlio (multipart upload, 32 MB parts, 16 in-flight)  
**Run dir**: `/tmp/dlio-checkpoint-20260426_172957`  
**Date**: 2026-04-26  

### Write Results

| Checkpoint | Total duration | **Throughput** |
|------------|----------------|----------------|
| Checkpoint 1 | ~47.8 s | **~2.192 GiB/s** |
| Checkpoint 2 | ~46.9 s | **~2.232 GiB/s** |
| **Mean** | **47.32 s** | **2.213 GiB/s** |

### Read Results

| Checkpoint | model_states read | optim_states read | Total duration | **Throughput** |
|------------|------------------|------------------|----------------|----------------|
| Checkpoint 1 | ~1.96 s | ~10.59 s | 12.55 s | **8.344 GiB/s** |
| Checkpoint 2 | ~1.80 s | ~10.57 s | 12.38 s | **8.459 GiB/s** |
| **Mean** | | | **12.46 s** | **8.401 GiB/s** |

### DLIO Metrics (from dlio.log)
```
[METRIC] Checkpoint save duration (seconds):          47.3214 (±0.6175)
[METRIC] Checkpoint save I/O Throughput (GiB/second):  2.2130 (±0.0289)
[METRIC] Checkpoint load duration (seconds):          12.4634 (±0.0856)
[METRIC] Checkpoint load I/O Throughput (GiB/second):  8.4013 (±0.0577)
```

### Object Inventory (verified via s3-cli stat)
16 objects total across 2 checkpoints × 4 ranks × 2 file types.  
All objects confirmed present and correct size after run.

---

## POSIX / Local Filesystem — /mnt/nvme_data

**Storage target**: `/mnt/nvme_data/mlperf_checkpoint_data`  
**Num layers**: 24 (scaled down from 32 to fit NVMe)  
**Checkpoints**: 1 write + 1 read  
**Run dir**: `/tmp/dlio-checkpoint-posix-20260426_172205`  
**Date**: 2026-04-26  

### Checkpoint Layout (24 layers)

| File type | Per-rank size | 4-rank total |
|-----------|--------------|-------------|
| `model_states.pt` (fp16) | 2.93 GB (3,149,144,064 B) | 11.72 GB |
| `optim_states.pt` (fp32) | 17.56 GB (18,856,341,504 B) | 70.25 GB |
| **Total per checkpoint** | **20.49 GB** | **81.95 GiB** |

### Write Results

| Checkpoint | Total duration | **Throughput** |
|------------|----------------|----------------|
| Checkpoint 1 | 57.87 s | **1.4161 GiB/s** |

### Read Results

| Checkpoint | model_states read | optim_states read | Total duration | **Throughput** |
|------------|------------------|------------------|----------------|----------------|
| Checkpoint 1 | 6.51 s | 22.48 s | 28.99 s | **2.8268 GiB/s** |

### DLIO Metrics (from dlio.log)
```
[METRIC] Checkpoint save duration (seconds): 57.8734 (0.0000)
[METRIC] Checkpoint save I/O Throughput (GiB/second): 1.4161 (0.0000)
[METRIC] Checkpoint load duration (seconds): 28.9913 (0.0000)
[METRIC] Checkpoint load I/O Throughput (GiB/second): 2.8268 (0.0000)
```

---

## Comparison Summary

| Metric | S3 (s3-ultra) | POSIX (NVMe) |
|--------|--------------|--------------|
| Storage backend | s3-ultra localhost:9500 (in-memory) | /mnt/nvme_data NVMe |
| Num layers | 32 | 24 |
| Checkpoint size | 104.70 GiB | 81.95 GiB |
| Write throughput | **2.213 GiB/s** | **1.416 GiB/s** |
| Read throughput | **8.401 GiB/s** | **2.827 GiB/s** |
| Write duration | 47.3 s (mean of 2) | 57.9 s |
| Read duration | 12.5 s (mean of 2) | 29.0 s |

**Notes:**
- S3 write throughput (2.21 GiB/s) now **exceeds** POSIX NVMe write (1.42 GiB/s). s3-ultra runs locally and consumes ~50% of the system; a dedicated remote S3 server would yield higher throughput.
- S3 read throughput (8.40 GiB/s) is much faster than POSIX because s3-ultra serves data from RAM (no disk I/O on reads).
- Write performance improved 2.25× (0.985 → 2.213 GiB/s) by aligning multipart upload part size (32 MB) with dgen-py's buffer granularity, eliminating the 4-buffer assembly stall that occurred with the previous 128 MB part size.
- Both tests used dgen-py for zero-copy random data generation (not verifying read-back correctness — data is random).
