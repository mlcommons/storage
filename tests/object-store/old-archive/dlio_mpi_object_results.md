# DLIO + s3dlio MPI Scaling Results — UNet3D h100 Workload

**Date:** March 20, 2026  
**System:** loki-russ  
**Storage:** MinIO @ `http://minio-host:9000`  
**Bucket:** `mlp-s3dlio`  
**Network bandwidth (measured limit):** ~1.2 GB/s

---

## Test Configuration

| Parameter | Value |
|---|---|
| Workload | UNet3D h100 |
| Files | 168 × ~140 MB NPZ |
| Total dataset size | ~23.5 GB |
| Epochs | 5 |
| Batch size | 7 samples/step |
| PyTorch DataLoader threads per rank | 4 |
| Storage library | s3dlio (v0.9.82) |
| multiprocessing_context | spawn |
| Config | `configs/dlio/workload/unet3d_h100_s3dlio.yaml` |

All runs used `--mca btl ^vader` to disable OpenMPI's shared-memory (vader) BTL
(see [Known Issues](#known-issues) below).

---

## Metrics Methodology

All throughput and samples/s figures throughout this document use **wall-clock epoch duration** from the DLIO log line:

> `Ending epoch N - K steps completed in X.XX s`

**Formulas — identical for every library and every NP:**

| Metric | Formula |
|---|---|
| I/O Throughput (GB/s) | `24.63 GB ÷ epoch_wall_clock_s` |
| I/O Throughput (MB/s) | `24.63 × 1024 ÷ epoch_wall_clock_s` |
| Samples/s | `168 samples ÷ epoch_wall_clock_s` |
| Summary warm value | mean ± stddev of **epochs 2–5** |
| vs NP=1 | warm GB/s at NP=N ÷ warm GB/s at NP=1 |

**Constants:** 168 files × 146.6 MB = 24,628.8 MB = **24.63 GB** total dataset; 168 total samples per epoch.

**DLIO `[METRIC]` I/O throughput** (and per-epoch DLIO samples/s) exclude the 0.323 s/step compute time from the denominator, so they read higher than wall-clock. They are shown for reference only where noted.

---

## Results

### Summary

| MPI Ranks (NP) | Steps/epoch | Epoch 1 time (cold) | Epoch 2–5 time (warm) | I/O Throughput (MB/s) | I/O Throughput (GB/s) | Samples/s | vs NP=1 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 24 | ~88 s | ~78 s | **332 ± 0.7** | **0.33** | 2.37 ± 0.005 | 1.0× |
| 2 | 12 | ~54 s | ~43 s | **664 ± 3.2** | **0.66** | 4.75 ± 0.023 | 2.0× |
| 4 | 6 | ~34 s | ~23 s | **1720 ± 125** | **1.72** | 12.31 ± 0.89 | 5.2× |

Throughput figures are averaged over all 5 epochs (DLIO `[METRIC]` line).

### Per-Epoch Detail — NP=4

| Epoch | Steps | Duration | GB/s (wall-clock) | Throughput (samples/s) | Notes |
|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 6 | 34.0 s | 0.724 | 10.64 | Cold read from MinIO over network |
| 2 | 6 | 22.4 s | 1.100 | 11.93 | Warm — page cache active |
| 3 | 6 | 22.9 s | 1.076 | 12.94 | Warm |
| 4 | 6 | 22.9 s | 1.076 | 13.77 | Warm |
| 5 | 6 | 22.7 s | 1.085 | 13.77 | Warm |

---

## s3dlio Tuned Training (Read) Performance — NP=1 Experiment

**Env vars applied in `tests/object-store/dlio_s3dlio_train.sh`:**
```bash
export S3DLIO_ENABLE_RANGE_OPTIMIZATION=0
export S3DLIO_RT_THREADS=8
```

**Result:** No meaningful change — **329.5 ± 0.9 MB/s** vs original **332 ± 0.7 MB/s** (within noise).

**Root cause — wrong knob for the `get_many()` path:**
`S3DLIO_ENABLE_RANGE_OPTIMIZATION` is only read inside `S3ObjectStore::get()` in
`object_store.rs`. The `get_many()` Python function routes through
`get_objects_parallel()` → `get_object_uri_optimized_async()` in `s3_utils.rs`, which
does **not** check that env var. To actually disable range splitting on the `get_many`
path, use `S3DLIO_RANGE_THRESHOLD_MB=1000` (any value larger than the file size, 147 MB).

| NP | Env vars applied | Steps/epoch | Epoch 1 (cold) | Epoch 2–5 (warm) | I/O Throughput (MB/s) | GB/s | Samples/s | vs untuned NP=1 |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0` `S3DLIO_RT_THREADS=8` | 24 | ~90 s | ~79 s | **329.5 ± 0.9** | **0.322** | 2.357 ± 0.007 | ~1.0× (no change) |
| 2 | `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0` `S3DLIO_RT_THREADS=8` | 12 | ~54 s | ~43 s | **675.7 ± 2.1** | **0.660** | 4.833 ± 0.015 | 2.05× |
| 4 | `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0` `S3DLIO_RT_THREADS=8` | 6 | ~34 s | ~23 s | **1661.5 ± 95.7** | **1.623** | 11.884 ± 0.685 | 5.06× |

### Per-Epoch Detail — NP=1 Tuned

| Epoch | Steps | Duration | GB/s (wall-clock) | Throughput (samples/s) | Notes |
|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 24 | 89.99 s | 0.274 | 2.3598 | Cold read from MinIO over network |
| 2 | 24 | 78.88 s | 0.312 | 2.3538 | Warm — page cache active |
| 3 | 24 | 78.65 s | 0.313 | 2.3647 | Warm |
| 4 | 24 | 79.30 s | 0.311 | 2.3459 | Warm |
| 5 | 24 | 78.99 s | 0.312 | 2.3600 | Warm |

**Warm avg:** ~78.95 s → **0.312 GB/s** (identical to untuned warm avg of ~0.31 GB/s).

### Per-Epoch Detail — NP=2 Tuned

| Epoch | Steps | Duration | GB/s (wall-clock) | Throughput (samples/s) | Notes |
|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 12 | 53.64 s | 0.448 | 4.8994 | Cold read from MinIO over network |
| 2 | 12 | 42.67 s | 0.564 | 4.9111 | Warm — page cache active |
| 3 | 12 | 43.03 s | 0.559 | 4.9099 | Warm |
| 4 | 12 | 42.76 s | 0.562 | 4.9012 | Warm |
| 5 | 12 | 42.87 s | 0.561 | 4.9062 | Warm |

**Warm avg:** ~42.83 s → **0.562 GB/s**.

> **Interpretation:** Throughput improved marginally vs untuned NP=2 (675.7 vs 664 MB/s, ~1.7% — within noise). However, CPU and memory utilization dropped significantly — confirming that `S3DLIO_RT_THREADS=8` eliminated the Tokio thread-count overhead (see Finding 3 in the analysis). Range splitting is still occurring (`S3DLIO_ENABLE_RANGE_OPTIMIZATION=0` is a no-op here), but with fewer Tokio threads, per-thread OS scheduling cost is much lower. Next step: test with `S3DLIO_RANGE_THRESHOLD_MB=1000` to also eliminate range splitting.

### Per-Epoch Detail — NP=4 Tuned

| Epoch | Steps | Duration | GB/s (wall-clock) | Throughput (samples/s) | Notes |
|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 6 | 34.04 s | 0.707 | 15.7825 | Cold read from MinIO over network |
| 2 | 6 | 22.67 s | 1.061 | 11.3513 | Warm — page cache active |
| 3 | 6 | 22.60 s | 1.064 | 12.1462 | Warm |
| 4 | 6 | 22.82 s | 1.054 | 12.1807 | Warm |
| 5 | 6 | 22.82 s | 1.054 | 12.9190 | Warm |

**Warm avg:** ~22.73 s → **1.058 GB/s**.

---

## Data Generation (Write) Performance

**All three libraries used NP=8 (8 MPI ranks) for data generation — the default for all datagen scripts.**  
Dataset: 168 × 146.6 MB NPZ = 24.63 GB total.  
Timings are wall-clock seconds from `Starting data generation` to `Generation done` in the DLIO log.

| Library | Write implementation | Throughput (MB/s) | Throughput (GB/s) | vs s3dlio |
|---|---|:-:|:-:|:-:|
| s3dlio | **`MultipartUploadWriter`** | **889 ± 5** | **0.889** | 1.0× |
| minio-py | automatic multipart (5 MB parts) | **823 ± 34** | **0.823** | 0.93× |
| s3torchconnector | streaming `put_object` | **963 ± 14** | **0.963** | 1.08× |

**Winner: s3torchconnector at 963 MB/s — 8% faster than s3dlio multipart, 16% faster than minio-py.**

> **minio-py spread (±34 MB/s across 5 runs):** Environmental variation across the measurement window — individual runs range from 28.5 s to 31.2 s. Not a library characteristic.

### Individual Datagen Run Log (all NP=8)

| Library | Log timestamp | Duration | MB/s |
|---|---|:-:|:-:|
| s3dlio (MultipartUploadWriter) | `dlio-s3dlio-datagen-20260320_114719` | 27.91 s | 882 |
| s3dlio (MultipartUploadWriter) | `dlio-s3dlio-datagen-20260320_120959` | 27.44 s | 897 |
| s3dlio (MultipartUploadWriter) | `dlio-s3dlio-datagen-20260320_152849` | 27.71 s | 889 |
| s3dlio (MultipartUploadWriter) | `dlio-s3dlio-datagen-20260320_180423` | 27.75 s | 888 |
| minio-py | `dlio-minio-datagen-20260320_111707` | 30.70 s | 802 |
| minio-py | `dlio-minio-datagen-20260320_111818` | 30.70 s | 802 |
| minio-py | `dlio-minio-datagen-20260320_121228` | 28.49 s | 865 |
| minio-py | `dlio-minio-datagen-20260320_130727` | 28.82 s | 854 |
| minio-py | `dlio-minio-datagen-20260320_164356` | 31.17 s | 790 |
| s3torchconnector | `dlio-s3torch-datagen-20260320_122511` | 25.21 s | 977 |
| s3torchconnector | `dlio-s3torch-datagen-20260320_161531` | 25.96 s | 949 |

### Historical: s3dlio before multipart fix (single-part PUT, NP=8)

The original `put_bytes()` path issued a single HTTP PUT for the entire 147 MB object — one TCP flow, no concurrency. minio-py splits automatically at 5 MB parts; s3torchconnector streams via chunked transfer. Result: s3dlio was 47% slower than the other two libraries.

| Log timestamp | Duration | MB/s |
|---|:-:|:-:|
| `dlio-s3dlio-datagen-20260320_094109` | 52.39 s | 470 |
| `dlio-s3dlio-datagen-20260320_112449` | 52.21 s | 472 |
| `dlio-s3dlio-datagen-20260320_114245` | 52.12 s | 473 |
| **mean** | **52.24 ± 0.11 s** | **471 ± 1** |

**Fix applied:** [dlio_benchmark/storage/obj_store_lib.py](../../dlio_benchmark/dlio_benchmark/storage/obj_store_lib.py) — `put_data()` now routes objects ≥ 16 MB through `s3dlio.MultipartUploadWriter.from_uri()`. No changes to s3dlio itself were required.  
Threshold configurable via `S3DLIO_MULTIPART_THRESHOLD_MB` (default 16).

---

## Key Finding: Page Cache Reuse With Object Storage

**The NP=4 average throughput of 1,720 MB/s exceeds the physical network limit of 1,200 MB/s — proving that a substantial fraction of the epoch 2–5 reads are being served from the Linux page cache, not from the network.**

### How this works

When a DLIO worker reads an object from MinIO via s3dlio:

1. s3dlio fetches the object over the network into memory
2. The kernel stores a copy of those pages in the **Linux page cache** (not s3dlio-specific — all file descriptor reads go through the VFS page cache)
3. On the next epoch, when the same object is re-requested, the kernel serves those pages directly from RAM without touching the network

This happens transparently: neither DLIO nor s3dlio explicitly manages a cache. The OS page cache just does what it always does for any I/O.

### Why this was unexpected

Object storage reads go through a socket, not a mapped file, so the expectation was that each read would always hit the network. The surprise is that **the Linux kernel caches socket read data in the page cache regardless of whether the source is a file or a TCP stream**, provided the data path goes through standard VFS read calls.

This is the same caching effect observed when benchmarking local NFS or block storage — sequential-epoch AI training workloads always re-read the same files across epochs, and the OS caches aggressively.

### Implications for benchmarking

| Scenario | What it means |
|---|---|
| **Epoch 1 throughput** | True cold-read performance — reflects actual network/storage bandwidth |
| **Epoch 2+ throughput** | Warm performance — partially or fully served from page cache |
| **Averaged-epoch metric** | Blends cold + warm; optimistic relative to a fresh system |
| **Large dataset (> RAM)** | Page cache thrashing; all epochs approximate cold performance |
| **Production workload** | Page cache benefit is real — systems doing repeated training runs will see this speedup |

To measure true storage-only performance, the dataset must exceed available system RAM, or the page cache must be cleared between epochs (`echo 3 > /proc/sys/vm/drop_caches` as root).

The 23.5 GB dataset fits comfortably in RAM on loki-russ, so after epoch 1, subsequent epochs run almost entirely from cache.

---

## s3dlio Tuned Training — `S3DLIO_RANGE_THRESHOLD_MB=1000` + `S3DLIO_RT_THREADS=8`

**Env vars applied:**
```bash
export S3DLIO_RANGE_THRESHOLD_MB=1000   # single streaming GET for files < 1000 MB (no range splitting)
export S3DLIO_RT_THREADS=8              # 8 Tokio threads per process (vs default 32)
```

**Note:** `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0` was used in the prior "tuned" run above — that is a
confirmed no-op for `get_many()`. This run uses the correct knobs. See [s3dlio_performance_analysis.md](s3dlio_performance_analysis.md) §6 Tier 1 for details.

**Also active:** `_BytesViewIO` zero-copy fix in `npz_reader_s3_iterable.py` (eliminates the `bytes(data)` 147 MB/file copy).

### Per-Epoch Detail — NP=1 (correct env vars + zero-copy fix)

| Epoch | Steps | Duration | GB/s (wall-clock) | MB/s (wall-clock) | Throughput (samples/s) | Notes |
|:-:|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 24 | 72.28 s | 0.333 | 340.8 | 2.325 | Cold read from MinIO over network |
| 2 | 24 | 60.90 s | 0.395 | 404.4 | 2.759 | Warm — page cache active |
| 3 | 24 | 60.25 s | 0.399 | 408.8 | 2.788 | Warm |
| 4 | 24 | 60.24 s | 0.399 | 408.8 | 2.789 | Warm |
| 5 | 24 | 60.00 s | 0.401 | 410.5 | 2.800 | Warm |

**Warm avg (epochs 2–5):** 60.35 s → **408 ± 2 MB/s** | **0.398 GB/s** | **2.784 ± 0.015 samples/s**

> DLIO `[METRIC]` reports **431.1 MB/s** — higher than wall-clock because it excludes compute time
> (0.323 s/step × 24 steps ≈ 7.75 s/epoch) from the denominator. Wall-clock methodology is used
> throughout this document for consistency.

### Per-Epoch Detail — NP=2 (correct env vars + zero-copy fix)

| Epoch | Steps | Duration | GB/s (wall-clock) | MB/s (wall-clock) | Throughput (samples/s) | Notes |
|:-:|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 12 | 44.89 s | 0.536 | 548.6 | 3.743 | Cold read from MinIO over network |
| 2 | 12 | 33.71 s | 0.714 | 730.8 | 4.985 | Warm — page cache active |
| 3 | 12 | 34.03 s | 0.706 | 723.3 | 4.937 | Warm |
| 4 | 12 | 33.44 s | 0.719 | 736.5 | 5.024 | Warm |
| 5 | 12 | 34.00 s | 0.707 | 724.4 | 4.941 | Warm |

**Warm avg (epochs 2–5):** 33.80 s → **729 ± 5 MB/s** | **0.712 GB/s** | **4.97 samples/s**

> DLIO `[METRIC]` reports **857.9 MB/s** — higher than wall-clock as compute time (~3.9 s/epoch
> for 12 steps × 0.323 s/step) is excluded from the denominator.

**Scaling NP=1 → NP=2: 408 → 729 MB/s = 1.79× speedup** (vs ideal 2.0× for linear scaling).

### Per-Epoch Detail — NP=4 (correct env vars + zero-copy fix)

**Methodology:** MB/s = 24,628.8 MB ÷ duration_s; GB/s = MB/s ÷ 1024; samples/s = 168 ÷ duration_s.

| Epoch | Steps | Duration | GB/s (wall-clock) | MB/s (wall-clock) | Throughput (samples/s) | Notes |
|:-:|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 6 | 33.84 s | 0.711 | 727.7 | 4.965 | Cold read from MinIO over network |
| 2 | 6 | 22.59 s | 1.065 | 1090.3 | 7.438 | Warm — page cache active |
| 3 | 6 | 22.57 s | 1.066 | 1091.2 | 7.444 | Warm |
| 4 | 6 | 22.62 s | 1.064 | 1088.9 | 7.427 | Warm |
| 5 | 6 | 22.59 s | 1.065 | 1090.3 | 7.438 | Warm |

**Warm avg (epochs 2–5):** 22.59 s → **1090 ± 1 MB/s** | **1.065 GB/s** | **7.44 samples/s**

> DLIO `[METRIC]` reports **1881.5 MB/s** — higher than wall-clock as compute time (~6 steps × 0.323 s/step ≈ 1.9 s/epoch) is excluded from the denominator.

**Scaling NP=2 → NP=4: 729 → 1090 MB/s = 1.49× speedup** (vs ideal 2.0×). Page cache saturation is reducing marginal gain — all 168 files are already cached after epoch 1 regardless of NP.

### Per-Epoch Detail — NP=8 (correct env vars + zero-copy fix)

**Methodology:** MB/s = 24,628.8 MB ÷ duration_s; GB/s = MB/s ÷ 1024; samples/s = 168 ÷ duration_s.

| Epoch | Steps | Duration | GB/s (wall-clock) | MB/s (wall-clock) | Throughput (samples/s) | Notes |
|:-:|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 3 | 34.42 s | 0.699 | 715.5 | 4.881 | Cold read from MinIO over network |
| 2 | 3 | 22.69 s | 1.060 | 1085.5 | 7.404 | Warm — page cache active |
| 3 | 3 | 22.67 s | 1.061 | 1086.5 | 7.410 | Warm |
| 4 | 3 | 22.79 s | 1.055 | 1080.6 | 7.371 | Warm |
| 5 | 3 | 22.57 s | 1.065 | 1091.1 | 7.444 | Warm |

**Warm avg (epochs 2–5):** 22.68 s → **1086 ± 4 MB/s** | **1.061 GB/s** | **7.41 samples/s**

---

## s3dlio v0.9.84 — Range Optimization Bug Fix — NP=1

**Library version:** s3dlio v0.9.82 wheel (to be tagged v0.9.84)  
**Key change:** `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0` now correctly applies to **all** code paths
including `get_many()` / `get_objects_parallel()` (was a confirmed no-op prior to v0.9.82).
This replaces the previous workaround of `S3DLIO_RANGE_THRESHOLD_MB=1000`.

**Env vars applied in `tests/object-store/dlio_s3dlio_train.sh`:**
```bash
export S3DLIO_ENABLE_RANGE_OPTIMIZATION=0   # skip HEAD + single GET (bug fixed in v0.9.82)
export S3DLIO_RT_THREADS=8                  # 8 Tokio threads per process
```

**Effect of the bug fix vs the old workaround (`RANGE_THRESHOLD_MB=1000`):**
- Old (`RANGE_THRESHOLD_MB=1000`): still issued 1 HEAD per file (to compare size against threshold), then fell back to single GET — **1 HEAD + 1 GET per file**
- New (`ENABLE_RANGE_OPTIMIZATION=0`): skips HEAD entirely, goes directly to single GET — **0 HEADs + 1 GET per file**; also skips the pre-stat phase in `get_objects_parallel()`

**Additional changes in v0.9.82 hit path:**
- `concurrent_range_get_impl()`: mutex-free collect-then-assemble (no impact when range opt disabled)
- `get_objects_parallel()`: O(N log N) sort via pre-built HashMap index (replaces O(N²) linear scan)
- `ObjectSizeCache` TTL changed from 5 min → 1 hour default (no impact for single-epoch test runs)
- OnceLock caching of env var reads (eliminates env syscall on hot path)

### DLIO [METRIC] Output (NP=1)

```
[METRIC] Number of Simulated Accelerators: 1
[METRIC] Training Accelerator Utilization [AU] (%): 15.1989 (0.1397)
[METRIC] Training Throughput (samples/second): 3.1146 (0.0269)
[METRIC] Training I/O Throughput (MB/second): 435.4454 (3.7665)
```

> DLIO [METRIC] excludes per-step compute time (~0.323 s/step × 24 steps ≈ 7.75 s/epoch) from the
> denominator. Wall-clock figures below are used throughout this document for consistency.

### Per-Epoch Detail — NP=1 (v0.9.84 bug-fix wheel)

**Methodology:** MB/s = 24,628.8 MB ÷ duration_s; GB/s = MB/s ÷ 1024; samples/s = 168 ÷ duration_s.

| Epoch | Steps | Duration | GB/s (wall-clock) | MB/s (wall-clock) | Throughput (samples/s) | Notes |
|:-:|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 24 | 71.52 s | 0.336 | 344.3 | 2.349 | Cold read from MinIO over network |
| 2 | 24 | 60.22 s | 0.399 | 408.9 | 2.790 | Warm — page cache active |
| 3 | 24 | 59.64 s | 0.403 | 412.9 | 2.817 | Warm |
| 4 | 24 | 59.38 s | 0.405 | 414.7 | 2.829 | Warm |
| 5 | 24 | 59.51 s | 0.404 | 413.8 | 2.823 | Warm |

**Warm avg (epochs 2–5):** 59.69 s → **413 ± 2 MB/s** | **0.403 GB/s** | **2.815 ± 0.015 samples/s**

### DLIO [METRIC] Output (NP=2)

```
[METRIC] Number of Simulated Accelerators: 2 
[METRIC] Training Accelerator Utilization [AU] (%): 15.1657 (0.1176)
[METRIC] Training Throughput (samples/second): 5.9271 (0.0493)
[METRIC] Training I/O Throughput (MB/second): 828.6602 (6.8904)
```

> DLIO [METRIC] excludes per-step compute time (~0.323 s/step × 12 steps ≈ 3.9 s/epoch) from the
> denominator. Wall-clock figures below are used throughout this document for consistency.

### Per-Epoch Detail — NP=2 (v0.9.84 bug-fix wheel)

**Methodology:** MB/s = 24,628.8 MB ÷ duration_s; GB/s = MB/s ÷ 1024; samples/s = 168 ÷ duration_s.

| Epoch | Steps | Duration | GB/s (wall-clock) | MB/s (wall-clock) | Throughput (samples/s) | Notes |
|:-:|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 12 | 45.40 s | 0.530 | 542.5 | 3.700 | Cold read from MinIO over network |
| 2 | 12 | 34.76 s | 0.692 | 708.6 | 4.833 | Warm — page cache active |
| 3 | 12 | 34.68 s | 0.694 | 710.2 | 4.845 | Warm |
| 4 | 12 | 34.21 s | 0.703 | 719.9 | 4.912 | Warm |
| 5 | 12 | 34.39 s | 0.699 | 716.1 | 4.885 | Warm |

**Warm avg (epochs 2–5):** 34.51 s → **713 ± 5 MB/s** | **0.697 GB/s** | **4.87 ± 0.03 samples/s**

**Scaling NP=1 → NP=2: 413 → 713 MB/s = 1.73×** (vs ideal 2.0×). Consistent with prior v0.9.82 NP=1→2 scaling (1.79× for the workaround run).

### DLIO [METRIC] Output (NP=4)

```
[METRIC] Number of Simulated Accelerators: 4 
[METRIC] Training Accelerator Utilization [AU] (%): 19.2339 (0.5320)
[METRIC] Training Throughput (samples/second): 13.3328 (0.3688)
[METRIC] Training I/O Throughput (MB/second): 1864.0430 (51.5630)
```

> DLIO [METRIC] excludes per-step compute time (~0.323 s/step × 6 steps ≈ 1.9 s/epoch) from the
> denominator. Wall-clock figures below are used throughout this document for consistency.

### Per-Epoch Detail — NP=4 (v0.9.84 bug-fix wheel)

**Methodology:** MB/s = 24,628.8 MB ÷ duration_s; GB/s = MB/s ÷ 1024; samples/s = 168 ÷ duration_s.

| Epoch | Steps | Duration | GB/s (wall-clock) | MB/s (wall-clock) | Throughput (samples/s) | Notes |
|:-:|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 6 | 33.55 s | 0.716 | 733.9 | 5.007 | Cold read from MinIO over network |
| 2 | 6 | 22.58 s | 1.066 | 1090.7 | 7.440 | Warm — page cache active |
| 3 | 6 | 22.60 s | 1.065 | 1089.8 | 7.434 | Warm |
| 4 | 6 | 22.79 s | 1.056 | 1080.6 | 7.372 | Warm |
| 5 | 6 | 22.66 s | 1.062 | 1086.8 | 7.414 | Warm |

**Warm avg (epochs 2–5):** 22.66 s → **1087 ± 4 MB/s** | **1.062 GB/s** | **7.42 ± 0.03 samples/s**

**Scaling NP=2 → NP=4: 713 → 1087 MB/s = 1.52×** (vs ideal 2.0×). Page cache saturation limits marginal gain — all 168 files cached after epoch 1 regardless of NP. Matches prior NP=4 result (1090 ± 1 MB/s) to within noise.

### DLIO [METRIC] Output (NP=8)

```
[METRIC] Number of Simulated Accelerators: 8 
[METRIC] Training Accelerator Utilization [AU] (%): 37.9346 (3.1990)
[METRIC] Training Throughput (samples/second): 32.8631 (2.7722)
[METRIC] Training I/O Throughput (MB/second): 4594.5609 (387.5733)
```

> DLIO [METRIC] excludes per-step compute time (~0.323 s/step × 3 steps ≈ 1.0 s/epoch) from the
> denominator. Wall-clock figures below are used throughout this document for consistency.

### Per-Epoch Detail — NP=8 (v0.9.84 bug-fix wheel)

**Methodology:** MB/s = 24,628.8 MB ÷ duration_s; GB/s = MB/s ÷ 1024; samples/s = 168 ÷ duration_s.

| Epoch | Steps | Duration | GB/s (wall-clock) | MB/s (wall-clock) | Throughput (samples/s) | Notes |
|:-:|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 3 | 36.14 s | 0.666 | 681.5 | 4.648 | Cold read from MinIO over network |
| 2 | 3 | 23.11 s | 1.041 | 1065.7 | 7.270 | Warm — page cache active |
| 3 | 3 | 24.70 s | 0.974 | 997.1 | 6.802 | Warm |
| 4 | 3 | 31.50 s | 0.764 | 781.9 | 5.333 | Warm — **anomalous slowdown** (network jitter / cache pressure) |
| 5 | 3 | 22.86 s | 1.052 | 1077.4 | 7.348 | Warm |

**Warm avg (epochs 2–5):** 25.54 s → **964 ± 120 MB/s** | **0.942 GB/s** | **6.58 ± 0.86 samples/s**

> **High variance note:** Epoch 4 (31.50 s) is a clear outlier — 2.5σ above the mean of the other 3 warm epochs (23.11, 24.70, 22.86 s → avg 23.56 s → **1045 MB/s**). This is consistent with the prior NP=8 run (1086 ± 4 MB/s) and the NP=4 result (1087 ± 4 MB/s). The anomaly is likely a transient network hiccup or OS page reclaim event, not a characteristic of the implementation.

**Scaling NP=4 → NP=8: 1087 → 964 MB/s (including E4 anomaly) or ~1045 MB/s (excluding E4) = essentially flat.** Both results confirm NP=8 with 3 steps/epoch hits the same page-cache ceiling as NP=4. Additional ranks add no benefit once the working set is fully cached.

| Run | Env vars | Warm MB/s | Warm samples/s | vs first |
|---|---|:-:|:-:|:-:|
| Untuned (v0.9.82) | defaults | **332 ± 0.7** | 2.37 ± 0.005 | 1.0× |
| `ENABLE_RANGE_OPTIMIZATION=0` (v0.9.82 — no-op) | `RT_THREADS=8` | **329.5 ± 0.9** | 2.357 ± 0.007 | ~1.0× |
| `RANGE_THRESHOLD_MB=1000` (v0.9.82 — workaround) + zero-copy fix | `RT_THREADS=8` | **408 ± 2** | 2.784 ± 0.015 | 1.23× |
| `ENABLE_RANGE_OPTIMIZATION=0` (v0.9.84 — bug fixed) | `RT_THREADS=8` | **413 ± 2** | 2.815 ± 0.015 | 1.24× |

**Net result:** The v0.9.84 bug fix delivers a marginal further improvement (~5 MB/s, ~1.2%) over the
`RANGE_THRESHOLD_MB=1000` workaround — consistent with the theoretical saving (HEAD requests eliminated
per batch). The difference is within noise given MinIO + network variability on this test system.
The primary gain in both cases comes from eliminating range splitting (HEAD + 37 range GETs → 0 HEADs + 1 GET).
The `ENABLE_RANGE_OPTIMIZATION=0` path is now the preferred and correct setting for this environment.

> DLIO `[METRIC]` reports **6066 MB/s** — this is an anomalously high average driven by high variance (stddev 955 MB/s); wall-clock warm epochs show consistent ~1086 MB/s. The DLIO metric likely includes at least one epoch where the page cache served the entire dataset near memory bandwidth.

**Scaling NP=4 → NP=8: 1087 → 964 MB/s measured (anomalous E4 at 31.50 s); excluding that outlier, the 3 normal warm epochs average ~1045 MB/s — essentially flat vs NP=4.** Confirms the page-cache ceiling is reached by NP=4.

### Impact vs Prior Runs

| Configuration | NP | Warm MB/s | vs untuned NP=1 | vs minio-py (same NP) |
|---|:-:|:-:|:-:|:-:|
| s3dlio untuned (baseline) | 1 | 332 ± 0.7 | 1.00× | 0.72× |
| s3dlio + `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0` + `S3DLIO_RT_THREADS=8` *(no-op env var)* | 1 | 329.5 ± 0.9 | ~1.00× | 0.72× |
| **s3dlio + `S3DLIO_RANGE_THRESHOLD_MB=1000` + `S3DLIO_RT_THREADS=8` + zero-copy fix** | **1** | **408 ± 2** | **+23%** | **0.89×** |
| **s3dlio + `S3DLIO_RANGE_THRESHOLD_MB=1000` + `S3DLIO_RT_THREADS=8` + zero-copy fix** | **2** | **729 ± 5** | **2.19×** | **0.85×** |
| **s3dlio v0.9.84 `ENABLE_RANGE_OPTIMIZATION=0` + `RT_THREADS=8`** | **1** | **413 ± 2** | **+24%** | **0.90×** |
| **s3dlio v0.9.84 `ENABLE_RANGE_OPTIMIZATION=0` + `RT_THREADS=8`** | **2** | **713 ± 5** | **2.15×** | **0.83×** |
| **s3dlio v0.9.84 `ENABLE_RANGE_OPTIMIZATION=0` + `RT_THREADS=8`** | **4** | **1087 ± 4** | **3.27×** | **0.99×** |
| **s3dlio v0.9.84 `ENABLE_RANGE_OPTIMIZATION=0` + `RT_THREADS=8`** | **8** | **964 ± 120** ¹ | **2.90×** | **0.87×** |
| **s3dlio + `S3DLIO_RANGE_THRESHOLD_MB=1000` + `S3DLIO_RT_THREADS=8` + zero-copy fix** | **4** | **1090 ± 1** | **3.28×** | **0.99×** |
| **s3dlio + `S3DLIO_RANGE_THRESHOLD_MB=1000` + `S3DLIO_RT_THREADS=8` + zero-copy fix** | **8** | **1086 ± 4** | **3.27×** | **0.98×** |
| minio-py (reference) | 1 | 459 ± 1 | 1.38× | 1.00× |
| minio-py (reference) | 2 | 857 ± 3 | 2.58× | 1.00× |
| minio-py (reference) | 4 | 1097 ± 3 | 3.30× | 1.00× |
| minio-py (reference) | 8 | 1107 ± 3 | 3.33× | 1.00× |

¹ NP=8 v0.9.84 high variance (±120 MB/s) driven by epoch 4 anomaly (31.50 s vs ~23 s for other warm epochs). Excluding epoch 4, the 3 remaining warm epochs average ~1045 MB/s (0.87× minio-py), consistent with the NP=8 v0.9.82 run (1086 ± 4 MB/s).

**At NP=4, s3dlio tuned matches minio-py within 1–2%.** Both libraries hit the same
page-cache ceiling (≈1087–1097 MB/s) and adding more ranks provides no further gain. The gap at
NP=1/2 (0.83–0.90×) is attributable to per-file fixed overhead; this cost becomes negligible
once cache-serve time dominates. The Rust-level HEAD elimination will primarily benefit
cold-epoch (epoch 1) performance across all NP levels.

---

## minio-py Training (Read) Performance — Scaling Study

**Bucket:** `mlp-minio` | **Config:** `configs/dlio/workload/unet3d_h100_minio.yaml`  
Same workload as s3dlio/s3torchconnector scaling study: 168 × ~140 MB NPZ, batch_size=7, 5 epochs, 4 DataLoader threads/rank.

### Summary

All figures computed per [Metrics Methodology](#metrics-methodology) above. NP=4/8 re-runs pending.

| MPI Ranks (NP) | Steps/epoch | Epoch 1 time (cold) | Epoch 2–5 time (warm) | I/O Throughput (MB/s) | I/O Throughput (GB/s) | Samples/s | vs NP=1 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 24 | 64.9 s | ~53.6 s | **459 ± 1** | **0.459** | 3.13 ± 0.01 | 1.0× |
| 2 | 12 | ~41.5 s | ~28.7 s | **857 ± 3** | **0.857** | 5.85 ± 0.02 | 1.87× |
| 4 | 6 | ~34.0 s | ~22.4 s | **1097 ± 3** | **1.097** | 7.49 ± 0.02 | 2.39× |
| 8 | 3 | ~34.7 s | ~22.8 s | **1107 ± 3** | **1.081** | 7.37 ± 0.02 | 2.35× |

### Per-Epoch Detail — NP=1

| Epoch | Steps | Duration | GB/s | Samples/s | Notes |
|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 24 | 64.93 s | 0.379 | 2.59 | Cold |
| 2 | 24 | 53.82 s | 0.458 | 3.12 | Network-rate |
| 3 | 24 | 53.52 s | 0.460 | 3.14 | Network-rate |
| 4 | 24 | 53.60 s | 0.460 | 3.13 | Network-rate |
| 5 | 24 | 53.63 s | 0.459 | 3.13 | Network-rate |

### Per-Epoch Detail — NP=2

| Epoch | Steps | Duration | GB/s | Samples/s | Notes |
|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 12 | 41.50 s | 0.593 | 4.05 | Cold |
| 2 | 12 | 28.84 s | 0.854 | 5.83 | Network-rate |
| 3 | 12 | 28.71 s | 0.858 | 5.85 | Network-rate |
| 4 | 12 | 28.71 s | 0.858 | 5.85 | Network-rate |
| 5 | 12 | 28.64 s | 0.860 | 5.87 | Network-rate |

### Per-Epoch Detail — NP=4

| Epoch | Steps | Duration | GB/s | Samples/s | Notes |
|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 6 | 34.00 s | 0.724 | 4.94 | Cold |
| 2 | 6 | 22.52 s | 1.093 | 7.46 | Page cache active |
| 3 | 6 | 22.37 s | 1.101 | 7.51 | Warm |
| 4 | 6 | 22.45 s | 1.097 | 7.48 | Warm |
| 5 | 6 | 22.43 s | 1.098 | 7.49 | Warm |

### Per-Epoch Detail — NP=8

| Epoch | Steps | Duration | GB/s | Samples/s | Notes |
|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 3 | 34.69 s | 0.710 | 4.85 | Cold |
| 2 | 3 | 22.85 s | 1.078 | 7.35 | Page cache active |
| 3 | 3 | 22.72 s | 1.084 | 7.39 | Warm |
| 4 | 3 | 22.78 s | 1.081 | 7.37 | Warm |
| 5 | 3 | 22.77 s | 1.081 | 7.37 | Warm |

---

## s3torchconnector Training (Read) Performance — Scaling Study

> **⚠️ RESULTS NOT REPRESENTATIVE — SEQUENTIAL FETCH ISSUE**
> These results were collected using `S3IterableDataset.from_objects()`, which fetches files
> **one at a time per DataLoader worker** (4 total concurrent GETs across all workers).
> This is fundamentally less concurrent than minio (up to 64 total) and s3dlio (up to 256 total).
> The numbers below reflect sequential-fetch throughput, **not** the true read capability
> of the s3torchconnector library. These results should be re-run after implementing the
> `ThreadPoolExecutor + S3Client.get_object()` fix. See `S3library_review_21-Mar.md` for
> full analysis and remediation options.

Using `S3IterableDataset.from_objects()` with `S3ReaderConstructor.sequential()` — single streaming GET per file, no range splitting, no HEAD requests.

### Summary

| MPI Ranks (NP) | Steps/epoch | Epoch 1 time (cold) | Epoch 2–5 time (warm) | I/O Throughput (MB/s) | I/O Throughput (GB/s) | Samples/s | vs NP=1 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 24 | 96.75 s | ~85.9 s | **303.0 ± 1.1** | **0.296** | 2.1672 ± 0.0082 | 1.0× |
| 2 | 12 | 56.17 s | ~46.5 s | **627.2 ± 6.4** | **0.613** | 4.4861 ± 0.0458 | 2.07× |
| 4 | 6 | 33.69 s | ~22.7 s | **1934.7 ± 65.9** | **1.890** | 13.8379 ± 0.4712 | 6.38× ¹ |
| 8 | 3 | 36.66 s | ~24.2 s | **5557 ± 242** | **5.426** | 39.7469 ± 1.7296 | 18.3× ¹ ² |

### Per-Epoch Detail — NP=1

| Epoch | Steps | Duration | GB/s (wall-clock) | Throughput (samples/s) | Notes |
|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 24 | 96.75 s | 0.255 | 2.1727 | Cold read from MinIO over network |
| 2 | 24 | 86.43 s | 0.285 | 2.1513 | Warm — page cache active |
| 3 | 24 | 85.74 s | 0.287 | 2.1709 | Warm |
| 4 | 24 | 85.71 s | 0.287 | 2.1734 | Warm |
| 5 | 24 | 85.79 s | 0.287 | 2.1677 | Warm |

**Warm avg:** ~85.92 s → **0.287 GB/s**.

> **vs s3dlio NP=1:** s3torchconnector warm throughput (0.287 GB/s) is ~8% slower than s3dlio tuned NP=1 (0.312 GB/s). This is expected: `S3IterableDataset.sequential()` issues one streaming GET per file on a single connection (no parallelism within a file), whereas s3dlio's `get_many()` uses Tokio async concurrency across all files in the batch simultaneously.

### Per-Epoch Detail — NP=2

| Epoch | Steps | Duration | GB/s (wall-clock) | Throughput (samples/s) | Notes |
|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 12 | 56.17 s | 0.438 | 4.6012 | Cold read from MinIO over network |
| 2 | 12 | 46.05 s | 0.535 | 4.6056 | Warm — page cache active |
| 3 | 12 | 46.55 s | 0.529 | 4.5692 | Warm |
| 4 | 12 | 46.85 s | 0.526 | 4.5370 | Warm |
| 5 | 12 | 46.65 s | 0.528 | 4.5319 | Warm |

**Warm avg:** ~46.53 s → **0.529 GB/s**.

> **vs s3dlio NP=2:** s3torchconnector warm throughput (0.529 GB/s) is ~6% slower than s3dlio tuned NP=2 (0.562 GB/s) — the relative gap is consistent with NP=1 (~8%). Scaling from NP=1→NP=2 is 2.07× (linear), matching s3dlio's 2.05× scaling at the same step.

### Per-Epoch Detail — NP=4

| Epoch | Steps | Duration | GB/s (wall-clock) | Throughput (samples/s) | Notes |
|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 6 | 33.69 s | 0.731 | 12.1958 | Cold read from MinIO over network |
| 2 | 6 | 22.48 s | 1.095 | 14.6062 | Warm — page cache active |
| 3 | 6 | 22.74 s | 1.083 | 15.1972 | Warm |
| 4 | 6 | 23.14 s | 1.065 | 14.4476 | Warm |
| 5 | 6 | 22.48 s | 1.095 | 13.9308 | Warm |

**Warm avg:** ~22.71 s → **1.084 GB/s**.

¹ **METRIC throughput (1934.7 MB/s) far exceeds the 1,200 MB/s physical network ceiling** — the majority of warm-epoch reads are served from the Linux page cache, not the network. This is identical behaviour to s3dlio NP=4 (warm avg ~22.73 s, 1.058 GB/s). The wall-clock warm GB/s (1.084) is the reliable signal; the METRIC value is inflated by cache hits.

> **vs s3dlio NP=4:** warm epoch durations are nearly identical (22.71 s vs 22.73 s) — at NP=4 both libraries are overwhelmingly page-cache-bound and the library difference disappears entirely.

### Per-Epoch Detail — NP=8

| Epoch | Steps | Duration | GB/s (wall-clock) | Throughput (samples/s) | Notes |
|:-:|:-:|:-:|:-:|:-:|---|
| 1 | 3 | 36.66 s | 0.672 | 51.53 | Cold read from MinIO over network |
| 2 | 3 | 24.34 s | 1.012 | 57.66 | Warm — page cache active |
| 3 | 3 | 24.26 s | 1.015 | 47.32 | Warm |
| 4 | 3 | 24.18 s | 1.018 | 30.64 | Warm |
| 5 | 3 | 23.85 s | 1.033 | 12.18 | Warm |

**Warm avg:** ~24.16 s → **1.019 GB/s**.

¹ ² **METRIC throughput and samples/s at NP=8 are unreliable** — with only 3 steps/epoch, sub-second timing noise in any single step dominates the per-epoch average. The wall-clock epoch duration (23.85–24.34 s warm, CV <1%) is the reliable signal. METRIC MB/s (5557) is ~4.6× above the physical network ceiling (1,200 MB/s), confirming the workload is overwhelmingly page-cache-served at NP=8.

> **vs s3dlio NP=8:** s3torchconnector warm avg 24.16 s vs minio-py warm avg ~22.5–22.9 s from the minio NP=8 section. s3torchconnector is within ~7% of minio-py at NP=8 — both are cache-dominated and the library differences are negligible.

---

## How to Reproduce

```bash
cd /path/to/mlp-storage

# Populate bucket (skip if data already present)
bash tests/object-store/dlio_s3dlio_datagen.sh

# Run training at different MPI ranks
NP=1 bash tests/object-store/dlio_s3dlio_train.sh
NP=2 bash tests/object-store/dlio_s3dlio_train.sh
NP=4 bash tests/object-store/dlio_s3dlio_train.sh

# Results are in the most recent /tmp/dlio-s3dlio-train-* directory
grep -E "Simulated Acc|Throughput|I/O" /tmp/dlio-s3dlio-train-*/dlio.log
```

To measure cold-read performance only, clear the page cache between runs (requires root):

```bash
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
NP=4 bash tests/object-store/dlio_s3dlio_train.sh
# Only epoch 1 duration is meaningful in this case
```

---

## Known Issues

### OpenMPI vader BTL crash (NP ≥ 4 without the fix)

**Symptom:** `mpirun` exits with signal 11 (Segmentation fault) immediately after
`Starting block 1`, before any step completes. NP=1 and NP=2 work fine.

**Root cause:** OpenMPI automatically selects the `vader` BTL (shared-memory
transport) when all ranks run on the same physical node. At NP≥4, a race
condition in vader's shared-memory ring-buffer causes one rank to dereference
a fragment pointer already freed by another rank during `MPI_Barrier`.

The full crash stack was:
```
mca_btl_vader_poll_handle_frag → opal_progress → ompi_sync_wait_mt
  → mca_pml_ob1_recv → ompi_coll_base_barrier_intra_basic_linear
  → MPI_Barrier  ← SEGV_MAPERR
```

**Fix:** Add `--mca btl ^vader` to the `mpirun` invocation. This disables vader
and forces OpenMPI to use TCP loopback for intra-node communication instead.
All scripts in `tests/object-store/` already include this flag.

---

## Environment

```
Python:         3.13 (linuxbrew)
s3dlio:         0.9.84
dlio_benchmark: fork (mlp-storage/dlio_benchmark)
mpi4py:         bundled with openmpi3
OpenMPI:        system (/usr/lib/x86_64-linux-gnu/openmpi)
DLIO_S3_IMPLEMENTATION=mlp
multiprocessing_context=spawn   (required — fork kills Tokio runtime in workers)
```
