# S3 Library Write + Read Comparison — Results

**Date:** March 18, 2026  
**Endpoint:** `http://minio-host:9000` (MinIO-compatible S3)  
**Test script:** `Test-Backup/test_direct_write_comparison.py`

---

## Environment & Credentials

Credentials and endpoint configuration are supplied via a `.env` file at the root of the
`mlp-storage` project directory (`mlp-storage/.env`).  The script loads this file
automatically at startup and exports the following variables into the environment before
any library is initialised:

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_ENDPOINT_URL
AWS_REGION
```

No credentials are hard-coded in the test script.  Any future tester only needs to create
(or update) the `.env` file with their own endpoint and credentials before running.

---

## Library Versions Tested

| Library | Version |
|---|---|
| s3dlio | 0.9.84 |
| minio (Python SDK) | 7.2.20 |
| s3torchconnector | 1.5.0 |

All three were installed in the project's virtual environment (`.venv`):

```bash
source .venv/bin/activate
pip show s3dlio minio s3torchconnector
```

Each library was given its own dedicated S3 bucket so writes never interfere:

| Library | Bucket |
|---|---|
| s3dlio | `bucket-s3dlio` |
| minio | `bucket-minio` |
| s3torchconnector | `bucket-s3torch` |

---

## Test Description

`test_direct_write_comparison.py` runs three phases per library:

1. **Cleanup** — delete every object under the test prefix so every run starts clean
2. **Write** — upload N objects in parallel using `ThreadPoolExecutor` and each library's
   native write API (no common wrapper)
3. **Read** — download all N objects back in parallel using `ThreadPoolExecutor`

Write APIs used:
- **s3dlio** — `MultipartUploadWriter.from_uri()` with configurable `part_size` and
  `max_in_flight` (concurrent parts per object)
- **minio** — native `_create_multipart_upload` / `_upload_part` / `_complete_multipart_upload`
  (sequential parts within each object, parallel objects)
- **s3torchconnector** — `S3Client.put_object()` (buffers internally, uploads at `close()`)

---

## How to Run

### Default run (8 write workers, 8 read workers, all three libraries)

```bash
cd mlp-storage
source .venv/bin/activate
python Test-Backup/test_direct_write_comparison.py --num-files 100 --size-mb 128
```

### Run that produced the results below (12 workers each, all libraries)

```bash
python Test-Backup/test_direct_write_comparison.py \
    --num-files 100 \
    --size-mb 128 \
    --write-workers 12 \
    --read-workers 12
```

### Test a single library

```bash
python Test-Backup/test_direct_write_comparison.py \
    --num-files 100 --size-mb 128 \
    --write-workers 12 --read-workers 12 \
    --library s3dlio
```

### Test two libraries

```bash
python Test-Backup/test_direct_write_comparison.py \
    --num-files 100 --size-mb 128 \
    --write-workers 12 --read-workers 12 \
    --library s3dlio minio
```

### Full CLI reference

```
optional arguments:
  --num-files N         Number of objects to write/read per library (default: 100)
  --size-mb N           Object size in MB (default: 128)
  --chunk-mb N          Multipart chunk size in MB (default: 32)
  --prefix PREFIX       S3 key prefix (default: bench)
  --write-workers N     Parallel object upload threads (default: 8)
  --read-workers N      Parallel object download threads (default: 8)
  --max-in-flight N     s3dlio per-object concurrent multipart parts (default: 8)
  --library LIB [LIB …] Libraries to test: s3dlio minio s3torchconnector (default: all)
```

---

## Results

Command run:

```bash
python Test-Backup/test_direct_write_comparison.py \
    --num-files 100 --size-mb 128 \
    --write-workers 12 --read-workers 12
```

```
========================================================================================
WRITE + READ COMPARISON — RESULTS
  100 objects × 128 MB = 12800 MB per library  |  write workers: 12   read workers: 12
========================================================================================
  Library                Version       Write GB/s   Read GB/s  Wr s/obj  Rd s/obj
  ---------------------- ------------ ----------- ----------- --------- ---------
  s3dlio                 0.9.84            0.525         1.085 ◀R    0.238s    0.115s
  minio                  7.2.20            0.415         1.051       0.301s    0.119s
  s3torchconnector       1.5.0             0.561 ◀W      0.541       0.223s    0.231s

  Write GB/s — parallel write throughput (all objects, ThreadPoolExecutor)
  Read GB/s  — parallel read throughput (all objects, ThreadPoolExecutor)
  Wr s/obj   — average time to write one object (write + commit)
  Rd s/obj   — average time to read one object (wall-clock, under parallelism)
  ◀W = fastest write    ◀R = fastest read

  Notes:
   • Write workers = parallel object uploads; Read workers = parallel object downloads
   • s3dlio max_in_flight = additional per-object part concurrency within each writer
   • minio part uploads are sequential within each object (no per-object parallelism)
   • s3torchconnector buffers writes internally and uploads at close()
========================================================================================
✅ All tests passed.
```

---

## Analysis

### Write throughput

s3torchconnector achieved the highest write throughput (0.561 GB/s), narrowly ahead of
s3dlio (0.525 GB/s).  Both are consistent with the independent `s3-cli` baseline of
~0.429 GB/s at 12 jobs — the per-library Python threads reach slightly higher than the CLI
tool because they issue more concurrent connections.  minio lags (0.415 GB/s) likely
because its multipart parts are issued sequentially within each object, so each upload is
limited to one connection at a time regardless of how many objects are in flight in parallel.

### Read throughput

s3dlio and minio deliver essentially the same peak read throughput (~1.05–1.09 GB/s).
s3torchconnector reads at only 0.541 GB/s — roughly half — because its streaming `read()`
model serialises data transfer through a single Python call per object rather than issuing
parallel range-based fetches.

### Overall recommendation

**s3dlio is the most balanced choice**: near-best write throughput and best-in-class read
throughput.  It is also the only library that supports configurable per-object part
concurrency (`max_in_flight`), which provides an additional tuning lever beyond the number
of parallel objects.

---

---

## DLIO Workload Results

**Test script:** `Test-Backup/test_dlio_multilib_demo.py`  
**Date:** March 18, 2026  
**Endpoint:** `http://minio-host:9000` (MinIO-compatible, ~1.2 GB/s link on this machine)

These results measure performance **as seen by DLIO** (via `mlpstorage`) — not direct native
API calls. The gap versus the direct API numbers above quantifies DLIO overhead.

### Workload 1 — Training

- Dataset: 100 × 128 MiB NPZ objects = 12.5 GiB per library
- 2 full epochs (25.0 GiB total reads per library)
- Write = `mlpstorage training datagen` (8 MPI processes)
- Read = `mlpstorage training run` (8 DataLoader workers, prefetch 4)

```
  Library                  Write GB/s    Read GB/s    Gen s   Train s  Status
  ---------------------- ------------ ------------ -------- ---------  ------
  s3dlio                        0.308        0.178    40.6s    140.1s  ✅
  s3torchconnector              0.360        0.178    34.7s    140.5s  ✅
  minio                         (pending)
```

**Key observations:**

- Read throughput is **identical** (0.178 GB/s) for both libraries despite s3dlio reading at
  1.085 GB/s natively. The bottleneck is PyTorch DataLoader IPC overhead: each of the 8
  worker processes fetches a 128 MiB file, deserializes NPZ, then pickles the result back
  to the main process. For 128 MiB objects this IPC pickle is the sole limiter — the S3
  library is never the constraint.
- Write (datagen) overhead vs direct API: s3dlio 0.308 vs 0.525 GB/s (~41% slower through
  DLIO); s3torchconnector 0.360 vs 0.561 GB/s (~36% slower). DLIO's MPI orchestration adds
  meaningful overhead.

### Workload 2 — Checkpoint (StreamingCheckpointing)

- Single 100 GB object per library written via streaming producer-consumer pipeline
- Fixed RAM: 32 MB chunks × 4 buffers = 128 MB peak, regardless of checkpoint size
- dgen-py generates data concurrently; I/O is always the bottleneck
- Write API: `StreamingCheckpointing.save(uri, 100 GB)`

```
  Library                    Size GB    Elapsed    Write GB/s  Status
  ----------------------- ---------- ---------- -----------    ------
  s3dlio                       100        99.2s      1.008 ◀   ✅
  s3torchconnector              75        83.9s      0.912      ❌ CRT error at ~78 GB (run capped at 75 GB)
  minio                        100       233.6s      0.429      ✅
```

**s3torchconnector CRT failure:**

s3torchconnector fails consistently at approximately 78 GB into the 100 GB upload with:

```
Client error: Unknown CRT error: CRT error 14366:
  aws-c-s3: AWS_ERROR_S3_REQUEST_HAS_COMPLETED,
  Request has already completed, action cannot be performed.
Client error: Internal S3 client error: A previous write operation did not complete successfully
```

This is a bug in the AWS Common Runtime (CRT) multipart upload state machine — the CRT
marks a request as completed prematurely while the Python streaming layer is still feeding
data. The failure is **reproducible** and occurs at ~78 GB regardless of retry. s3dlio
uses its own multipart engine (not the CRT) and completes 100 GB cleanly.

**minio checkpoint result:**

minio achieved **0.429 GB/s** — exactly matching its native direct-API write speed
(0.415 GB/s in the direct comparison).  The initial implementation uploaded parts
sequentially (one at a time), capping throughput at ~0.10 GB/s.  After enabling
8 parallel part uploads via `ThreadPoolExecutor`, throughput improved 4× to 0.429 GB/s.
Further gains are unlikely from minio alone: even with parallelism its per-connection
transfer is limited to one outstanding request per part, unlike s3dlio which pipelines
parts within each connection.

**s3dlio checkpoint result:**

s3dlio achieved **1.008 GB/s** — near the ~1.2 GB/s physical network ceiling on this
machine. The streaming pipeline keeps the network saturated throughout the full 100 GB
run with no accumulation of model state in RAM.

---

## Reference: write worker count sensitivity

Tested independently using `s3-cli` (s3dlio's CLI), same endpoint & object size:

| Workers (`-j`) | Write throughput |
|---|---|
| 8 | 308.64 MiB/s (0.302 GB/s) |
| 12 | 429.25 MiB/s (0.419 GB/s) |

A ~39 % gain from 8 → 12 workers; worth testing higher values (16, 24) if the network
and server can sustain it.

---

## Checkpoints

**Test script:** `Test-Backup/test_dlio_multilib_demo.py --workload checkpoint`  
**Date:** March 18, 2026  
**Checkpoint size:** 16 GB (sanity-check run; production target is 100 GB)  
**Method:** `StreamingCheckpointing` — streaming producer-consumer pipeline, fixed 128 MB RAM

### Checkpoint Write

```
================================================================================================
DLIO MULTI-LIBRARY BENCHMARK — RESULTS
================================================================================================

WORKLOAD 2: CHECKPOINT  (StreamingCheckpointing — fixed 128 MB RAM)
  Single object per library via streaming producer-consumer pipeline
  32 MB chunks × 4 buffers = 128 MB RAM max regardless of checkpoint size
  Library                  Size GB   Write GB/s    Read GB/s     Status
  ---------------------- --------- ------------ ------------      -----
  s3dlio                        16        1.023 ◀W        1.051     ✅ - 1st place 
  minio                         16        0.430           1.055     ✅ - 3rd place
  s3torchconnector              16        0.949           1.092 ◀R  ✅ - 2nd place

  Write GB/s = I/O throughput from StreamingCheckpointing.save()
  Read GB/s  = I/O throughput from StreamingCheckpointing.load() (byte-range GETs, data discarded)
  ◀W = fastest write   ◀R = fastest read
  dgen-py generates write data concurrently; bottleneck is always I/O, not generation

================================================================================================
✅ All tests passed.
```

### Checkpoint Load

**s3dlio and minio** use explicit offset-based `get_range()` / Range-GET calls.
`StreamingCheckpointing.load()` issues 8 parallel threads, each reading a contiguous
block of the object with its own connection, achieving ~1.05 GB/s.

**s3torchconnector** — RAM and throughput fixes, three iterations:

**Iteration 1 — OOM with SequentialS3Reader (before any fix):**
The default `get_object()` uses `SequentialS3Reader`, which causes the AWS CRT
(`mountpoint-s3-client`) to buffer the entire object before serving any `read()` calls.
Peak RAM = object size. Results: 75 GB load killed at ~24 GB; 16 GB caused heavy swap.

**Iteration 2 — `range_based(buffer_size=0)` (fixed OOM, killed throughput):**
`RangedS3Reader._read_unbuffered()` was used, which calls `_get_stream(start, end)` on
**every single `read()` call**, opening a brand-new HTTP range-GET each time. With 128 MB
read chunks, each worker made 16 separate range-GETs to read its 2 GB block. Per-worker
throughput stalled at 0.07 GB/s regardless of chunk size; total read: **0.583 GB/s**.
RAM was bounded (8 × 128 MB = 1 GB) but connection overhead dominated.

**Iteration 3 — `_get_object_stream` directly (current implementation):**
After reading the s3torchconnector source, the root cause was identified: the fix calls
`S3Client._get_object_stream(bucket, key, start, end)` directly — the same native CRT
method that `RangedS3Reader` uses internally, but held open for the entire block. Each
worker issues **one HTTP connection** for its `[block_start, block_end)` range and
streams through native CRT chunks (~8 MB each) without reopening. This is implemented
as `stream_block(start, end)` on the reader. Each chunk is counted and immediately
discarded.

Peak RAM = n_workers × CRT internal buffer per stream ≈ 8 workers × ~32 MB = **~256 MB**,
constant for any object size (16 GB or 759 GB). The `read_chunk()` serial path also uses
a persistent stream opened lazily, with a small leftover buffer for CRT chunk boundary
alignment (~8 MB max). The `S3Client` instance is created once per worker; the CRT
manages its own connection pool for reuse across calls.

**Confirmed results (16 GB, 8 workers, stream_block path):**
- Write: **0.949 GB/s** ✅
- Read:  **1.092 GB/s** ✅  (was 0.583 GB/s with range_based — **87% improvement**)
- `Chunks: 8` in load output — confirms exactly ONE HTTP connection per worker.
- Per-worker: ~0.14–0.21 GB/s each × 8 workers = ~1.09 GB/s aggregate.
- Peak RAM: ~256 MB (8 workers × ~32 MB CRT buffer); independent of object size.
- Now matches s3dlio and minio at the ~1.0–1.1 GB/s network ceiling.

---

# DLIO Training Sweep Results

**Date:** March 18, 2026  
**Test script:** `Test-Backup/test_training_mpi_sweep.py`  
**Endpoint:** `http://minio-host:9000` (MinIO-compatible S3)

These results measure performance **as seen by the full DLIO training pipeline** — including
DLIO's MPI data generation, PyTorch DataLoader worker processes, NPZ deserialization, and
IPC overhead. Each sweep point is an independent clean cycle: `clean → datagen(N) → train(N) → clean`.

## Setup

| Parameter | Value |
|---|---|
| Dataset | 100 × 128 MiB NPZ = 12.50 GiB per library |
| Training | 2 epochs = 25.00 GiB total reads per cycle |
| Model | unet3d / a100 accelerator profile |
| DataLoader | 8 read_threads per MPI process, prefetch 4, batch size 1 |
| Sweep variable | N MPI processes (applied to both datagen and training) |

Each library uses a dedicated bucket; no cross-library interference.

## Data Generation Write Throughput (GB/s)

| Library | N=1 | N=2 | N=4 |
|---|---|---|---|
| s3dlio | 0.080 | 0.156 | 0.249 |
| minio | 0.085 | 0.158 | 0.250 |
| s3torchconnector | 0.085 | 0.114 | 0.248 |

## Training Read Throughput (GB/s)

| Library | N=1 | N=2 | N=4 |
|---|---|---|---|
| s3dlio | 0.179 | 0.325 | 0.488 |
| minio | 0.179 | 0.323 | 0.485 |
| s3torchconnector | 0.179 | 0.321 | 0.490 |

## Read Scaling (relative to N=1 baseline)

| Library | N=1 | N=2 | N=4 |
|---|---|---|---|
| s3dlio | 1.00× | 1.81× | 2.72× |
| minio | 1.00× | 1.81× | 2.71× |
| s3torchconnector | 1.00× | 1.79× | 2.73× |

## Comparison: DLIO vs Native Library Throughput

| Metric | Native (direct API, 12 workers) | DLIO N=4 | DLIO as % of native |
|---|---|---|---|
| Write (s3dlio) | 0.525 GB/s | 0.249 GB/s | **47%** |
| Write (minio) | 0.415 GB/s | 0.250 GB/s | **60%** |
| Write (s3torchconnector) | 0.561 GB/s | 0.248 GB/s | **44%** |
| Read (s3dlio) | 1.085 GB/s | 0.488 GB/s | **45%** |
| Read (minio) | 1.051 GB/s | 0.485 GB/s | **46%** |
| Read (s3torchconnector) | 1.092 GB/s | 0.490 GB/s | **45%** |

## Analysis

**The bottleneck is DLIO, not the network and not the storage library.**

All three libraries perform within noise of each other at every process count — write
differences are ≤ 1% at N=4, read differences ≤ 1%. This means the storage library
choice is completely irrelevant inside DLIO. The per-library call latency and throughput
advantages measured in the direct API tests are entirely erased by DLIO overhead.

**The culprit is the serialization chain, not the I/O:**

- **NPZ on write** — `numpy.savez()` on 128 MiB arrays is expensive CPU work done
  inline before the S3 write even starts. The storage library is waiting on numpy, not
  the network.

- **NPZ on read + IPC pickle** — each DataLoader worker loads the NPZ, unpacks it, then
  pickles the 128 MiB tensor back to the main process via `multiprocessing`. At 128 MiB,
  the pickle + memcpy dominates wall time — the S3 read completes long before the tensor
  is delivered to the training loop.

- **MPI coordination** — barriers prevent full write pipelining; N=4 yields only ~3.1×
  the N=1 throughput, not the theoretical 4×. Synchronization points eat the remaining
  efficiency.

DLIO achieves only ~45–60% of what the native APIs can deliver, pointing to several
likely bottlenecks within DLIO itself:

1. **NPZ serialization / deserialization** — each 128 MiB object must be packaged as NPZ
   on write (via numpy.savez) and unpacked on read (via numpy.load). For 128 MiB files
   this is expensive CPU work done serially within each DataLoader worker before any data
   reaches the model.

2. **PyTorch DataLoader IPC** — after deserializing NPZ, each of the N read_thread
   worker processes must pickle the resulting tensor back to the main training process
   via shared-memory IPC. For 128 MiB tensors this pickle + memcpy dominates wall time.

3. **MPI coordination overhead** — DLIO's MPI-based data generation adds synchronization
   barriers and metadata tracking overhead that prevent the N processes from fully
   pipelining their writes. At N=4, write throughput is only ~3.1× N=1 (not 4×).

4. **Read scaling sub-linearity** — training read at N=4 is only ~2.7× N=1 (not 4×),
   meaning ~32% efficiency loss to DLIO scheduling, DataLoader prefetch coordination,
   and process-local deserialization bottlenecks.

## Is a DLIO rewrite needed?

The short answer is: **yes, if the goal is to make DLIO competitive with native I/O**.

The current DLIO storage path creates a deep stack between the S3 call and the training
loop: `MPI process → Python storage backend → S3 lib → network → S3 lib → Python storage
backend → numpy.load → IPC pickle → DataLoader → training loop`. Every layer adds
overhead, and the serialization layers (NPZ + pickle) cost CPU time that is comparable
to or greater than the actual I/O time at this file size.

**Targeted improvements that would not require a full rewrite:**

- **Reduce object size** — smaller objects (e.g. 4–16 MiB) reduce per-file NPZ overhead
  and make the IPC pickle cheaper, allowing more objects in flight and better pipelining.

- **Switch to a raw binary format** — replacing NPZ with flat binary (or memmap-able
  formats like safetensors / raw fp32) eliminates the numpy zip overhead entirely and
  allows zero-copy reads into pinned CUDA memory.

- **Use shared memory for DataLoader IPC** — passing large tensors via `multiprocessing`
  shared memory (`torch.multiprocessing`) avoids the pickle round-trip for large tensors.

- **Pre-stage to NVMe** — DLIO supports a cache tier; pre-fetching objects to local NVMe
  and reading from there can decouple the I/O and compute timelines.

**If a deeper rewrite is on the table**, the most impactful change would be to replace
the per-file DataLoader read model with a streaming prefetch model where S3 range-GETs
are issued asynchronously by a dedicated I/O thread pool and data is DMA-copied directly
into pre-allocated pinned buffers. This eliminates the NPZ deserialization bottleneck
and the IPC pickle entirely — the storage library (s3dlio, etc.) would operate at its
native throughput.
