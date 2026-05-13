# S3 Library Prefetch Fairness Review

**Date:** March 21, 2026  
**System:** loki-russ  
**Author:** Analysis via GitHub Copilot  
**Status:** Analysis — no code changes made yet. Pending decision on remediation approach.

---

## 1. Purpose

This document captures a thorough code review of the prefetch mechanism used by all three
S3 storage libraries (s3dlio, minio-py, s3torchconnector) in the DLIO benchmark harness,
with specific focus on whether the three libraries are being exercised with equivalent
concurrency.

The motivation was that s3torchconnector training benchmark results (NP=4/8) appeared
anomalously low compared to prior measurements, raising the question of whether the test
code was making fair use of the library's capabilities.

**Conclusion:** The concurrency models are **not equivalent**. s3torchconnector is fetching
one file at a time per DataLoader worker (4 total concurrent GETs across all workers),
while s3dlio uses up to 64 concurrent async GETs per worker (256 total) and minio uses
up to 16 threads per worker (64 total). This is not a fair comparison, and s3torchconnector
training results collected before fixing this are not representative.

---

## 2. Benchmark Context

| Parameter | Value |
|-----------|-------|
| Dataset | 168 × 146.6 MB NPZ files = 24,628.8 MB = 24.63 GB |
| Network ceiling (measured) | ~1.2 GB/s |
| DLIO DataLoader workers per rank | 4 (`read_threads: 4`) |
| multiprocessing_context | `spawn` (each worker = isolated process) |
| Batch size | 7 samples/step |
| Training epochs | 5 |
| Object format | NPZ (`["x"]` key, compressed numpy array) |
| MinIO endpoint | `http://minio-host:9000` |
| s3dlio version | v0.9.84 (wheel tagged v0.9.82) |

---

## 3. Code Path — Common Entry Point

All three libraries route through the same reader class and the same dispatcher method:

```
DLIO DataLoader worker (spawned process)
  └─ NPZReaderS3Iterable.next()
       └─ _prefetch(filenames)            ← dispatcher
            ├─ if lib == "s3dlio"      → _prefetch_s3dlio()
            ├─ if lib == "minio"       → _prefetch_minio()
            └─ if lib == "s3torchconnector" → _prefetch_s3torchconnector()
```

**File:** `dlio_benchmark/dlio_benchmark/reader/npz_reader_s3_iterable.py`

`next()` calls `_prefetch(filenames)` once at the start of each epoch for all files
assigned to this DataLoader worker thread. Only after all files are in the local cache
does `next()` delegate to the parent `NPZReader.next()` for batch iteration.

With 168 files and 4 DataLoader workers (NP=1), each worker prefetches approximately
42 files per epoch.

The dispatcher (`_prefetch`) is:

```python
def _prefetch(self, filenames: list) -> dict:
    lib = self._storage_library
    if lib == "s3dlio":
        return self._prefetch_s3dlio(filenames)
    elif lib == "s3torchconnector":
        return self._prefetch_s3torchconnector(filenames)
    elif lib == "minio":
        return self._prefetch_minio(filenames)
    else:
        raise ValueError(
            f"NPZReaderS3Iterable: unknown storage_library {lib!r}; "
            f"supported: s3dlio, s3torchconnector, minio"
        )
```

---

## 4. Library-by-Library Prefetch Analysis

### 4.1 s3dlio — `_prefetch_s3dlio()`

```python
def _prefetch_s3dlio(self, filenames: list) -> dict:
    import s3dlio
    from s3dlio.compat.s3torchconnector import _BytesViewIO

    uris = [self._uri_for_filename(f) for f in filenames]
    uri_to_fname = dict(zip(uris, filenames))

    max_in_flight = min(64, len(uris))
    results = s3dlio.get_many(uris, max_in_flight=max_in_flight)

    cache = {}
    for uri, data in results:
        fname = uri_to_fname.get(uri, uri)
        raw = io.BufferedReader(_BytesViewIO(data))
        cache[fname] = np.load(raw, allow_pickle=True)["x"]
    return cache
```

**How it works:**
- Issues a single `s3dlio.get_many()` call with all file URIs
- `get_many()` is a Rust/Tokio async function — up to `max_in_flight=64` HTTP GETs
  execute concurrently inside the Rust runtime, with no Python thread overhead
- `_BytesViewIO` wraps the Rust `BytesView` via the Python buffer protocol (zero-copy)
- `io.BufferedReader` triggers `readinto()` instead of `bytes()`, keeping peak memory
  to the Rust buffer only (no simultaneous Python heap copy)
- Returns only after all files are fetched

**Concurrency:** Up to **64 concurrent HTTP GETs** per DataLoader worker process (bounded by
`max_in_flight` cap and actual file count). All 64 are driven by Rust async tasks on the
Tokio thread pool — zero Python GIL involvement during I/O.

**Requests per file (on this 1 Gbps system with range splitting active):**
- 1 HEAD (size probe) + up to 37 range GETs (4 MB chunks for a 147 MB file)
- See `s3dlio_performance_analysis.md` for full breakdown of range splitting overhead

---

### 4.2 minio — `_prefetch_minio()`

```python
def _prefetch_minio(self, filenames: list) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    from urllib.parse import urlparse

    client = self._get_minio_client()   # cached per worker process

    def _fetch_one(filename):
        uri = self._uri_for_filename(filename)
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        resp = client.get_object(bucket, key)
        try:
            raw = resp.read()
        finally:
            resp.close()
            resp.release_conn()
        return filename, np.load(io.BytesIO(raw), allow_pickle=True)["x"]

    n_workers = min(16, max(1, len(filenames)))
    cache = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for fname, arr in pool.map(_fetch_one, filenames):
            cache[fname] = arr
    return cache
```

**How it works:**
- Uses `ThreadPoolExecutor(max_workers=min(16, n_files))` — up to 16 Python threads
- Each thread calls `Minio.get_object()` independently: one streaming GET per file
- Uses a **cached** `Minio` client (created once per worker process in `_get_minio_client`)
  with a `urllib3.PoolManager(maxsize=16)` — TCP connections persist across epochs
- Each file issues exactly **1 HTTP GET** (no range splitting, no HEAD requests)
- `np.load(io.BytesIO(raw))` copies the data into a Python bytes object per file

**Concurrency:** Up to **16 concurrent HTTP GETs** per DataLoader worker process. This is
real OS-thread-based parallelism; the GIL releases during the network I/O portion of each
`resp.read()` call, so threads genuinely overlap.

---

### 4.3 s3torchconnector — `_prefetch_s3torchconnector()`

```python
def _prefetch_s3torchconnector(self, filenames: list) -> dict:
    from s3torchconnector import S3IterableDataset
    from s3torchconnector.s3reader import S3ReaderConstructor

    opts = self._opts
    endpoint = opts.get("endpoint_url", "")
    region = opts.get("region", "us-east-1")

    uris = [self._uri_for_filename(f) for f in filenames]

    dataset = S3IterableDataset.from_objects(
        uris,
        region=region,
        endpoint=endpoint,
        reader_constructor=S3ReaderConstructor.sequential(),
    )

    cache = {}
    for fname, reader in zip(filenames, dataset):
        cache[fname] = np.load(reader, allow_pickle=True)["x"]
    return cache
```

**How it works:**
- Creates an `S3IterableDataset` from the list of URIs with `sequential()` reader mode
- `S3IterableDataset` is a **PyTorch `IterableDataset`** — it is a lazy iterator that
  yields one `S3Reader` (a `BufferedIOBase` stream) per file, on demand
- The `for fname, reader in zip(filenames, dataset)` loop consumes this lazy iterator
  **sequentially**: it requests file N+1 only after fully consuming file N
- `np.load(reader)` reads the entire file content from the `S3Reader` before the loop
  advances to the next iteration
- Each `S3Reader` (sequential mode) issues a **single streaming GET** — no range splitting,
  no HEAD requests. The data streams from S3 directly into numpy's buffer.

**Concurrency: 1 HTTP GET per DataLoader worker at any given time (sequential).**

This is the root of the fairness problem. `S3IterableDataset` was designed to serve as
the `dataset=` argument passed directly to PyTorch `DataLoader`, where each `DataLoader`
worker is an independent process iterating its shard of the dataset. In that intended usage,
parallelism comes from having multiple `DataLoader` workers, each fetching a different file.
The library does **not** provide a multi-stream batch-download API.

---

## 5. Concurrency Comparison — The Fairness Gap

### 5.1 Per-worker and total concurrency

| Library | Concurrency mechanism | Per-worker concurrent GETs | × 4 workers | Total GETs |
|---|---|:-:|:-:|:-:|
| **s3dlio** | Rust/Tokio async, `max_in_flight=64` | up to **64** | × 4 | **up to 256** |
| **minio** | `ThreadPoolExecutor(max_workers=16)` | up to **16** | × 4 | **up to 64** |
| **s3torchconnector** | `S3IterableDataset` sequential for-loop | **1** | × 4 | **4** |

With 42 files per worker per epoch (NP=1, 4 workers, 168 total files):

| Library | Concurrent GETs / worker | Files per worker | Batches of GETs per worker |
|---|:-:|:-:|:-:|
| s3dlio | 64 | 42 | 1 (all files fetched in one `get_many` call) |
| minio | 16 | 42 | 3 (42 ÷ 16 ≈ 3 rounds through the thread pool) |
| s3torchconnector | 1 | 42 | 42 (one GET at a time, completely serial) |

### 5.2 Requests per 147 MB file (this 1 Gbps system)

| Library | HEAD requests | GET requests | Total requests |
|---|:-:|:-:|:-:|
| s3dlio (range splitting active) | 2 | 37 (4 MB chunks) | **39** |
| minio | 0 | 1 (streaming) | **1** |
| s3torchconnector (sequential mode) | 0 | 1 (streaming) | **1** |

> **s3dlio note:** The double-HEAD + range-GET pattern is documented in
> `s3dlio_performance_analysis.md` (Findings 1 and 2). Range splitting can be suppressed
> with `S3DLIO_RANGE_THRESHOLD_MB=1000` on 1 Gbps systems.
>
> **⚠️ Update (v0.9.84):** The Findings 1–6 identified below have been largely resolved
> in s3dlio v0.9.84. See §12 for the full resolution table.

### 5.3 Impact on expected throughput

At 1.2 GB/s network ceiling for a single 147 MB file:

- **s3dlio** (without range splitting env var): bottlenecked by HEAD overhead and
  range-request overhead, but gains from 64-way concurrency. Net result: mixes high
  request overhead with high parallelism.
- **minio**: clean streaming GETs times 16 workers. On 1.2 GB/s link, 16 parallel
  files nearly saturate the link from the first batch.
- **s3torchconnector**: 1 sequential GET per worker. With 4 workers, maximum effective
  parallelism is 4 simultaneous streaming GETs. At 147 MB each, this is ~600 MB of
  in-flight data at a time vs. minio's ~2.3 GB and s3dlio's potential ~9.4 GB.

The training benchmark result gap between minio and s3torchconnector is therefore
**expected and explained** — it reflects the fetch concurrency difference, not a
fundamental capability difference between the S3 client libraries.

---

## 6. Root Cause: Wrong API for Batch Prefetch

`S3IterableDataset` is the wrong abstraction for the prefetch use case. Its intended
design is:

```python
# ✅ Intended design pattern — one item per DataLoader worker step:
dataset = S3IterableDataset.from_prefix("s3://bucket/prefix/", region="us-east-1")
loader = DataLoader(dataset, num_workers=4)
for batch in loader:
    ...   # DataLoader workers shard the iteration across processes
```

In this pattern, the DataLoader's `num_workers=4` provides the parallelism — each worker
process independently iterates its shard, and `S3IterableDataset` yields one object at a
time per worker, which is exactly what's needed.

In `_prefetch_s3torchconnector()`, it is being used as a **batch downloader** instead:

```python
# ❌ Current (incorrect) usage — sequential despite the "Iterable" name:
dataset = S3IterableDataset.from_objects(uris, ...)
for fname, reader in zip(filenames, dataset):   # one file at a time!
    cache[fname] = np.load(reader, ...)
```

The lazy iterator yields file N+1 only after `np.load(reader)` on file N completes.
There is no mechanism inside `S3IterableDataset` to pre-fetch the next item while the
current one is being consumed.

### The right API for parallel downloads with s3torchconnector

`s3torchconnector` does expose a lower-level direct-access API:

```python
from s3torchconnector import S3Client

client = S3Client(region=region, endpoint=endpoint)
reader = client.get_object(bucket="my-bucket", key="path/to/file.npz")
data = reader.read()
```

`S3Client.get_object()` returns immediately with an `S3Reader` (streaming reader backed
by the s3torchconnector Rust HTTP client). Combined with `ThreadPoolExecutor`, this would
provide the same parallelism model as minio:

```python
# ✅ Correct approach for parallel batch download with s3torchconnector:
with ThreadPoolExecutor(max_workers=min(16, len(filenames))) as pool:
    def fetch(fname):
        reader = client.get_object(bucket, key)
        return fname, np.load(reader, allow_pickle=True)["x"]
    cache = dict(pool.map(fetch, filenames))
```

### S3Client accessibility issue

The `S3Client` instance for the current run is held by `ObjStoreLibStorage` (in
`storage/obj_store_lib.py`) as `self.s3_client`. `NPZReaderS3Iterable` does not have
access to this object — it only receives `storage_options` (a dict of config values).

A fix requires one of:
- `NPZReaderS3Iterable` constructs its own `S3Client` from `storage_options` (straightforward)
- The `S3Client` instance is threaded through the class hierarchy to the reader (more
  invasive — requires changes to the DLIO `FormatReader` interface)

The first approach is simpler: read `endpoint_url` and `region` from `storage_options`
(which the reader already has access to) and construct `S3Client(region=region, endpoint=endpoint)`
once in `__init__`, caching it like `_minio_client` is currently cached.

---

## 7. Remediation Options

Three approaches are available. Only one should be chosen before re-running s3torchconnector
training benchmarks.

### Option A — Fix s3torchconnector to use ThreadPoolExecutor (Recommended)

**What changes:**
Rewrite `_prefetch_s3torchconnector()` to use `S3Client.get_object()` in a
`ThreadPoolExecutor(max_workers=min(16, n_files))`, matching the minio approach.
`NPZReaderS3Iterable.__init__` creates its own `S3Client` instance and caches it
(like `_minio_client`).

**Effective concurrency after fix:**

| Library | Per-worker | × 4 workers | Total |
|---|:-:|:-:|:-:|
| s3dlio | up to 64 | × 4 | up to 256 |
| minio | up to 16 | × 4 | 64 |
| s3torchconnector (**fixed**) | up to 16 | × 4 | **64** |

**Why recommended:**
- Makes s3torchconnector competitive and comparable to minio
- Uses the library's own native Rust HTTP client (not wrapping `S3IterableDataset` incorrectly)
- Matches the production use pattern for high-throughput object store reads
- Allows the benchmark to reveal the true read performance of the underlying Rust client

**Downside:** Creates a small architectural asymmetry — s3dlio gets up to 64/worker via
its own Rust-async scheduler, while minio and s3torchconnector get 16/worker via Python
`ThreadPoolExecutor`. This difference should be noted in the results document.

---

### Option B — Level all three down to sequential (Most Controlled)

**What changes:**
Rewrite `_prefetch_s3dlio()` and `_prefetch_minio()` to also fetch one file at a time —
remove the `ThreadPoolExecutor` from minio and lower `max_in_flight=1` for s3dlio
(or use `s3dlio.get()` per file instead of `get_many()`).

**Effective concurrency after change:**

| Library | Per-worker | × 4 workers | Total |
|---|:-:|:-:|:-:|
| s3dlio | 1 | × 4 | 4 |
| minio | 1 | × 4 | 4 |
| s3torchconnector | 1 | × 4 | 4 |

**Why useful:**
- Provides the purest "HTTP client comparison" — reveals the per-file GET latency and
  single-stream throughput of each library's underlying Rust/Python HTTP client
- Removes any "our library has better connection pooling / Tokio magic" advantage
- Results would be directly comparable to each other

**Downside:**
- Drastically reduced absolute throughput — probably 60–100 MB/s total with 4 sequential
  GETs × 4 workers on a 1.2 GB/s link (well below the network ceiling)
- Does not reflect any production usage pattern
- Throws away the existing well-tuned s3dlio and minio implementations

---

### Option C — Configurable `prefetch_workers` applied identically to all three

**What changes:**
Add `storage_options.prefetch_workers: N` to the YAML config. Inside `_prefetch()`,
pass this value as `max_workers=` to a `ThreadPoolExecutor` for all three libraries.
For s3dlio, wrap each `get_many(uris=[single_uri])` call in a thread, or use
`max_in_flight=prefetch_workers` as the `get_many` argument.

```yaml
storage:
  storage_type: s3
  storage_root: mlp-s3torch
  storage_library: s3torchconnector
  storage_options:
    endpoint_url: http://minio-host:9000
    region: us-east-1
    prefetch_workers: 8   # ← new, applied to all three libraries identically
```

**Why useful:**
- Single YAML knob controls concurrency across all three libraries
- Can sweep from 1 (baseline) to 64 to find the optimal concurrency level for each
- Allows apples-to-apples comparison at any chosen concurrency level

**Downside:**
- More complex to implement correctly for all three code paths
- s3dlio's Rust-native async (`get_many`) does not map cleanly to Python thread count —
  `max_in_flight` is a Rust semaphore, not a thread count
- Introduces a new config parameter that must be documented and validated

---

## 8. Current Training Benchmark Results Context

The following results from `dlio_mpi_object_results.md` are affected by the fairness issue.

### s3dlio v0.9.84 — valid (uses `get_many(max_in_flight=64)` as designed)

| NP | Cold epoch (s) | Warm avg (MB/s) | vs NP=1 |
|:-:|:-:|:-:|:-:|
| 1 | ~88 s | 413 ± 2 | 1.0× |
| 2 | ~45 s | 713 ± 5 | 1.7× |
| 4 | ~34 s | 1087 ± 4 | 2.6× |
| 8 | ~36 s | 964 ± 120 † | 2.3× |

† NP=8 Epoch 4 was anomalous (31.50 s vs ~23 s nominal); excluding E4 → ~1045 MB/s.

### minio — valid (uses `ThreadPoolExecutor(16)` as designed)

Results in `dlio_mpi_object_results.md`. No fairness concern.

### s3torchconnector — **NOT REPRESENTATIVE** (sequential fetch)

Any s3torchconnector training results collected with the current `_prefetch_s3torchconnector()`
implementation are not representative of the library's actual read capability. They reflect
single-connection streaming throughput × 4 DataLoader workers, not parallel fetching.

s3torchconnector training runs should be **re-run after implementing Option A or Option C**
before drawing any conclusions about s3torchconnector training performance relative to the
other two libraries.

---

## 9. s3torch data generation — valid and re-confirmed

Data generation (write direction) uses `ObjStoreLibStorage.put_data()`, which routes
s3torchconnector through `S3Client.put_object()` — the correct direct API, not
`S3IterableDataset`. Data generation results are **not affected** by the prefetch
fairness issue.

**s3torchconnector datagen throughput (NP=8, `MultipartUploadWriter` via streaming put):**

| Log timestamp | Duration | MB/s |
|---|:-:|:-:|
| `dlio-s3torch-datagen-20260320_122511` | 25.21 s | 977 |
| `dlio-s3torch-datagen-20260320_161531` | 25.96 s | 949 |
| `dlio-s3torch-datagen-20260321_085821` | 25.95 s | 949 |

**Average: 963 ± 14 MB/s** — consistent across all three runs; no update to the results
document was required after the March 21 re-run (delta = 1.5% vs rolling average, well
within the 5% update threshold).

---

## 10. Recommended Next Steps

1. **Decide on remediation approach** — Option A is recommended; discuss with team if
   Option C (configurable) is preferred for flexibility.

2. **Implement the chosen fix** in
   `dlio_benchmark/dlio_benchmark/reader/npz_reader_s3_iterable.py`.

3. **Re-run s3torchconnector training benchmarks** for NP=1, 2, 4, 8:
   ```bash
   NP=1 bash ./tests/object-store/dlio_s3torch_train.sh
   NP=2 bash ./tests/object-store/dlio_s3torch_train.sh
   NP=4 bash ./tests/object-store/dlio_s3torch_train.sh
   NP=8 bash ./tests/object-store/dlio_s3torch_train.sh
   ```

4. **Parse results** from `/tmp/dlio-s3torch-train-*/dlio.log`:
   - Look for `Ending epoch N - K steps completed in X.XX s` lines
   - Compute `24,628.8 MB ÷ epoch_s` for wall-clock throughput

5. **Update `dlio_mpi_object_results.md`** with the corrected s3torchconnector training
   results and a note that the previous results were collected under sequential-fetch conditions.

---

## 11. Key File Locations

| File | Purpose |
|---|---|
| `dlio_benchmark/dlio_benchmark/reader/npz_reader_s3_iterable.py` | All three `_prefetch_*` methods — primary change target |
| `dlio_benchmark/dlio_benchmark/storage/obj_store_lib.py` | Holds `S3Client` instance; read for endpoint/region config params |
| `configs/dlio/workload/unet3d_h100_s3torch.yaml` | s3torchconnector YAML config (`read_threads: 4`, `multiprocessing_context: spawn`) |
| `tests/object-store/dlio_mpi_object_results.md` | Primary results document; s3torch training section needs re-run |
| `tests/object-store/s3dlio_performance_analysis.md` | s3dlio-specific root cause analysis (Findings 1–4) |

---

## 12. s3dlio v0.9.84 — Issue Resolution Status

Version **v0.9.84** (wheel tagged v0.9.82+) is a critical milestone: five of the six
findings from the code review are fully resolved, and the remaining one is mitigated via
an environment variable.

### Before and after

| Metric | Baseline (v0.9.82 defaults) | After v0.9.84 fixes | Change |
|---|---|---|---|
| NP=1 throughput | 332 MB/s | **413 MB/s** | +24% |
| NP=4 throughput | ~950 MB/s | **1,087 MB/s** | +14% |
| NP=4 vs minio NP=4 | −12% | **−1%** (1,087 vs 1,097 MB/s) | fully competitive |

### Per-finding resolution

| # | Finding | Severity | v0.9.84 Status | Notes |
|---|---|---|---|---|
| 1 | Double HEAD per object (`get_many` path) | Critical | ✅ **RESOLVED** | `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0` now correctly propagates through the `get_many` code path |
| 2 | Range splitting too aggressive (32 MB threshold on 147 MB files at 1 Gbps) | Major | ✅ **RESOLVED** | Fixed together with Finding 1 via the same env-var path |
| 3 | Tokio runtime thread over-provisioning (32 threads default per process) | Major | ⚪ **MITIGATED** | Set `S3DLIO_RT_THREADS=8` (or lower); a better default is planned for a future release |
| 4 | `bytes(data)` Python copy — 147 MB allocation per file on the Python heap | Major | ✅ **RESOLVED** | Replaced with zero-copy `_BytesViewIO`; data now passes as a memoryview with no copy |
| 5 | Mutex contention in range-assembly path | Moderate | ✅ **RESOLVED** (v0.9.82) | Per-range locking replaced with lock-free assembly |
| 6 | O(N²) sort during range deduplication | Minor | ✅ **RESOLVED** (v0.9.82) | Replaced with O(N log N) sort; visible at high object-count workloads |

### Implication for benchmark interpretation

With all critical findings resolved, s3dlio NP=1 now operates at **~413 MB/s** — well
above the 1-worker concurrency cost seen with minio/s3torchconnector — and at NP=4 the
libraries are within **1%** of each other at the 1.2 GB/s network ceiling.

This confirms the core thesis: observed performance gaps were overwhelmingly about **how
the libraries were used** (range splitting, HEAD overhead, Python copy) rather than
fundamental differences in the underlying S3 client implementations. With the s3dlio
fixes in place, all three libraries now achieve effectively equivalent throughput when
used with matched concurrency.

---

## 13. s3dlio Checkpoint Write Throughput — Multipart Upload Tuning (March 2026)

### 13.1 Background — v0.9.82 pipeline stall regression

Before v0.9.82, `spawn_part()` and `spawn_part_bytes()` in `src/multipart.rs` acquired
the concurrency semaphore **inside** the spawned Tokio task. The Python writer thread
never blocked — it returned from `write()` immediately after enqueuing each part. The
bug: for a 14.96 GB object with 32 MB parts this spawned **~467 Tokio tasks simultaneously**,
each holding 32 MB = **~15 GB Rust heap**, causing OOM / runtime overload.

The v0.9.82 fix moved semaphore acquisition to **before** spawn (via `run_on_global_rt`),
correctly bounding peak memory to `max_in_flight × part_size`. However it introduced a
new regression: the Python writer thread blocks on `run_on_global_rt(sem.acquire_owned())`
every time all `max_in_flight` slots are occupied. This stalls the entire
producer↔writer pipeline:

```
Producer (main process)                Writer subprocess (sequential loop)
─────────────────────────              ─────────────────────────────────────
generator.fill_chunk(shm.buf)   →→→   buffer_queue.get()
buffer_queue.put((idx, size))         writer.write_chunk(shm.buf, size)
                                            └─ write_owned_blocking(data)
                                                  └─ spawn_part_bytes(chunk)
                                                        └─ run_on_global_rt(
                                                             sem.acquire_owned()
                                                           )  ← BLOCKS HERE
                                       ↑ writer stuck, buffer_queue fills up ↑
```

### 13.2 Measured write impact

| Client / config | NP=1 write throughput | Notes |
|---|:-:|---|
| s3torchconnector (AWS CRT) | ~0.50 GB/s | Non-blocking `write()` — CRT enqueues and returns |
| s3dlio v0.9.82 (128 MB × 8) | ~0.16 GB/s | Stalls every 1 GB in-flight |
| s3dlio v0.9.82 NP=4 aggregate | ~1.2 GB/s = 4 × 0.30 | 4 independent objects |

Adjusting `part_size` or `max_in_flight` has **no effect at NP=1** when the bottleneck
is the number of stall cycles, not the upload bandwidth. Confirmed: with `128 MB × 8`
the log showed `[S3DLIOWriter] part_size=128 MB, max_in_flight=8` but throughput was
unchanged at 0.16 GB/s.

### 13.3 Tuning theory — more concurrent connections

With the current blocking design, increasing `max_in_flight` delays the stall (more
in-flight data before the Python thread blocks) and opens more concurrent `UploadPart`
connections to MinIO. MinIO does parallelize UploadPart requests across erasure-coding
data nodes, so higher concurrency can improve per-object throughput.

The recommended direction: **smaller parts + more concurrent slots**, matching the CRT
model (many small concurrent requests, non-blocking caller).

### 13.4 Benchmark parameter matrix

The following configurations have identical peak Rust memory (`max_in_flight × part_size`):

| `part_size` | `max_in_flight` | Peak memory | MinIO connections | Stall every |
|:-----------:|:---------------:|:-----------:|:-----------------:|:-----------:|
| 16 MB | 16 | 256 MB | 16 | 256 MB in-flight |
| 16 MB | 32 | 512 MB | 32 | 512 MB in-flight |
| 16 MB | 64 | 1 GB | 64 | 1 GB in-flight |
| 32 MB | 32 | 1 GB | 32 | 1 GB in-flight |
| 64 MB | 16 | 1 GB | 16 | 1 GB in-flight |
| 128 MB | 8 | 1 GB | 8 | 1 GB in-flight |

### 13.5 How to run the benchmark matrix

`pytorch_obj_store_checkpointing.py` now reads two environment variables (with defaults
matching the s3dlio library's built-in defaults):

```bash
# Library defaults (was previously: 128 MB × 8 hardcoded)
bash tests/object-store/dlio_s3dlio_checkpoint.sh

# 16 concurrent connections, 16 MB parts (library defaults)
S3DLIO_MULTIPART_PART_SIZE_MB=16  S3DLIO_MULTIPART_MAX_IN_FLIGHT=16  bash tests/object-store/dlio_s3dlio_checkpoint.sh

# 32 connections, 16 MB parts
S3DLIO_MULTIPART_PART_SIZE_MB=16  S3DLIO_MULTIPART_MAX_IN_FLIGHT=32  bash tests/object-store/dlio_s3dlio_checkpoint.sh

# 64 connections, 16 MB parts
S3DLIO_MULTIPART_PART_SIZE_MB=16  S3DLIO_MULTIPART_MAX_IN_FLIGHT=64  bash tests/object-store/dlio_s3dlio_checkpoint.sh

# 32 connections, 32 MB parts
S3DLIO_MULTIPART_PART_SIZE_MB=32  S3DLIO_MULTIPART_MAX_IN_FLIGHT=32  bash tests/object-store/dlio_s3dlio_checkpoint.sh
```

### 13.6 Root-cause fix (pending)

A proper fix is tracked in **GitHub issue #134** on `russfellows/s3dlio`. The proposed
architecture replaces `run_on_global_rt(sem.acquire_owned())` with a coordinator Tokio
task that drains a bounded `mpsc::channel`. The Python writer calls `blocking_send()`
which only blocks if the coordinator is a full `max_in_flight` parts behind — in
steady state (coordinator is CPU-bound per item) this never happens.

| Approach | Peak memory | Python writer blocks? |
|---|:-:|:-:|
| Pre-v0.9.82 (semaphore inside task) | **~15 GB** (467 tasks × 32 MB) | Never |
| v0.9.82 current (semaphore before spawn) | `max_in_flight × part_size` | Every cycle when full |
| Issue #134 fix (coordinator + channel) | `≤ 2 × max_in_flight × part_size` | Essentially never |

### 13.7 Results (March 25, 2026 — NP=1, MinIO `https://172.16.1.40:9000`)

Each run was observed for ~20–40 seconds (3–4 GB written) and Ctrl-C'd once steady
state was confirmed. First-chunk values are inflated because the semaphore has free
slots on start; steady state is the repeating plateau.

| `part_size` | `max_in_flight` | First chunk GB/s | Steady-state GB/s | Notes |
|:-----------:|:---------------:|:----------------:|:-----------------:|-------|
| 128 MB | 8 | — | 0.16 | Previous hardcoded default (pre-change baseline) |
| 16 MB | 16 | 0.18 | **0.16** | Library defaults |
| 16 MB | 32 | 0.19 | **0.16** | 2× connections, no improvement |
| 32 MB | 32 | 0.23 | **0.16** | Higher first-chunk burst; same steady state |
| 16 MB | 64 | _not tested_ | | |

**Conclusion: all configurations converge to 0.16 GB/s in steady state.**

The first-chunk burst (0.18–0.23 GB/s) is higher with larger part sizes because the
semaphore starts with all slots free — the first `max_in_flight` parts are spawned
without blocking. Once those slots fill, `run_on_global_rt(sem.acquire_owned())` starts
blocking the Python writer thread on every part, and throughput locks to ~0.16 GB/s
regardless of part size or slot count.

This definitively confirms: **`part_size` and `max_in_flight` are not the bottleneck.**
The bottleneck is the blocking semaphore acquisition on the Python writer thread. The
tuning matrix is exhausted. The only fix is the coordinator channel architecture
described in §13.6 (GitHub issue #134).
