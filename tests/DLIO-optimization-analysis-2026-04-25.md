# DLIO Benchmark Optimization Analysis

**Date**: April 25, 2026  
**Author**: Copilot research session  
**Scope**: `dlio_benchmark/dlio_benchmark/` and `mlp-storage/mlpstorage_py/`

---

## Overview

This document records three code changes made to `dlio_benchmark` to improve S3 datagen
throughput and checkpoint save throughput, plus a fourth change to fix a Rust/Tokio
thread-safety issue in the streaming checkpoint producer-consumer pipeline.  It also
documents the optimal usage pattern for `dgen-py` and explains why zero-copy must be
maintained end-to-end.

---

## Problem 1: Datagen Was Not Uploading Concurrently

### Root Cause

The original `_generate_files()` in `data_generator/data_generator.py` used a
`ThreadPoolExecutor` where **each worker thread both generated AND uploaded one file**.
With `write_threads` auto-sized at `min(per_rank_cpus, cap)`:

- 28 CPUs, NP=8 → 3.5 CPUs/rank → **3 threads/rank** (the old formula)
- 3 threads × 1 upload/thread = **3 concurrent uploads/rank**

Because `np.savez` generates the data (~8 ms for 140 MiB with dgen-py) much faster than
the upload takes (~280 ms at 500 MB/s), each thread spent most of its time waiting for
the network — and only 3 uploads were ever in flight at once.

### Fix: True Async Pipeline (commit: data_generator.py)

For object-store paths, the generation and upload are now separated into a pipeline:

```
Main thread:  [gen file 1] → [gen file 2] → [gen file 3] → … (14 ms each, fast)
Upload pool:      [upload 1]   [upload 2]   [upload 3]   … (280 ms each, slow)
              ↑ pipeline: main thread always 1 step ahead
```

Implementation:

```python
# Main thread generates into BytesIO (fast — dgen-py Rust, ~14 ms)
write_fn(i, ..., out_path_spec, False, output)

# Block only if n_workers uploads are already in flight (back-pressure)
_sem.acquire()

# Submit upload immediately; main thread continues with next file
_futures.append(pool.submit(_upload, out_path_spec, output, _sem))
```

A `threading.Semaphore(n_workers)` provides back-pressure: if all `n_workers` upload
slots are busy, the main thread blocks until one finishes, bounding peak memory to
`n_workers × file_size`.

### Fix: CPU-Proportional write_threads Scaling (commit: config.py)

The auto-sizing formula for S3 changed from `min(per_rank_cpus, cap)` to
`max(4, min(per_rank_cpus * 2, cap))`:

| System | NP | CPUs/rank | Old threads | New threads |
|---|---|---|---|---|
| 28-core machine | 8 | 3.5 | 3 | **7** |
| 28-core machine | 1 | 28 | 8 | **16** |
| 16-core machine | 4 | 4 | 4 | **8** |
| 256-core machine | 8 | 32 | 8 | **16** |
| 256-core machine | 1 | 256 | 8 | **16** |

Rationale:

- S3 uploads are **I/O-bound** — threads release the GIL during network I/O, so thread
  count ≠ CPU core count constraint.
- `× 2` multiplier: standard heuristic for I/O-bound work (twice as many threads as
  cores because half are blocked at any given time).
- Cap (`DLIO_MAX_AUTO_THREADS`, default 16): prevents hundreds of threads on very large
  machines where the S3 server would be the bottleneck anyway.
- Minimum 4: ensures meaningful concurrency even on tiny VMs.
- **Local FS path unchanged**: disk writes are CPU+I/O mixed; `min(per_rank_cpus, cap)`
  remains the correct formula.

---

## Problem 2: Checkpoint Save Was ~4.5× Slower Than Load

### Observed Numbers (s3-ultra, NP=8, 2 cycles)

| Operation | Average time | Throughput |
|---|---|---|
| Checkpoint **save** | 54.2 s | **1.93 GiB/s** |
| Checkpoint **load** | 11.9 s | **8.81 GiB/s** |
| Gap | | **4.56×** |

### Wrong Hypothesis (Discarded)

Initial analysis incorrectly attributed the gap to data-generation asymmetry (checkpoint
save calling dgen-py while load reads real data).  Both paths actually use dgen-py for
data generation at equivalent speeds (~55 GB/s streaming), so this cannot explain a
4.56× difference.

### Correct Root Cause: Concurrent Request Overload on the Server-Side Event Loop

**Important note**: s3dlio 0.9.92 uses **HTTP/1.1 by default** (not HTTP/2).
`DEFAULT_H2C_ENABLED = false` in `s3dlio/src/constants.rs` — h2c was disabled
as the default in v0.9.92 after benchmarking showed HTTP/2 reduces throughput on
plain `http://` endpoints compared with HTTP/1.1 and an unlimited connection pool.
The `S3DLIO_H2C` variable is NOT set in `.env`, confirming HTTP/1.1 is in use.

The real cause is **too many concurrent TCP connections / requests** being driven into
s3-ultra's Tokio runtime during saves compared to loads:

| Path | Formula | With NP=8 | Total concurrent requests |
|---|---|---|---|
| **Load** `num_parallel_readers` | `max(2, 8 // world_size)` | 2/rank | **16 concurrent GETs** |
| **Save** `max_in_flight` | env default `"8"` | 8/rank | **64 concurrent UploadPart POSTs** |

s3-ultra's Tokio event loop was handling **4× more concurrent requests** during saves
than during loads. With HTTP/1.1, each request is a separate TCP connection, so 64
concurrent UploadPart connections saturate the server's Tokio connection-accept queue
and TCP receive buffers, while 16 GET connections do not. This matches the 4.56× gap.

Additionally, each UploadPart (128 MiB) is a large inbound body that Tokio must buffer
and acknowledge on the server side — inbound large-body handling is heavier than outbound
range-GET streaming, so the 4× connection imbalance translates to more than 4× server
CPU overhead during saves.

### Fix: Match Save Concurrency to Load Concurrency (commit: pytorch_obj_store_checkpointing.py)

Target: **16 total UploadPart streams** across all ranks — symmetric with the 16
range-GETs used by load.

```python
_TARGET_TOTAL_INFLIGHT = 16

# Per-rank in-flight = ceil(16 / world_size)
_default_inflight = max(2, (_TARGET_TOTAL_INFLIGHT + _mpi_world_size - 1) // _mpi_world_size)
```

| NP | Per-rank in-flight | Total streams | Load streams |
|---|---|---|---|
| 1 | 16 | 16 | 16 |
| 4 | 4 | 16 | 16 |
| 8 | **2** | **16** | **16** |

Override via `S3DLIO_MULTIPART_MAX_IN_FLIGHT` environment variable.

### Fix: num_buffers Sized to Prevent Producer Stalls

The shared-memory buffer pool in `StreamingCheckpointing` must be deep enough that the
dgen-py producer **never blocks** waiting for a free buffer while all `max_in_flight`
uploads are in progress.

Required depth:

```
num_buffers = max_in_flight × (part_size / chunk_size)
```

Example (NP=8, s3dlio, 128 MiB parts, 32 MiB chunks):

```
num_buffers = 2 × (128 / 32) = 8   (was 4)
peak RAM = 8 × 32 MiB = 256 MiB per rank
```

Without this fix, the producer would stall after filling 4 buffers (the old pool size)
even though only 2 parts were being uploaded — limiting effective pipeline depth.

---

## Problem 3: `fork` Breaks Rust/Tokio Worker Threads

### Why `fork` Is Dangerous With Rust Libraries

Both `s3dlio` and `dgen-py` are Rust extensions using PyO3.  They rely on:

- **s3dlio**: Tokio async runtime with a dedicated thread pool
- **dgen-py**: Rayon parallel computation thread pool

When Python calls `os.fork()` (via `multiprocessing.get_context('fork')`):

1. The child process gets an **identical copy of parent memory**.
2. Only the thread that called `fork()` continues in the child; all other threads
   **cease to exist immediately**.
3. Any OS mutex, `AtomicBool`, or condvar that was held by a killed thread in the parent
   is **permanently locked** in the child — causing guaranteed deadlocks on first use.
4. Rust's Tokio runtime (in s3dlio) uses a global `OnceCell<Runtime>`.  After fork, the
   `OnceCell` appears "already initialized" in the child but points to a **dead runtime**
   with no live threads.  The first Tokio `.await` hangs forever.

### Location of the Bug

`mlp-storage/mlpstorage_py/checkpointing/streaming_checkpoint.py`, line 182:

```python
# BEFORE (WRONG — kills Tokio/Rayon threads):
try:
    ctx = mp.get_context('fork')
except ValueError:
    ctx = mp.get_context()   # spawn fallback only on non-Linux
```

The comment said "Uses 'fork' to inherit environment variables (AWS credentials, etc.)"
— but this is **incorrect**.  Python's `spawn` context also passes `os.environ` to the
child at creation time (it serializes the parent's environment variables into the child
invocation).  There is no advantage to `fork` here.

### Fix: Always Use `spawn` (commit: streaming_checkpoint.py)

```python
# AFTER (CORRECT — child gets a clean Python interpreter):
ctx = mp.get_context('spawn')
```

With `spawn`:

- A **fresh Python interpreter** is started in the child.
- Rust libraries (`s3dlio`, `dgen-py`) are imported fresh in the child — Tokio and Rayon
  create new thread pools cleanly.
- All `os.environ` variables (AWS credentials, endpoint URL, etc.) are inherited from
  the parent process at startup — the original justification for `fork` does not apply.
- `shared_memory.SharedMemory` names, `mp.Queue`, and `mp.Event` all work correctly with
  `spawn` (they use OS-level IPC, not forked file descriptors).
- The `_writer_process` is a `@staticmethod` and receives all state through its
  arguments — no closure over parent-process objects that would break with spawn.

**Note on startup latency**: `spawn` takes ~100–500 ms longer than `fork` to launch the
child process (fresh interpreter import).  For checkpoint cycles that take tens to
hundreds of seconds, this overhead is negligible.

---

## dgen-py: Optimal Usage Patterns

The `dgen-py` library (Rust, PyO3 + Rayon) has two distinct usage modes with very
different performance characteristics.  **Using the wrong mode can result in 3–4× lower
throughput.**

### Mode 1 — Streaming (preferred for large, sequential data)

```python
gen = dgen_py.Generator(
    size=total_bytes,        # Total data to generate
    chunk_size=32*1024*1024, # 32 MB per fill_chunk() call (default)
    max_threads=max_threads, # Throttle under MPI
)
buffer = bytearray(gen.chunk_size)   # Pre-allocate ONCE
while not gen.is_complete():
    nbytes = gen.fill_chunk(buffer)  # Fill in-place — ZERO COPY to buffer
```

| Characteristic | Value |
|---|---|
| Thread pool | Created **once**, reused for every `fill_chunk()` call |
| Throughput | 52–63 GB/s on 12-core VM |
| Memory | Constant 32 MB (chunk size) |
| Use case | Checkpoints, large sequential blobs |

**This is the pattern used in `streaming_checkpoint.py`** — `generator.fill_chunk(shm.buf)`
writes directly into shared memory (zero-copy).

### Mode 2 — Per-Object (for seeded, independent files)

```python
gen = dgen_py.Generator(size=file_bytes, seed=per_file_seed)
bytesview = gen.get_chunk(file_bytes)  # Returns BytesView (Rust-owned, immutable)
arr = np.frombuffer(bytesview, dtype=dtype).reshape(shape)  # ZERO COPY
# bytesview stays alive (referenced by arr) until arr goes out of scope
```

| Characteristic | Value |
|---|---|
| Thread pool | Created **per Generator** — per file for independent seeds |
| Throughput | 17–20 GB/s (100 MB–10 GB objects) |
| Memory | Rust-owned buffer via `BytesView` (released when `gen` is GC'd) |
| Use case | NPZ training files where each file needs a reproducible per-file seed |

The 3–4× lower throughput vs streaming is **acceptable for the NPZ datagen case** because:
- Generation (~8 ms for 140 MiB) is ≪ upload time (~280 ms at 500 MB/s)
- The async pipeline overlaps generation with uploads, so generation is never on the
  critical path
- Per-file seeding cannot be achieved with the streaming API without a `reseed()` method

### Zero-Copy Chain for NPZ Files

The complete data path, showing where copies occur:

```
dgen_py.Generator.get_chunk(N)         → BytesView (Rust allocation, zero-copy)
  ↓ np.frombuffer(bytesview, dtype)    → numpy array (zero-copy view, read-only)
  ↓ np.savez(output_bytesio, x=arr)    → NPZ serialization (ONE UNAVOIDABLE COPY)
  ↓ BytesIO.getbuffer()                → memoryview of internal BytesIO buffer (zero-copy)
  ↓ s3dlio.put_bytes / MultipartWrite  → sends memoryview bytes to S3 (zero-copy)
```

**Total copies: 1** (the NPZ format serialization — unavoidable).

Key requirement: pass `BytesIO` directly to `put_data()` (not `BytesIO.getvalue()`).
`getvalue()` makes a full copy of the internal buffer; `getbuffer()` returns a zero-copy
memoryview.  The `put_data()` implementation checks for `getbuffer` first:

```python
if hasattr(data, 'getbuffer'):
    payload = data.getbuffer()   # zero-copy memoryview ← CORRECT PATH
elif hasattr(data, 'getvalue'):
    payload = data.getvalue()    # extra copy ← avoid this path
```

### What NOT To Do

```python
# ❌ Creates new Rayon thread pool for every file — 3-4× slower than streaming
#    (acceptable for NPZ files, but avoid in tight loops for small objects)
for file in files:
    gen = dgen_py.Generator(size=file_size)
    data = gen.get_chunk(file_size)

# ❌ Extra copy — bypasses zero-copy path in put_data()
storage.put_data(path, output.getvalue())  # getvalue() makes a copy!

# ✓ Correct — zero-copy path
storage.put_data(path, output)   # BytesIO.getbuffer() is called inside put_data

# ❌ NumPy random generation for large files — single-threaded, plateaus at ~2.5 GB/s
arr = np.random.default_rng().random(size=shape)  # 2.5 GB/s max

# ✓ dgen-py for large files — parallel Rayon, 17-20 GB/s per-object
gen = dgen_py.Generator(size=total_bytes, seed=seed)
bytesview = gen.get_chunk(total_bytes)   # 17-20 GB/s (vs NumPy's 2.5 GB/s)
arr = np.frombuffer(bytesview, dtype=dtype).reshape(shape)  # zero-copy
```

---

## Summary of All Changes

| File | Change | Reason |
|---|---|---|
| `dlio_benchmark/utils/config.py` | S3 write_threads: `max(4, min(per_rank_cpu * 2, cap))` | Scale with CPUs; 2× for I/O-bound; cap at 16 |
| `dlio_benchmark/data_generator/data_generator.py` | True async pipeline for object stores | Decouple fast generation from slow upload |
| `dlio_benchmark/data_generator/data_generator.py` | `_write_one`: pass `output` (BytesIO) not `output.getvalue()` | Zero-copy through `getbuffer()` in put_data |
| `dlio_benchmark/checkpointing/pytorch_obj_store_checkpointing.py` | `max_in_flight = max(2, ceil(16/world_size))` | Match load's 16 total GET streams |
| `dlio_benchmark/checkpointing/pytorch_obj_store_checkpointing.py` | `num_buffers = max_in_flight × chunks_per_part` | Prevent producer stalls |
| `mlp-storage/mlpstorage_py/checkpointing/streaming_checkpoint.py` | `mp.get_context('fork')` → `mp.get_context('spawn')` | Fork kills Tokio/Rayon threads |

---

## Expected Performance Impact

| Metric | Before | Expected After |
|---|---|---|
| Datagen concurrent uploads (NP=8, 28-core) | 3/rank | **7/rank** |
| Checkpoint save throughput (NP=8) | 1.93 GiB/s | **~8–9 GiB/s** |
| Checkpoint load throughput (NP=8) | 8.81 GiB/s | unchanged |
| Checkpoint save/load symmetry | 4.56× gap | **~1× (symmetric)** |
| Fork-related deadlock risk | Present | **Eliminated** |
