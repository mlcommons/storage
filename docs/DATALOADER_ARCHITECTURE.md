# Data Loader Architecture: Map-Style vs. Iterable-Style

**Status**: Implemented in `dlio_benchmark/dlio_benchmark/data_loader/torch_data_loader.py`.
**Relevant workloads**: UNet3D (NPZ), RetinaNet (JPEG), and any NPY/PNG workload on S3 or POSIX storage.

---

## Table of Contents

1. [Background: The Conventional Wisdom](#background-the-conventional-wisdom)
2. [What Actually Matters for Object Storage](#what-actually-matters-for-object-storage)
3. [Implementation: TorchIterableDatasetSimple](#implementation-torchiterabledatasetsimple)
4. [Known Limitations and Future Work](#known-limitations-and-future-work)
5. [Summary](#summary)
6. [Related Documents](#related-documents)
7. [O_DIRECT Local Storage: Two Independent Paths](#o_direct-local-storage-two-independent-paths)
   - [Why O_DIRECT Matters for NVMe Benchmarks](#why-o_direct-matters-for-nvme-benchmarks)
   - [Path 1: `odirect: true` — Python O_DIRECT (legacy map-style)](#path-1-odirect-true--python-o_direct-legacy-map-style)
   - [Path 2: `storage_library: direct` — Rust/Tokio O_DIRECT (new async path)](#path-2-storage_library-direct--rusttokio-o_direct-new-async-path)
   - [Comparison](#comparison)
   - [Which Path to Use](#which-path-to-use)

---

## Background: The Conventional Wisdom

A common recommendation is that **iterable-style data loaders are better for large datasets**.
This advice is correct in its original context — local filesystem reads on spinning disk — but the
reasoning does *not* transfer directly to object storage. Understanding *why* iterable can be better
(and when it is not) is critical for choosing the right approach.

The original case for iterable:

- **Map-style requires a full index upfront** — you must know `len(dataset)` to build a sampler.
- **Map-style with shuffled indices causes random seeks** — on HDDs, jumping around the dataset
  produces catastrophically bad throughput.
- **Iterable-style reads sequentially** — the iterator delivers samples in whatever order it
  generates them, which aligns naturally with sequential disk I/O.

For object storage, neither of these concerns applies. There is no seek penalty — an S3 GET for
object #7,199 costs the same as a GET for object #0. The raw "iterable is better" rule does not
carry over.

---

## What Actually Matters for Object Storage

The real performance argument for iterable-style on object storage is about
**concurrency pipeline depth**, not seek patterns.

### Previous path — map-style TorchDataset, 4 workers (replaced)

```
Worker 0: __getitem__(idx_0) → read_index() → _s3_ensure_cached() → get_many([1 object])
Worker 1: __getitem__(idx_1) → read_index() → _s3_ensure_cached() → get_many([1 object])
Worker 2: __getitem__(idx_2) → read_index() → _s3_ensure_cached() → get_many([1 object])
Worker 3: __getitem__(idx_3) → read_index() → _s3_ensure_cached() → get_many([1 object])
```

Total in-flight S3 requests: **4** (one per DataLoader worker). Map-style is still used for
format types that do not have iterator-based readers (e.g. SYNTHETIC, HDF5 without S3 backend).

### Current path — TorchIterableDatasetSimple, 4 workers (implemented)

```
Worker 0: next() → _s3_prefetch_all() → get_many([~1800 objects, max_in_flight=64])
Worker 1: next() → _s3_prefetch_all() → get_many([~1800 objects, max_in_flight=64])
Worker 2: next() → _s3_prefetch_all() → get_many([~1800 objects, max_in_flight=64])
Worker 3: next() → _s3_prefetch_all() → get_many([~1800 objects, max_in_flight=64])
```

For local / POSIX storage, `_localfs_prefetch_all()` is used instead:
```
Worker k: next() → _localfs_prefetch_all() → ThreadPoolExecutor(64 threads) → pread(1 file each)
```

Total in-flight: up to **64 objects per worker × 4 workers = 256 concurrent S3 GETs**
(or 256 concurrent `pread` calls for local FS).

While the compute side is processing one object, up to 63 more are already being fetched for that
worker alone. This keeps the network link and storage server fully utilized even when individual
GETs have variable latency.

---

## Implementation: TorchIterableDatasetSimple

The fix is `TorchIterableDatasetSimple` in `torch_data_loader.py`, which activates for all
`_simple_iterable_formats = (NPZ, NPY, JPEG, PNG)` on both S3 and local FS.

Key mechanics:

1. **File sharding** — `__iter__` computes `my_files = all_files[worker_id::num_workers]`,
   giving each PyTorch worker a distinct non-overlapping file subset.

2. **file_map installation** — the shard is installed as
   `reader.file_map[thread_index] = [(global_idx, filename, sample_in_file), ...]`
   so that `reader.next()` (which reads `file_map[thread_index]`) picks it up.

3. **Bulk prefetch** — `reader.next()` calls `_s3_prefetch_all()` (S3) or
   `_localfs_prefetch_all()` (local FS) before starting iteration. All files for this
   worker's shard are fetched in parallel (up to 64 in-flight) before any sample is yielded.

4. **Yield** — one dummy item is yielded per complete batch, consistent with the Parquet
   `TorchIterableDataset` pattern. `batch_size=None` in the DataLoader passes items through
   unchanged. FormatReader.next() handles drop-last internally.

The DLIO log now prints `TorchIterableDatasetSimple(bulk-prefetch, N workers)` for these
formats instead of `TorchDataset(map-style, N workers)`.

---

## Known Limitations and Future Work

### 1. Per-epoch file shuffle in workers

PyTorch DataLoader workers are spawned with a pickled snapshot of `ConfigArguments`.
When the main process calls `reconfigure(epoch+1)`, the shuffled `file_list_train` is
not propagated to persistent workers. Each worker's `_file_list` reflects the epoch-1
ordering for all subsequent epochs.

For a **storage I/O benchmark**, this is acceptable: throughput and latency measurements
are not affected by file ordering on object storage (no HDD seek penalty). File order
does not affect whether all files are read.

For **ML training correctness**, per-epoch reshuffling matters. A future improvement:
pass an epoch seed into `TorchIterableDatasetSimple` and shuffle `all_files` with
`np.random.default_rng(seed + epoch)` inside `__iter__`.

### 2. Prefetch memory for small objects

`_s3_prefetch_all()` issues GETs for all objects in a worker's slice (up to ~1,800 for NP=4)
with 64 in-flight. The cache stores `{key: byte_count}` only — actual bytes are consumed
by s3dlio's callback immediately after transfer. Memory footprint is bounded by the
in-flight window size (64 × object_size), not the full epoch size.

For UNet3D (140 MiB objects): 64 × 140 MiB ≈ 9 GiB peak per worker.
For RetinaNet (315 KB objects): 64 × 315 KB ≈ 20 MiB peak per worker — negligible.

### 3. Drop-last behavior

`FormatReader.next()` drops the final partial batch if `len(shard) % batch_size != 0`.
This matches the map-style `drop_last=True` behavior. No action needed.

---

## Summary

| Property | Map-style (old) | TorchIterableDatasetSimple (current) |
|---|---|---|
| Formats | All | NPZ, NPY, JPEG, PNG |
| Storage backends | All | S3 (s3dlio/minio/s3torch) **and** POSIX/local FS |
| In-flight S3 requests | `1 × num_workers` | `64 × num_workers` |
| In-flight local reads | `1 × num_workers` | `64 × num_workers` (ThreadPool) |
| Per-object bandwidth | Good (s3dlio byte-range) | Same |
| Worker file partitioning | Automatic via Sampler | `all_files[worker_id::num_workers]` |
| Per-epoch file shuffle | Via VirtualIndexMap | `_file_list` as-is (epoch 1 order) |
| Implementation status | Retired for NPZ/NPY/JPEG/PNG | **Active** |

The most important validation step: a side-by-side benchmark sweep (UNet3D and RetinaNet,
identical NP/config) measuring `train_throughput_MB_per_second` with the new vs. old path.
Expected improvement is largest for small objects (RetinaNet 315 KB: no byte-range splitting,
pipeline depth was 1 per worker, now 64 per worker).

---

## Related Documents

- [UNet3D_NP_Scaling_Results.md](UNet3D_NP_Scaling_Results.md) — benchmark results where this
  architectural choice is most relevant
- [ARCHITECTURE.md](ARCHITECTURE.md) — overall system architecture
- [STORAGE_LIBRARIES.md](STORAGE_LIBRARIES.md) — s3dlio capabilities (get_many, byte-range GETs,
  ObjectSizeCache)
- [PARQUET_FORMATS.md](PARQUET_FORMATS.md) — the Parquet iterable reader that already uses the
  `TorchIterableDataset` path

---

# O_DIRECT Local Storage: Two Independent Paths

DLIO has **two separate mechanisms** for bypassing the Linux page cache when reading local
(POSIX/NVMe) files. Both are preserved and intentionally kept distinct so they can be compared
against each other directly.

---

## Why O_DIRECT Matters for NVMe Benchmarks

The Linux page cache caches file data in DRAM. After the first read pass, subsequent reads of the
same files are served entirely from memory, not from the storage device. For an I/O benchmark
intended to stress NVMe drives this is fatal: repeated runs measure DRAM bandwidth (40–60 GB/s
on a modern server) rather than NVMe device bandwidth (3–15 GB/s per drive). The numbers are
plausible-looking but completely wrong.

`O_DIRECT` opens files with the `O_DIRECT` flag, which instructs the kernel to transfer data
directly between the storage device and a userspace buffer, bypassing the page cache entirely.
Cold-run and warm-run throughput become essentially identical, accurately reflecting the hardware.
The tradeoff: userspace buffers must be 4 KiB-aligned and reads must be a multiple of the block
size (512 B or 4096 B depending on the device).

---

## Path 1: `odirect: true` — Python O_DIRECT (legacy map-style)

Activated by setting the top-level `odirect: true` flag in the DLIO YAML config:

```yaml
odirect: true
```

**Implementation**: `reader_factory.py` detects `odirect == True` and routes to
`NPZReaderODIRECT` / `NPYReaderODirect` instead of the default readers.

**How it works** (`npy_reader_odirect.py`, `npz_reader_odirect.py`):

1. `os.open(filepath, os.O_RDONLY | os.O_DIRECT)` — opens the file with O_DIRECT in Python.
2. A 4 KiB-aligned buffer is manually allocated with `ctypes` + `bytearray` arithmetic.
3. `os.readv(fd, [mem_view])` — single synchronous read into the aligned buffer.
4. `parse_npy()` / `parse_npz()` — full NPY/NPZ format decode in Python: `struct.unpack` header
   parsing, optional `zlib.decompress()` (NPZ), and `np.ndarray()` construction from the
   in-memory buffer (zero-copy array view).

**Concurrency model**: map-style `__getitem__` path. Each PyTorch DataLoader worker calls
`odirect_read()` once per sample index, synchronously. There is no prefetch, no concurrency
within a worker, and no inter-worker coordination. Concurrency is provided only by the number of
DataLoader workers (`num_workers` in `torch.utils.data.DataLoader`).

**PyTorch involvement**: PyTorch provides the outer loop (the DataLoader process pool and
`__getitem__` dispatch). PyTorch does **not** issue any I/O itself — all reads are done by the
Python `os.open` + `os.readv` path above. The term "PyTorch O_DIRECT" would be misleading;
this is purely Python-level O_DIRECT wired into the PyTorch DataLoader's index-based interface.

---

## Path 2: `storage_library: direct` — Rust/Tokio O_DIRECT (new async path)

Activated by setting `storage_library: direct` inside `storage_options` in the DLIO YAML config:

```yaml
storage:
  storage_type: local_fs
  storage_root: /mnt/nvme/dataset
  storage_options:
    storage_library: direct   # activates Rust async O_DIRECT
```

**Implementation**: `_LocalFSIterableMixin._localfs_init()` reads `storage_options.storage_library`.
When set to `"direct"`, it sets `self._use_direct = True` and validates that `s3dlio` is
importable. `_localfs_prefetch_all()` then dispatches to `_prefetch_direct()` instead of
`_prefetch_buffered()`.

**How it works** (`_local_fs_iterable_mixin.py`):

1. Converts each local path to a `direct://` URI: `f"direct://{os.path.abspath(path)}"`.
2. Calls `s3dlio.get_many(uris, max_in_flight=min(64, len(uris)))`.
3. s3dlio's Rust backend (`file_store_direct.rs`) opens each file with `libc::O_DIRECT`,
   allocates 4 KiB-aligned buffers in Rust, and reads via Tokio async I/O. The GIL is fully
   released for all I/O.
4. `_prefetch_direct()` collects byte counts from `BytesView` objects (O(1), no Python copy).
5. Byte counts are accumulated into `_total_bytes_read` / `_total_objects_read` for
   `finalize_local_bytes()` reporting.

**Concurrency model**: iterable-style `TorchIterableDatasetSimple` path. Each worker calls
`_localfs_prefetch_all()` once per shard, submitting up to 64 O_DIRECT reads concurrently into
the Tokio runtime. Results are streamed back as they complete (not in submission order).
Total concurrency: `64 × num_workers` simultaneous O_DIRECT reads.

---

## Comparison

| Property | `odirect: true` (Path 1) | `storage_library: direct` (Path 2) |
|---|---|---|
| Config key | `odirect: true` (top-level) | `storage_options.storage_library: direct` |
| I/O syscall | `os.open + os.readv` (Python) | `libc::open + O_DIRECT` (Rust, Tokio) |
| Alignment | Python `ctypes` manual alignment | Rust automatic 4 KiB alignment |
| GIL behavior | Held during `os.readv` | Released for all I/O |
| Prefetch depth | 1 per DataLoader worker | 64 per DataLoader worker |
| DataLoader style | Map-style (`__getitem__`) | Iterable-style (`__iter__`) |
| Concurrency | `1 × num_workers` | `64 × num_workers` |
| NPY/NPZ decode | Full in-Python decode per file | None (byte count only, decode deferred) |
| Page cache bypass | Yes (`O_DIRECT`) | Yes (`O_DIRECT` via `direct://` URI) |
| s3dlio dependency | No | Yes (must be installed) |
| Formats | NPZ, NPY | NPZ, NPY, JPEG, PNG |
| Status | Preserved (comparison baseline) | Implemented (high-concurrency path) |

---

## Which Path to Use

Both paths are intentionally preserved. Neither removes the other.

- **Use `odirect: true`** as a baseline. It provides the simplest possible O_DIRECT
  implementation: one synchronous Python read per file per worker. If this path achieves the
  same throughput as Path 2, it means the bottleneck is not I/O concurrency (perhaps it is
  CPU-side decode or tensor construction).

- **Use `storage_library: direct`** when you want maximum I/O concurrency on NVMe. The Rust
  async path with 64 in-flight reads per worker is the correct model for high-queue-depth NVMe
  drives, which perform best when saturated with many parallel requests (QD=32–128 is typical
  for NVMe SSDs). Python map-style with 1 read per worker cannot saturate a modern NVMe device
  regardless of the number of DataLoader workers.

- **Comparing the two** directly — identical config except swapping `odirect: true` vs.
  `storage_library: direct` — isolates the contribution of:
  1. I/O concurrency depth (1 vs. 64 per worker)
  2. GIL contention (held during Python `os.readv` vs. fully released in Rust)
  3. Prefetch pipelining (none vs. up to 64 in-flight while compute processes the previous batch)

This comparison is one of the primary intended use cases for keeping both paths available.
