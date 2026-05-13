# NPZ Datagen Optimization Analysis

**Date:** 2026-04-25  
**Goal:** Reach 8 GB/s aggregate throughput for unet3d NPZ datagen with NP=8

---

## 1. Current Measured Performance

| Run | Model | Storage Lib | Runtime | Throughput |
|-----|-------|-------------|---------|------------|
| 2026-04-25T12:16 | unet3d | s3dlio | 21.2 s | ~1.11 GB/s |
| 2026-04-25T12:17 | unet3d | minio  | 24.7 s | ~0.95 GB/s |

- 168 files × 8 MPI ranks = 21 files/rank
- Each file: 139.8 MiB (shape `(6053, 6053, 1)` float32)
- s3-ultra listening on `0.0.0.0:9101`

---

## 2. Object and Array Size Derivation

Config: `record_length_bytes=146600628`, `record_length_bytes_stdev=68341808`, dtype=float32

```
record_length (elements) = 146600628 / 4 = 36650157
dimension = floor(sqrt(36650157)) = 6053
Array shape: (6053, 6053, 1) float32
Array size: 6053 × 6053 × 1 × 4 = 146,572,036 bytes = 139.8 MiB
NPZ size (STORED, no compression): ≈ 139.9 MiB (header overhead ~100 bytes)
```

---

## 3. Critical Finding: Installed dlio_benchmark is STALE

**mlp-storage uses a wheel installed from git, NOT our local modified source.**

Evidence:
```
source file:    /home/eval/Documents/Code/dlio_benchmark/dlio_benchmark/utils/utility.py  (24879 bytes)
installed file: ...site-packages/dlio_benchmark/utils/utility.py                          (19154 bytes)
```

The installed version is missing:
- Singleton `_DGEN_PROC_GEN` pattern (avoids re-creating Rayon thread pool per file)
- Async pipeline in `data_generator.py` (upload pool running while main thread generates)
- `write_threads` floor=8 cap=32 in `config.py`
- Raw-bytes dgen path in `gen_random_tensor()`

**Impact:** Without the async pipeline, each file is: serialize (270ms) + upload (sequential, ~1s) = ~1.3s/file × 21 files = ~27s ≈ matches measured 21s.

With the async pipeline correctly installed, expected: 21 files × 280ms generation = 5.9s dominated by serial generation, but uploads overlapped → should be much faster.

---

## 4. Per-File Timing Breakdown

### np.savez baseline (actual unet3d shape)

```
Shape: (6053, 6053, 1) float32 = 139.8 MiB
  Run 0: 270 ms, 518 MB/s
  Run 1: 270 ms, 518 MB/s
  Run 2: 272 ms, 514 MB/s
```

np.savez cost: ~270 ms/file  
dgen-py generation (BytesView from singleton): < 10 ms  
Upload 140 MiB at ~140 MB/s per rank: ~1 s/file

### Where 270ms goes in np.savez

1. `ZipFile` object creation + internal buffer setup: ~1 ms
2. NPY header write: ~0.1 ms
3. Array data write to BytesIO (140 MiB memcpy): ~130 ms (at ~1 GB/s BytesIO write speed)
4. ZIP local file header + CRC32 computation: ~140 ms (CRC32 at ~1 GB/s)

Key observation: `np.savez` creates an uninitialized `BytesIO`, then grows it from 0 → 140 MiB via ZipFile writes. Python's `BytesIO` uses a `bytearray` internally that **doubles on reallocation** — this causes multiple 70+ MiB allocations and copies during the write.

---

## 5. NPZ Format Structure

NPZ = ZIP archive containing `.npy` files.

NPY 1.0 format:
```
\x93NUMPY          (6 bytes magic)
\x01\x00           (2 bytes: version 1.0)
HLEN               (2 bytes LE: header data length)
HEADER_DICT\n      (HLEN bytes: Python dict string, padded to 64-byte boundary)
DATA               (raw array bytes, C-contiguous little-endian)
```

**Key insight from user:** The DATA bytes do NOT need to be valid float32 values. Any random bytes are acceptable since the training workload discards data after benchmarking. Only the NPY header (shape, dtype, format descriptors) needs to be correct.

---

## 6. Optimization Strategy

### Strategy A: Fix the Installation (IMMEDIATE — critical)

Update mlp-storage's `uv.lock` to use local editable dlio_benchmark:
```toml
# pyproject.toml [tool.uv.sources]
dlio-benchmark = { path = "/home/eval/Documents/Code/dlio_benchmark", editable = true }
```

**Expected impact:** Enables async pipeline + dgen singleton → likely ~3-4× speedup from 1.11 GB/s to 3-5 GB/s.

### Strategy B: Bypass numpy for NPZ serialization

Current path:
```
gen_random_tensor() → ndarray(6053,6053,1)  ~10ms
np.savez(BytesIO, x=arr, y=[0])             ~270ms  (BytesIO growth + CRC32)
put_data(path, BytesIO)                     ~1000ms
```

Optimized path:
```
dgen_py.generate_buffer(total_bytes)        ~10ms   (BytesView, no copy)
build_npz_raw(BytesView, shape)             ~?ms    (manual ZIP+NPY, pre-alloc)
put_data(path, BytesIO)                     ~?ms
```

Techniques:
1. **Pre-allocate BytesIO** to exact NPZ size → avoid BytesIO reallocation overhead
2. **Skip numpy array creation** — use `bytes(BytesView)` directly as NPY data
3. **Stream-write via `zf.open()`** — avoids building combined `npy_header + data` bytes
4. **Buffer protocol write** — `zf.open('x.npy','w').write(bytesview)` — zero extra copy if ZipFile accepts bytes-like objects

### Strategy C: Rust NPZ generator in s3dlio

Add Python-callable Rust function:
```python
s3dlio.generate_npz_bytes(shape=(6053,6053,1), dtype='<f4') -> bytes
```

Internally:
- dgen-rs generates random bytes (Rayon parallel, ~15 GB/s)
- NPY header built from shape/dtype parameters
- ZIP STORED wrapper constructed without Python GIL
- Returns `Bytes` zero-copy via PyO3

**Expected impact:** ~500+ MB/s → 1+ GB/s per rank serialization (Rust memcpy vs Python BytesIO growth).

### Strategy D: Direct scatter/gather PUT (longest-term)

Use `s3dlio.put_many()` or multipart upload to stream NPY header + raw dgen bytes directly to S3 without any BytesIO intermediary. Eliminates all copying.

---

## 7. Arithmetic: Path to 8 GB/s

With NP=8 ranks:
- Each rank needs: 8 GB/s ÷ 8 = 1 GB/s per rank
- Each rank uploads 21 files × 139.8 MiB = 2936 MiB
- At 1 GB/s: 2936 MiB / 1024 MB/GiB × 1 s/GB ≈ 2.9 s per rank

For 2.9 s total per rank:
- Async pipeline: generation of 21 files = 21 × 10ms (dgen) = 210ms (if savez removed)
- 21 uploads, 8 concurrent: ceil(21/8) × upload_time_per_file ≤ 2.9s
- Max upload time per file: 2.9s / 3 batches ≈ 970ms
- Required per-file upload speed: 139.8 MiB / 970ms ≈ 144 MB/s per rank

s3-ultra capability: 47,883 MB/s for 1 MiB on loopback, 49,926 MB/s for 8 MiB.
With 8 concurrent ranks × 1 connection each: should be well above 144 MB/s/rank.

**Bottleneck is likely the async pipeline not being used (installation bug), followed by np.savez overhead.**

---

## 8. s3-ultra Large Object Note

From Performance.md: "Objects > 32 MiB use streaming path — Chunked encoding, slightly higher overhead."

Our 139.8 MiB files are 4× over the 32 MiB threshold. The PUT path uses chunked transfer encoding which:
1. Doesn't send `Content-Length` upfront
2. Requires chunked encoding overhead
3. s3dlio may not pipeline chunks optimally

Potential fix in s3-ultra: buffer large objects up to a threshold and use `Content-Length` response for GETs.

---

## 9. Experiment Log

### Experiment 1 — Baseline (2026-04-25)
- **Config:** unet3d, NP=8, s3dlio, endpoint 127.0.0.1:9101
- **Runtime:** 21.2 s, **Throughput:** 1.11 GB/s
- **Note:** Using OLD installed dlio_benchmark (stale git wheel — async pipeline NOT active)

### Experiment 2 — Baseline minio (2026-04-25)  
- **Config:** unet3d, NP=8, minio, endpoint 127.0.0.1:9101
- **Runtime:** 24.7 s, **Throughput:** 0.95 GB/s
- **Note:** Same stale install issue

### Experiment 3 — (PLANNED) Fix installation, re-run
- Fix: `uv add --editable /home/eval/Documents/Code/dlio_benchmark` in mlp-storage
- Expected: significant improvement from async pipeline

### Experiment 4 — (PLANNED) Fast NPZ path
- Bypass np.savez with raw-bytes NPZ builder
- Expected: save ~260ms/file serialization overhead

### Experiment 5 — (PLANNED) s3dlio Rust NPZ generator
- Add `generate_npz_bytes()` to s3dlio Python API
- Build/install new s3dlio wheel
- Expected: eliminate Python overhead entirely for serialization

---

## 10. Test Infrastructure Notes

- s3-ultra: PID 3765782, `0.0.0.0:9101`, db `/tmp/s3-ultra-mlp-test`
- Buckets: `mlp-s3dlio`, `mlp-minio`, `mlp-s3torch`
- mlp-storage: `/home/eval/Documents/Code/mlp-storage/`, `uv run`
- dlio_benchmark source: `/home/eval/Documents/Code/dlio_benchmark/` (our modified version)
- s3dlio source: `/home/eval/Documents/Code/s3dlio/`
- All commands via: `uv run mlpstorage training datagen ...`
- NEVER use boto3 or aws-cli — always `s3-cli`
