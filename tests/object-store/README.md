# Object Store Tests

Performance tests and benchmarks for object storage backends (s3dlio, minio,
s3torchconnector) used by `mlpstorage`.

All tests load credentials from a `.env` file at the **project root** (`mlp-storage/.env`):

```
AWS_ACCESS_KEY_ID=<key>
AWS_SECRET_ACCESS_KEY=<secret>
AWS_ENDPOINT_URL=http://<host>:<port>
AWS_REGION=us-east-1
```

For HTTPS endpoints with a self-signed certificate, set the CA bundle path:

```bash
export AWS_CA_BUNDLE=/path/to/selfsigned.crt
```

`AWS_CA_BUNDLE` is read by s3dlio and by the Python test scripts in this directory.
s3torchconnector also reads the same `AWS_CA_BUNDLE` name. See **[How to Test with SSL (HTTPS)](#how-to-test-with-ssl-https)** below
for full setup instructions.

Environment variables already set in the shell take precedence over the `.env` file.
No credentials are hard-coded in any test.

---

## How to Test with SSL (HTTPS)

By default all tests use plain HTTP (`http://`). If you want to test with HTTPS — for
example against a MinIO instance configured with TLS — there are several steps required
because each library resolves TLS trust differently.

### Step 1 — Generate the correct server certificate (on the MinIO host)

The certificate **must** be generated with `basicConstraints=CA:FALSE`. Rust-based
libraries (s3dlio, s3torchconnector) use **rustls**, which strictly enforces RFC 5280
and rejects any server certificate that advertises itself as a CA (`CA:TRUE`). OpenSSL
and curl do not enforce this, so the error only appears with Rust clients.

```bash
# Run on the MinIO server as root (or the MinIO user)
openssl req -x509 -newkey rsa:4096 -sha256 -days 3650 -nodes \
  -keyout /home/minio-user/.minio/certs/private.key \
  -out    /home/minio-user/.minio/certs/public.crt \
  -subj "/CN=<minio-ip-or-hostname>" \
  -addext "subjectAltName=IP:<minio-ip-or-hostname>" \
  -addext "basicConstraints=CA:FALSE" \
  -addext "keyUsage=digitalSignature,keyEncipherment" \
  -addext "extendedKeyUsage=serverAuth"
```

Replace `<minio-ip-or-hostname>` with your MinIO server's IP or DNS name, e.g.
`your-minio-host`.  The `subjectAltName` is **required** — modern TLS clients reject
certificates that only set a `CN` with no SAN.

Fix ownership then restart MinIO:

```bash
chown minio-user:minio-user /home/minio-user/.minio/certs/private.key \
                             /home/minio-user/.minio/certs/public.crt
chmod 600 /home/minio-user/.minio/certs/private.key
chmod 644 /home/minio-user/.minio/certs/public.crt
systemctl restart minio
systemctl status minio    # verify it came up cleanly
```

### Step 2 — Copy the certificate to the client machine

```bash
# Run on the client (e.g. loki-russ)
scp <minio-user>@<minio-host>:/home/minio-user/.minio/certs/public.crt \
    ~/Documents/Code/mlp-storage/.certs/minio-selfsigned.crt
```

### Step 3 — Trust the certificate on the client

```bash
sudo cp ~/Documents/Code/mlp-storage/.certs/minio-selfsigned.crt \
    /usr/local/share/ca-certificates/minio-selfsigned.crt
sudo update-ca-certificates
# Expected output: "1 added, 0 removed; done."
```

> **Note — linuxbrew Python:** If Python is installed via linuxbrew
> (`/home/linuxbrew/...`), its OpenSSL is isolated from the system CA store.
> The minio Python SDK will **not** pick up the cert from `update-ca-certificates`
> automatically.  See **Step 5** below.

### Step 4 — Verify with curl and openssl

```bash
# 1. Quick TLS check — should negotiate TLS and return HTTP 403 (AccessDenied is expected)
curl -v https://<minio-ip>:9000/

# 2. Inspect the deployed certificate
openssl x509 -in /usr/local/share/ca-certificates/minio-selfsigned.crt \
    -noout -text | grep -A3 "Basic Constraints"
# Must show: CA:FALSE

# 3. Confirm SAN is present
openssl x509 -in /usr/local/share/ca-certificates/minio-selfsigned.crt \
    -noout -text | grep -A2 "Subject Alternative Name"
# Must show: IP Address:<minio-ip>
```

A successful curl output will include:
```
* SSL certificate verify ok.
* subjectAltName: host "<minio-ip>" matched cert's IP address!
< HTTP/1.1 403 Forbidden   ← expected; means TLS is working
```

### Step 5 — Configure each library

Update `.env` to use `https://`:

```
AWS_ENDPOINT_URL=https://<minio-ip>:9000
```

Set the CA bundle environment variable (required even with a system-store cert, because
not all libraries read the system store):

```bash
export AWS_CA_BUNDLE=/usr/local/share/ca-certificates/minio-selfsigned.crt
```

#### How each library resolves TLS trust

Each library takes a different path to TLS certificate verification:

| Library | TLS layer | Reads `AWS_CA_BUNDLE` | Reads system store | How trust is established |
|---|---|---|---|---|
| s3dlio | Rust/rustls | ✅ | ✅ rustls-native-certs | `AWS_CA_BUNDLE` env var, or system store after `update-ca-certificates` |
| minio Python SDK | Python/urllib3/OpenSSL | ❌ | ❌ (linuxbrew isolates it) | Custom `urllib3.PoolManager(ssl_context=ctx)` built from `AWS_CA_BUNDLE` — handled automatically in `test_s3lib_get_bench.py` |
| s3torchconnector | Rust/AWS SDK for Rust | ✅ | ✅ rustls-native-certs | System store pickup after `update-ca-certificates`, or `AWS_CA_BUNDLE` env var |

**Key points:**
- All three libraries now share the same env var name: `AWS_CA_BUNDLE` (the standard AWS SDK convention).
  `test_s3lib_get_bench.py` reads it and passes the path to urllib3 explicitly for the minio Python SDK.
- The minio Python SDK ignores AWS env vars entirely. `test_s3lib_get_bench.py`
  reads `AWS_CA_BUNDLE` and passes it to urllib3 explicitly via
  `_make_minio_client()`.
- rustls enforces RFC 5280 strictly: a certificate with `basicConstraints: CA:TRUE` is
  rejected with `CaUsedAsEndEntity` even if it is trusted. OpenSSL/curl silently accept
  it. This is why the cert **must** be generated with `basicConstraints=CA:FALSE`.
- s3torchconnector reads the system CA store via `rustls-native-certs`, so
  `update-ca-certificates` is sufficient for it without any extra env var.

---

## Library Selection — `storage_library` YAML Key

The `storage_library` key in the YAML config controls **which S3 client library is used**
for all I/O operations (reads, writes, listing). It lives in the `storage:` section —
**not** in `dataset:`.

```yaml
storage:
  storage_type: s3          # the protocol family ("s3" = object storage)
  storage_root: mlp-minio   # the bucket name
  storage_library: minio    # which library to use ← this is the selector
```

**Valid values:**

| `storage_library` | Library | Notes |
|---|---|---|
| `s3dlio` | s3dlio (Rust-based, Tokio async) | `get_many()` parallel batch, `MultipartUploadWriter` |
| `minio` | minio Python SDK | `ThreadPoolExecutor`, automatic 5 MB multipart |
| `s3torchconnector` | Amazon s3torchconnector (Rust) | `S3Client.get_object()` (direct, optimal); ⚠️ DLIO reader currently uses `S3IterableDataset` (sequential, 1 GET/worker) — see `S3library_review_21-Mar.md` |

The three separate workload configs differ only on this key (and the bucket name):
- `configs/dlio/workload/unet3d_h100_s3dlio.yaml` → `storage_library: s3dlio`
- `configs/dlio/workload/unet3d_h100_minio.yaml` → `storage_library: minio`
- `configs/dlio/workload/unet3d_h100_s3torch.yaml` → `storage_library: s3torchconnector`

### How `storage_library` flows from YAML → code

1. **`config.py` (LoadConfig, ~line 1094–1097):** `LoadConfig` reads
   `storage.storage_library` from the YAML and **injects it** into
   `args.storage_options["storage_library"]`. This is necessary because DLIO's `Args`
   dataclass has no first-class `storage_library` field — the value piggybacks inside
   the free-form `storage_options` dict.

2. **`config.py` (Args.validate(), ~line 387):** `validate()` reads it back from
   `storage_options.get("storage_library", "s3torchconnector")` (default is
   `s3torchconnector` for backwards compat with configs that predate this key).
   It uses the value to:
   - Verify the library package is installed (fails fast with a clear error if not)
   - Set the correct `reader_classname` for the DataLoader
   - Enforce the right `checkpoint_mechanism` (`pt_s3_save` for s3torchconnector,
     `pt_obj_save` for minio / s3dlio)

3. **`storage/obj_store_lib.py` (`ObjStoreLibStorage.__init__()`, ~lines 161–166):**
   Reads `storage_options.get("storage_library")` and instantiates the correct client:

   ```python
   if storage_library == "s3dlio":
       # s3dlio Rust client
   elif storage_library == "s3torchconnector":
       # S3Client from s3torchconnector
   elif storage_library == "minio":
       # Minio Python SDK client
   ```

   This single branch point controls all read, write, and list operations for the
   entire training/datagen run.

---

## Results

**[S3library_review_21-Mar.md](S3library_review_21-Mar.md)** — Prefetch fairness code review (March 21, 2026): analysis of concurrency models across all three libraries in the DLIO reader, root cause of the s3torchconnector benchmark gap, and remediation options. Includes s3dlio v0.9.84 fix status.

**[Object_Perf_Results.md](Object_Perf_Results.md)** — Full benchmark results including:
- Direct native-API write + read throughput (all three libraries, 12 parallel workers)
- DLIO streaming checkpoint write + read throughput (16 GB and 100 GB)
- DLIO training MPI sweep (N=1, 2, 4 processes × all three libraries)
- Analysis of DLIO overhead vs native API performance

---

## Test Files

### Cross-Library Comparisons

#### `test_s3lib_get_bench.py`
Benchmarks **GET throughput** across all three libraries with three rigorously fair
test modes. All libraries read from the **same bucket and same objects** — no
per-library data locality effects.

| Mode | What it measures | Concurrency model |
|---|---|---|
| `serial` | Per-request latency (p50/p95/p99/max) + single-stream MB/s | One GET at a time, no parallelism |
| `parallel` | Aggregate MB/s at matched concurrency | `ThreadPoolExecutor(max_workers=N)` — identical across all libraries |
| `native` | s3dlio Rust async vs Python threads | `s3dlio.get_many(uris, max_in_flight=N)` |

```bash
cd mlp-storage && source .venv/bin/activate

# Default: all modes, existing training data (mlp-s3dlio bucket), concurrency 1/4/8/16
python tests/object-store/test_s3lib_get_bench.py

# Write 20 synthetic 128 MB objects first, then run all tests against them
python tests/object-store/test_s3lib_get_bench.py \
    --write --write-num-files 20 --write-size-mb 128

# Serial-only test — per-request latency and single-stream MB/s
python tests/object-store/test_s3lib_get_bench.py --mode serial --num-files 30

# Parallel sweep with custom worker counts
python tests/object-store/test_s3lib_get_bench.py \
    --mode parallel --workers 1 4 8 16 32 64

# Test only s3dlio native get_many (Rust Tokio async) vs ThreadPoolExecutor
python tests/object-store/test_s3lib_get_bench.py \
    --mode native --workers 1 4 8 16 32

# Test only two libraries
python tests/object-store/test_s3lib_get_bench.py --libraries s3dlio minio

# Custom bucket and prefix
python tests/object-store/test_s3lib_get_bench.py \
    --bucket my-bucket --prefix data/train/ --num-files 50

# CLI reference
python tests/object-store/test_s3lib_get_bench.py --help
```

#### Sample Output

*Results below use HTTPS (with a self-signed MinIO certificate
and `AWS_CA_BUNDLE` set — the more realistic and secure configuration.*

```console
(.venv) eval@loki-russ:~/Documents/Code/mlp-storage$ python ./tests/object-store/test_s3lib_get_bench.py
Loaded credentials from: /path/to/mlp-storage/.env

════════════════════════════════════════════════════════════════════════
S3 LIBRARY GET BENCHMARK
════════════════════════════════════════════════════════════════════════
  Endpoint:   https://minio-host:9000
  Libraries:  s3dlio, minio, s3torchconnector
  Mode:       all
  Workers:    [1, 4, 8, 16]  (concurrency sweep)

── Listing objects ──────────────────────────────────────────────────────
  Bucket: mlp-s3dlio  Prefix: test-run/unet3d/train/  (max 20)
  Found 20 objects  (first: test-run/unet3d/train/img_000_of_168.npz)
[s3dlio] Loading CA bundle from: /usr/local/share/ca-certificates/minio-172-16-1-40_selfsigned.crt
  Objects:  20 × 213.7 MB = 4274 MB total

── Serial GET ───────────────────────────────────────────────────────────
  [s3dlio              ] serial: 20 × 1 GET …
  [s3dlio              ]  done: 515 MB/s (stream), p50=0.279s
  [minio               ] serial: 20 × 1 GET …
  [minio               ]  done: 511 MB/s (stream), p50=0.280s
  [s3torchconnector    ] serial: 20 × 1 GET …
  [s3torchconnector    ]  done: 389 MB/s (stream), p50=0.358s

── Parallel GET (ThreadPoolExecutor) ────────────────────────────────────
  [s3dlio              ] parallel workers=  1: …    574 MB/s
  [minio               ] parallel workers=  1: …    507 MB/s
  [s3torchconnector    ] parallel workers=  1: …    402 MB/s
  [s3dlio              ] parallel workers=  4: …   1049 MB/s
  [minio               ] parallel workers=  4: …   1025 MB/s
  [s3torchconnector    ] parallel workers=  4: …    544 MB/s
  [s3dlio              ] parallel workers=  8: …   1065 MB/s
  [minio               ] parallel workers=  8: …    930 MB/s
  [s3torchconnector    ] parallel workers=  8: …    516 MB/s
  [s3dlio              ] parallel workers= 16: …   1043 MB/s
  [minio               ] parallel workers= 16: …    916 MB/s
  [s3torchconnector    ] parallel workers= 16: …    570 MB/s

── s3dlio native get_many() ─────────────────────────────────────────────
  [s3dlio native       ] get_many max_in_flight=  1: …    653 MB/s
  [s3dlio native       ] get_many max_in_flight=  4: …    946 MB/s
  [s3dlio native       ] get_many max_in_flight=  8: …    971 MB/s
  [s3dlio native       ] get_many max_in_flight= 16: …    972 MB/s
```

**Serial GET** — one object at a time, no parallelism (20 objects)

| Library | p50 | p95 | p99 | max | MB/s |
|---|---|---|---|---|---|
| s3dlio | 0.279s | 0.454s | 0.498s | 0.509s | **515 ◀** |
| minio | 0.280s | 0.449s | 0.464s | 0.468s | 511 |
| s3torchconnector | 0.358s | 0.600s | 0.633s | 0.641s | 389 |

*p50/p95/p99/max — per-GET wall-clock latency (s) · MB/s — single-stream throughput (sum\_bytes / sum\_latency) · ◀ = fastest library*

**Parallel GET** — `ThreadPoolExecutor`, same concurrency for all (20 objects, same bucket + objects for all libraries)

| Library | w=1 | w=4 | w=8 | w=16 |
|---|---|---|---|---|
| s3dlio | **574 ◀** | **1,049 ◀** | **1,065 ◀** | **1,043 ◀** |
| minio | 507 | 1,025 | 930 | 916 |
| s3torchconnector | 402 | 544 | 516 | 570 |

*All values in MB/s · All libraries use `ThreadPoolExecutor(max_workers=N)` — identical concurrency model · ◀ = fastest library at that worker count*

**s3dlio Native get_many()** — Rust Tokio async, s3dlio only (20 objects)

| max\_in\_flight | MB/s | vs ThreadPoolExecutor |
|---|---|---|
| 1 | 653 | +13.7% vs w=1 |
| 4 | 946 | −9.8% vs w=4 |
| 8 | 971 | −8.9% vs w=8 |
| 16 | 972 | −6.9% vs w=16 |

*`get_many()` uses s3dlio's Rust Tokio async engine; all requests are scheduled in a single Rust thread pool — no Python GIL or thread creation overhead.*

---

#### `test_direct_write_comparison.py`
Measures **native API write + read throughput** across all three libraries side-by-side,
without any DLIO involvement. Each library gets its own dedicated bucket.

```bash
cd mlp-storage && source .venv/bin/activate

# Default: 100 × 128 MiB objects, 8 write + 8 read workers, all three libraries
python tests/object-store/test_direct_write_comparison.py

# Reproduce the 12-worker results in Object_Perf_Results.md
python tests/object-store/test_direct_write_comparison.py \
    --num-files 100 --size-mb 128 --write-workers 12 --read-workers 12

# Single library
python tests/object-store/test_direct_write_comparison.py --library s3dlio

# CLI reference
python tests/object-store/test_direct_write_comparison.py --help
```

#### `test_dlio_multilib_demo.py`
Runs **DLIO-driven training and checkpoint workloads** across all three libraries.
I/O goes through DLIO's MPI data generation and PyTorch DataLoader — this is the
realistic DLIO performance as seen by a training job, not direct API throughput.

```bash
cd mlp-storage && source .venv/bin/activate

# Training workload (100 × 128 MiB NPZ, 2 epochs)
python tests/object-store/test_dlio_multilib_demo.py --workload training

# Checkpoint workload (~105 GB streaming checkpoint, llama3-8b profile)
python tests/object-store/test_dlio_multilib_demo.py --workload checkpoint

# Single library
python tests/object-store/test_dlio_multilib_demo.py --workload training --library s3dlio
```

#### `test_training_mpi_sweep.py`
Sweeps MPI **process count (N = 1, 2, 4)** for both datagen and training across all
three libraries. Each (library, N) combination runs as an independent clean cycle:
`clean → datagen(N) → train(N) → clean`. Both write (datagen) and read (training)
throughput are measured at each N.

```bash
cd mlp-storage && source .venv/bin/activate

# Full sweep: all libraries, N = 1, 2, 4
python tests/object-store/test_training_mpi_sweep.py

# Custom process counts
python tests/object-store/test_training_mpi_sweep.py --process-counts 1 2 4 8

# Single library
python tests/object-store/test_training_mpi_sweep.py --library s3dlio

# Skip datagen (use data already in bucket)
python tests/object-store/test_training_mpi_sweep.py --skip-datagen

# Keep objects after the run (skip cleanup)
python tests/object-store/test_training_mpi_sweep.py --skip-cleanup
```

---

### Per-Library Checkpoint Tests

Each of these tests the `StreamingCheckpointing` pipeline for a single library:
a fixed-RAM streaming producer-consumer pipeline where dgen-py generates data
concurrently while the library uploads it. Memory usage is constant at ~128 MB
regardless of checkpoint size.

#### `test_s3dlio_checkpoint.py`
StreamingCheckpointing with the **s3dlio** backend.

```bash
cd mlp-storage && source .venv/bin/activate
python tests/object-store/test_s3dlio_checkpoint.py --size-gb 16
python tests/object-store/test_s3dlio_checkpoint.py --size-gb 100
python tests/object-store/test_s3dlio_checkpoint.py --help
```

#### `test_s3torch_checkpoint.py`
StreamingCheckpointing with the **s3torchconnector** backend.

```bash
cd mlp-storage && source .venv/bin/activate
python tests/object-store/test_s3torch_checkpoint.py --size-gb 16
python tests/object-store/test_s3torch_checkpoint.py --help
```

#### `test_minio_checkpoint.py`
StreamingCheckpointing with the **minio** backend.

```bash
cd mlp-storage && source .venv/bin/activate
python tests/object-store/test_minio_checkpoint.py --size-gb 16
python tests/object-store/test_minio_checkpoint.py --help
```

---

### Direct s3dlio API Tests

#### `test_s3dlio_direct.py`
Tests the two s3dlio write APIs directly (no DLIO, no mlpstorage wrapper):
- `PyObjectWriter` — streaming writer (`write_chunk` + `finalize`)
- `MultipartUploadWriter` — multipart upload (`write` + `close`)

```bash
cd mlp-storage && source .venv/bin/activate

# Uses defaults from .env (bucket: bucket-s3dlio)
python tests/object-store/test_s3dlio_direct.py

# Custom bucket
python tests/object-store/test_s3dlio_direct.py --bucket my-bucket
python tests/object-store/test_s3dlio_direct.py --help
```

---

### Shell Script Tests

These shell scripts run the full `mlpstorage` CLI pipeline for each library —
datagen, training, and checkpoint — using the **standard unet3d h100 workload**
(`unet3d_h100.yaml`): 168 files × ~140 MB each (~23 GB total), batch_size=7,
5 epochs, computation_time=0.323 s. This matches the real MLPerf Storage h100
submission workload.

#### `test_mlp_s3dlio.sh`
Full mlpstorage datagen + training with **s3dlio** as the storage backend,
using the standard unet3d h100 workload paramters.

```bash
cd mlp-storage
bash tests/object-store/test_mlp_s3dlio.sh
```

#### `test_mlp_minio.sh`
Full mlpstorage datagen + training with **minio** as the storage backend,
using the standard unet3d h100 workload parameters.

```bash
cd mlp-storage
bash tests/object-store/test_mlp_minio.sh
```

#### `test_mlp_s3torch.sh`
Full mlpstorage datagen + training with **s3torchconnector** as the storage backend,
using the standard unet3d h100 workload parameters.

```bash
cd mlp-storage
bash tests/object-store/test_mlp_s3torch.sh
```

#### `test_s3dlio_multilib.sh`
Shell-based multi-library comparison using s3dlio directly (not via mlpstorage).

```bash
cd mlp-storage
bash tests/object-store/test_s3dlio_multilib.sh
```

#### `demo_streaming_checkpoint.sh`
Quickstart demo showing the two major optimisations: dgen-py integration (155×
faster data generation) and StreamingCheckpointing (192× memory reduction).
Compares old vs new method for both file and object storage.

```bash
TEST_SIZE_GB=1 TEST_CHECKPOINT_DIR=/tmp/ckpt-demo \
    bash tests/object-store/demo_streaming_checkpoint.sh
```

---

## Credential Setup

Create `mlp-storage/.env` (never commit this file):

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_ENDPOINT_URL=http://your-minio-host:9000
AWS_REGION=us-east-1
```

`.env` is already listed in `.gitignore`. All scripts and Python tests read it
automatically at startup; shell environment variables always take precedence.

---

## Real Checkpoint Tests — `dlio_xxx_checkpoint.sh`

These scripts run **end-to-end LLaMA 3 8B checkpoint workloads** directly through
`dlio_benchmark` using the mlp-storage storage backends. They are the authoritative
benchmark for checkpoint write and read throughput, equivalent to what a real
distributed training run produces during a checkpoint save/restore cycle.

> **No data generation required** — checkpoint workloads synthesize tensor data
> on the fly using the model sizing parameters. Run these tests standalone without
> any prior `datagen` step.

### Common parameters

| Variable | Default | Description |
|---|---|---|
| `NP` | `1` | MPI rank count — simulates that many GPU processes |
| `CHECKPOINTS` | `2` | Number of checkpoint write + read cycles |

**NP guidance:**

> **Important:** NP controls the number of shards, **not** the total amount of data.
> The LLaMA 3 8B checkpoint has two components that are always saved together:
> model weights (~16 GB, fp16) and optimizer state (~89 GB, fp32). Combined that is
> ~105 GB total per checkpoint. All NP settings produce the same ~105 GB total I/O —
> NP only splits that data into more, smaller per-rank objects.

| NP | Total I/O per checkpoint | Per-rank object size | s3dlio | minio | s3torchconnector |
|---|---|---|---|---|---|
| `1` | ~105 GB write + ~105 GB read | ~105 GB | ✅ | ✅ | ❌ fails (> 78 GB limit) |
| `2` | ~105 GB write + ~105 GB read | ~52.5 GB | ✅ | ✅ | ✅ |
| `4` | ~105 GB write + ~105 GB read | ~26 GB | ✅ | ✅ | ✅ |
| `8` | ~105 GB write + ~105 GB read | ~13.1 GB | ✅ | ✅ | ✅ |

> **s3torchconnector NP=1 failure:** The AWS CRT library (used internally by
> s3torchconnector) cannot write a single object larger than approximately 78 GB. At
> NP=1 the full ~105 GB checkpoint (weights + optimizer state) is written as one object,
> which exceeds this limit and causes the upload to fail. Use NP=2 or larger with
> s3torchconnector — with 2 ranks the per-rank shard is ~52.5 GB, well within the CRT
> limit. s3dlio and minio are not affected by this limit.

Each rank independently writes its shard to a unique object key under:
```
s3://chckpt-test1/<library>/llama3-8b/<checkpoint_id>/<rank>.pt
```

### Prerequisites

```bash
cd /path/to/mlp-storage
source .venv/bin/activate

# Ensure credentials and endpoint are set
source .env

# Verify bucket exists and is reachable
python3 -c "import s3dlio; print(s3dlio.list('s3://chckpt-test1/', recursive=False))"
```

For HTTPS endpoints (self-signed MinIO certificate), set:
```bash
# Already in .env if configured — verify with:
echo $AWS_CA_BUNDLE    # should point to the .crt file
```

### Scripts

All three scripts share identical interface — only the storage library and bucket
prefix differ.

#### `dlio_s3dlio_checkpoint.sh` — s3dlio (Rust / Tokio)

```bash
cd /path/to/mlp-storage

# Single-rank sanity check (default, ~13 GB I/O)
bash tests/object-store/dlio_s3dlio_checkpoint.sh

# 2-rank run
NP=2 bash tests/object-store/dlio_s3dlio_checkpoint.sh

# Full 8-rank llama3-8b reference (~89 GB total, 8 × ~11 GB shards)
NP=8 bash tests/object-store/dlio_s3dlio_checkpoint.sh

# Quick 1-checkpoint run (write once, read once)
CHECKPOINTS=1 bash tests/object-store/dlio_s3dlio_checkpoint.sh

# Combine overrides
NP=4 CHECKPOINTS=1 bash tests/object-store/dlio_s3dlio_checkpoint.sh
```

Objects land at: `s3://chckpt-test1/s3dlio/llama3-8b/`

#### `dlio_minio_checkpoint.sh` — minio Python SDK

```bash
cd /path/to/mlp-storage

bash tests/object-store/dlio_minio_checkpoint.sh          # NP=1 (default)
NP=2 bash tests/object-store/dlio_minio_checkpoint.sh
NP=8 bash tests/object-store/dlio_minio_checkpoint.sh    # full reference
CHECKPOINTS=1 bash tests/object-store/dlio_minio_checkpoint.sh
```

Objects land at: `s3://chckpt-test1/minio/llama3-8b/`

#### `dlio_s3torch_checkpoint.sh` — s3torchconnector (AWS CRT)

> ⚠️ **Known limitation — NP=1 will fail.**  The AWS CRT library used by
> s3torchconnector cannot write a single object larger than ~78 GB. At NP=1 the full
> LLaMA 3 8B checkpoint (~105 GB: model weights ~16 GB + optimizer state ~89 GB) is
> written as one object and the upload fails with a CRT internal error.  **Always use
> NP≥2 with s3torchconnector.**  This is not a configuration problem — it is a hard
> limit in the AWS CRT library.

```bash
cd /path/to/mlp-storage

# NP=1 WILL FAIL for llama3-8b (105 GB object > 78 GB CRT limit)
# bash tests/object-store/dlio_s3torch_checkpoint.sh

# Minimum working rank count for s3torchconnector
NP=2 bash tests/object-store/dlio_s3torch_checkpoint.sh
NP=4 bash tests/object-store/dlio_s3torch_checkpoint.sh
NP=8 bash tests/object-store/dlio_s3torch_checkpoint.sh  # full reference
CHECKPOINTS=1 bash tests/object-store/dlio_s3torch_checkpoint.sh
```

Objects land at: `s3://chckpt-test1/s3torch/llama3-8b/`

> **Note:** `s3torchconnector` only supports AWS S3 and S3-compatible endpoints that
> accept AWS Signature V4. It does not support Azure or GCS endpoints.

### Progress output

During a checkpoint write each library prints a live throughput line that updates in
place (carriage-return style):

```
[Writer] 6.55 GB, 0.31 GB/s   
```

The line shows cumulative GB written and the current instantaneous throughput. When the
upload completes the line is finalised with a newline and DLIO prints per-rank summary
statistics.

### Cleanup

After a run, delete the objects to reclaim bucket space:

```bash
bash tests/object-store/dlio_s3dlio_cleanup.sh
bash tests/object-store/dlio_minio_cleanup.sh
bash tests/object-store/dlio_s3torch_cleanup.sh
```

---

## Full Workflow — Datagen → Train → Checkpoint

The scripts below run the complete DLIO UNet3D H100 workload for each library. Use
these when you want to benchmark **training data loading** rather than checkpointing.

### Phase 1 — Generate training data

Data generation writes synthetic NPZ files to the object store. This is a one-time
step per bucket/library combination; you can reuse the generated data for multiple
training runs.

```bash
# Generate UNet3D training data (do this once per library bucket)
bash tests/object-store/dlio_s3dlio_datagen.sh    # → mlp-s3dlio bucket
bash tests/object-store/dlio_minio_datagen.sh     # → mlp-minio bucket
bash tests/object-store/dlio_s3torch_datagen.sh   # → mlp-s3torch bucket
```

Override the number of samples (default varies per config):
```bash
NUM_FILES=100 bash tests/object-store/dlio_s3dlio_datagen.sh
```

### Phase 2 — Training throughput

Runs the training I/O loop (no actual GPU compute — pure storage benchmark):

```bash
NP=1  bash tests/object-store/dlio_s3dlio_train.sh
NP=2  bash tests/object-store/dlio_minio_train.sh
NP=4  bash tests/object-store/dlio_s3torch_train.sh
```

### Phase 3 — Checkpoint (standalone)

See **[Real Checkpoint Tests](#real-checkpoint-tests--dlio_xxx_checkpointsh)** above.
Checkpointing does not require training data — it runs independently.

### Phase 4 — Full cycle (datagen + train + checkpoint)

```bash
bash tests/object-store/dlio_s3dlio_cycle.sh    # all three phases, s3dlio
bash tests/object-store/dlio_minio_cycle.sh     # all three phases, minio
bash tests/object-store/dlio_s3torch_cycle.sh   # all three phases, s3torch
```

### Cleanup

```bash
bash tests/object-store/dlio_s3dlio_cleanup.sh
bash tests/object-store/dlio_minio_cleanup.sh
bash tests/object-store/dlio_s3torch_cleanup.sh
```
