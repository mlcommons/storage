# Object-Store Tests

Tests for S3-compatible object storage backends used by `mlpstorage` and `dlio_benchmark`.

All scripts read credentials and runtime configuration from a `.env` file at the
**project root** (`mlp-storage/.env`) — no credentials or site-specific values are
embedded in any script or config file.

---

## Recommended Hardware

**Linux only** — macOS and Windows are not supported.

These are minimum requirements per `NP` (number of simulated accelerators).
Running below spec will likely cause OOM crashes:

| NP | CPU cores (incl. threads) | RAM |
|:---:|---:|---:|
| 1 | 8 | 16 GB |
| 2 | 16 | 32 GB |
| 4 | 32 | 64 GB |
| 8 | 64 | 128 GB |

NP scales linearly — each doubling of NP requires 2× the CPU and RAM.
You may be able to run some workloads below these numbers, but OOM crashes are expected.

---

## Structure

```
tests/object-store/
│
├── — Data Generators (run once, before benchmarking) ——————————————
│   gen_retinanet_jpeg.sh   generate 50k JPEG files for RetinaNet (~15 GiB)
│   gen_unet3d_npz.sh       generate 7,200 NPZ files for UNet3D   (~984 GiB)
│                           (DLRM and Flux generate data inline via run_*_bench.sh)
│
├── — Benchmark Runners ————————————————————————————————————————————
│   run_dlrm_bench.sh       DLRM:      Parquet, NP=1..8, prints AU + throughput
│   run_flux_bench.sh       Flux:      Parquet, NP=1..8, prints AU + throughput
│   test_retinanet.sh       RetinaNet: JPEG,    NP=1..4, smoke test + benchmark
│   test_unet3d.sh          UNet3D:    NPZ,     NP=1..4, smoke test + benchmark
│
├── — Checkpointing ————————————————————————————————————————————————
│   run_checkpointing.sh    LLaMA 3 8B checkpoint write + read (s3dlio/minio/s3torch)
│
├── — Utilities ————————————————————————————————————————————————————
│   run_cleanup.sh          delete all objects written by tests above
│   show_results.sh         print throughput summary from results/dlrm/
│
├── sweeps/                 NP and compute-time scaling studies (run after smoke tests)
│   sweep_dlrm_compute.sh   DLRM:      computation_time sweep at NP=1
│   sweep_dlrm_np.sh        DLRM:      NP scaling (1, 2, 4, 8)
│   sweep_flux.sh           Flux:      NP × read_threads scaling
│   sweep_retinanet_np.sh   RetinaNet: NP scaling (1, 2, 4)
│   sweep_unet3d_np.sh      UNet3D:    NP scaling (1, 2, 4)
│
└── old-archive/            deprecated scripts kept for reference — not maintained

Performance results and analysis live in docs/ (see Performance Results below).
```

### Four model types, one generator + one benchmark each

| Model | Format | Generator | Benchmark |
|---|---|---|---|
| **DLRM** | Parquet | *(inline in run_dlrm_bench.sh)* | `run_dlrm_bench.sh` |
| **Flux** | Parquet | *(inline in run_flux_bench.sh)* | `run_flux_bench.sh` |
| **RetinaNet** | JPEG | `gen_retinanet_jpeg.sh` | `test_retinanet.sh` |
| **UNet3D** | NPZ | `gen_unet3d_npz.sh` | `test_unet3d.sh` |

**Checkpointing** is a separate workflow (`run_checkpointing.sh`) — it tests LLaMA 3 8B
checkpoint write + read and is independent of the four model types above.

---

## Quick Start

```bash
# 1. Install dependencies
cd /path/to/mlp-storage
uv sync

# 2. Create .env with your credentials (see Credential Setup below)
cp .env.example .env

# 3a. DLRM or Flux — data is generated inline, just run the benchmark
NP=1 bash tests/object-store/run_dlrm_bench.sh
NP=1 bash tests/object-store/run_flux_bench.sh

# 3b. RetinaNet or UNet3D — generate data first, then benchmark
bash tests/object-store/gen_retinanet_jpeg.sh
bash tests/object-store/test_retinanet.sh

bash tests/object-store/gen_unet3d_npz.sh
bash tests/object-store/test_unet3d.sh

# 3c. Checkpointing
bash tests/object-store/run_checkpointing.sh
```

---

## Prerequisites

### 1 — Install dependencies

```bash
cd /path/to/mlp-storage
uv sync
```

### 2 — Create `.env`

Copy the example and fill in your values:

```bash
cp .env.example .env
# edit .env — never commit this file
```

`.env` must contain (at minimum):

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_ENDPOINT_URL=https://your-s3-host:9000   # or http:// for plain HTTP
AWS_REGION=us-east-1
STORAGE_LIBRARY=s3dlio                        # s3dlio | minio | s3torchconnector
```

`BUCKET` is optional — when unset each script derives a default from `STORAGE_LIBRARY`:

| `STORAGE_LIBRARY` | Auto-default `BUCKET` |
|---|---|
| `s3dlio` | `mlp-s3dlio` |
| `minio` | `mlp-minio` |
| `s3torchconnector` | `mlp-s3torch` |

Set `BUCKET` explicitly to use a different bucket name.

For HTTPS endpoints with a self-signed certificate, set `AWS_CA_BUNDLE` (used by
**s3dlio** and **minio**):

```bash
AWS_CA_BUNDLE=/path/to/your-cert.crt
```

> **`s3torchconnector` does NOT use `AWS_CA_BUNDLE`.** It reads from the system
> certificate store instead — see [TLS / HTTPS Setup](#tls--https-setup) below.

Shell environment variables already set take precedence over the `.env` file.

### 3 — Ensure the bucket exists

Create your bucket in MinIO (or your S3-compatible store) before running tests:

```bash
# Verify bucket is reachable
uv run python -c "import s3dlio; print(s3dlio.list('s3://your-bucket/', recursive=False))"
```

---

### `run_checkpointing.sh` — Checkpoint write + read

Runs a LLaMA 3 8B checkpoint cycle via `dlio_benchmark`:

1. **Write** — saves `CHECKPOINTS` checkpoint(s) to the object store
2. **Read** — restores each checkpoint back

All storage runtime parameters are injected as Hydra overrides at run time —
the YAML config contains only model/workload sizing.

```bash
cd /path/to/mlp-storage

# Default run: s3dlio, NP=4, 2 checkpoints — BUCKET auto-defaults to mlp-s3dlio
bash tests/object-store/run_checkpointing.sh

# Full llama3-8b run (8 MPI ranks ≈ 210 GB I/O per checkpoint cycle)
NP=8 bash tests/object-store/run_checkpointing.sh

# minio, 1 checkpoint — BUCKET auto-defaults to mlp-minio
STORAGE_LIBRARY=minio CHECKPOINTS=1 bash tests/object-store/run_checkpointing.sh

# s3torchconnector (NP>=4 required) — BUCKET auto-defaults to mlp-s3torch
STORAGE_LIBRARY=s3torchconnector bash tests/object-store/run_checkpointing.sh
```

**Runtime parameters:**

| Variable | Default | Description |
|---|---|---|
| `BUCKET` | auto-derived | `mlp-s3dlio` / `mlp-minio` / `mlp-s3torch` based on `STORAGE_LIBRARY`; set explicitly to override |
| `STORAGE_LIBRARY` | `s3dlio` | `s3dlio`, `minio`, or `s3torchconnector` |
| `NP` | `4` | MPI rank count — `4` is the recommended default; use `8` for full llama3-8b |
| `CHECKPOINTS` | `2` | Number of write + read cycles |
| `MODEL` | `llama3_8b_checkpoint` | DLIO workload config name |
| `S3_PROFILE` | *(unset)* | AWS credential profile for s3torchconnector (default: `mlp-minio`) |

> **`s3torchconnector` requires `NP>=4`:** At NP=1 the full ~105 GB checkpoint becomes a
> single object, exceeding the AWS CRT client's ~78 GB single-object limit — this
> **will fail**. The default `NP=4` already satisfies this requirement. s3dlio and
> minio are not affected.

---

### `run_cleanup.sh` — Cleanup

Deletes all objects written by the three test scripts above.  Supports dry-run
mode to preview what will be deleted before committing.

```bash
cd /path/to/mlp-storage

# Preview what would be deleted (no objects removed)
BUCKET=my-test-bucket DRY_RUN=1 bash tests/object-store/run_cleanup.sh

# Delete everything written by all tests
BUCKET=my-test-bucket bash tests/object-store/run_cleanup.sh

# Delete only training data (leave checkpoints)
BUCKET=my-test-bucket SKIP_CHECKPOINT=1 bash tests/object-store/run_cleanup.sh

# Delete only checkpoints written with minio
BUCKET=my-test-bucket STORAGE_LIBRARY=minio SKIP_TRAINING=1 SKIP_BENCH=1 \
    bash tests/object-store/run_cleanup.sh
```

**Runtime parameters:**

| Variable | Default | Description |
|---|---|---|
| `BUCKET` | auto-derived | `mlp-s3dlio` / `mlp-minio` / `mlp-s3torch` based on `STORAGE_LIBRARY`; set explicitly to override |
| `STORAGE_LIBRARY` | `s3dlio` | `s3dlio`, `minio`, or `s3torchconnector` — determines default `BUCKET` when unset |
| `MODEL` | `unet3d` | Model name (for training data prefix) |
| `DATA_DIR` | `test-run/` | Object prefix (must match datagen) |
| `BENCH_PREFIX` | `bench` | Prefix used by benchmark scripts |
| `SKIP_TRAINING` | `0` | Set to `1` to skip training data cleanup |
| `SKIP_CHECKPOINT` | `0` | Set to `1` to skip checkpoint cleanup |
| `SKIP_BENCH` | `0` | Set to `1` to skip benchmark object cleanup |
| `DRY_RUN` | `0` | Set to `1` to list deletions without executing |

---

## Credential Setup

Create `mlp-storage/.env` (never commit — it is already in `.gitignore`):

```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_ENDPOINT_URL=https://your-minio-host:9000
AWS_REGION=us-east-1
STORAGE_LIBRARY=s3dlio                # s3dlio | minio | s3torchconnector
# BUCKET=your-bucket                  # optional — auto-derived from STORAGE_LIBRARY if unset
```

See `.env.example` at the repo root for a fully annotated template.

---

## TLS / HTTPS Setup

The three storage libraries handle TLS certificates **differently** — this is the most
common source of connectivity failures when testing against a custom HTTPS endpoint.

### Certificate requirements (all libraries)

1. Generate the cert with `basicConstraints=CA:FALSE`  
   (Rust-based libraries use **rustls** and strictly enforce RFC 5280 — `CA:TRUE` is rejected)
2. The cert must include a `subjectAltName` (SAN) matching the server IP or hostname

### Per-library TLS configuration

| Library | TLS certificate source | Configuration |
|---|---|---|
| **s3dlio** | `AWS_CA_BUNDLE` env var | Set `AWS_CA_BUNDLE=/path/to/cert.crt` in `.env` |
| **minio** | `AWS_CA_BUNDLE` env var | Set `AWS_CA_BUNDLE=/path/to/cert.crt` in `.env` |
| **s3torchconnector** | **System certificate store** | Install cert system-wide — `AWS_CA_BUNDLE` is **ignored** |

> **`s3torchconnector` does NOT use `AWS_CA_BUNDLE`.**  
> The AWS CRT client reads only the **system certificate store**.  
> Setting `AWS_CA_BUNDLE` has no effect, regardless of its value.

### Installing the certificate for s3torchconnector

```bash
# Install the cert into the system CA directory
sudo cp /path/to/your-cert.crt /usr/local/share/ca-certificates/my-s3-server.crt

# Rebuild the system CA bundle
sudo update-ca-certificates
```

After `update-ca-certificates` completes, s3torchconnector will trust the certificate
without any further configuration.

### Verify TLS is working

```bash
# Should return HTTP 403 (AccessDenied) — means TLS handshake succeeded
curl -v https://your-minio-host:9000/
```

---

## Performance Results

Current benchmark results are in `docs/` — these are the authoritative numbers,
updated as new sweeps are run:

| Model | Results doc |
|---|---|
| DLRM | [docs/DLRM_NP_Scaling_Results.md](../../docs/DLRM_NP_Scaling_Results.md) |
| Flux | [docs/Flux_NP_ReadThreads_Scaling_Results.md](../../docs/Flux_NP_ReadThreads_Scaling_Results.md) |
| RetinaNet | [docs/RetinaNet_NP_Scaling_Results.md](../../docs/RetinaNet_NP_Scaling_Results.md) |
| UNet3D | [docs/UNet3D_NP_Scaling_Results.md](../../docs/UNet3D_NP_Scaling_Results.md) |

Sweep runs also write timestamped results to `results/<model>_np_sweep/<timestamp>/`.

---

## Adding More Libraries

Runtime parameters — library, bucket, endpoint, credentials — all flow from
environment variables. To test a new storage library:

1. Add it to `mlpstorage_py/storage/` and register it in `obj_store_lib.py`
2. Set `STORAGE_LIBRARY=<new-library>` in `.env`
3. Run the relevant benchmark script with `STORAGE_LIBRARY=<new-library>`

---

## Archived Tests

Older scripts and historical results are preserved in `tests/object-store/old-archive/`
for reference. They are **not maintained** and may not work with current code.

Archived materials include raw API experiments, older library-comparison helpers,
format serialization benchmarks, and generic multi-model wrappers. Prefer the
maintained scripts listed in [Structure](#structure) for current testing.
