# Object Storage Configuration Guide for mlp-storage / DLIO

This document describes every setting required to run `mlpstorage` training
benchmarks against S3-compatible object storage using `s3dlio` as the storage
library backend.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Variables (.env)](#environment-variables)
4. [CLI Flags](#cli-flags)
5. [Auto-Injected DLIO Parameters](#auto-injected-dlio-parameters)
6. [MPI Workaround (single-node)](#mpi-workaround-single-node)
7. [Complete Example Commands](#complete-example-commands)
8. [Verified Run Log](#verified-run-log)
9. [Troubleshooting](#troubleshooting)

---

## Overview

mlp-storage wraps `dlio_benchmark` and provides a `--file` / `--object` flag to
switch between local filesystem storage and S3-compatible object storage.

When `--object` is passed:
- mlpstorage reads credentials and endpoint from a `.env` file in the working
  directory (or the script's parent directory).
- Four DLIO parameters are automatically injected (no `--params` needed for
  them).
- The `--data-dir` argument becomes the S3 **key prefix** (not a filesystem
  path).

---

## Prerequisites

### 1. python-dotenv installed

```bash
cd mlp-storage
uv add python-dotenv
```

(Already present in `pyproject.toml` as of April 2026.)

### 2. s3dlio Python library installed

```bash
uv add s3dlio
```

`s3dlio` is the default `STORAGE_LIBRARY`. It reads credentials from
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_ENDPOINT_URL`
environment variables.

### 3. An S3-compatible endpoint

For local testing, `s3-ultra` (in this workspace) is the recommended fake S3 server:

```bash
# Start s3-ultra on port 9101 (plain HTTP, h2c + HTTP/1.1)
/path/to/s3-ultra serve --port 9101 --db-path /tmp/s3-ultra-mlp-test &

# Create the bucket (no-sign-request because s3-ultra has no authentication)
aws --endpoint-url http://127.0.0.1:9101 --no-sign-request s3 mb s3://<bucket-name>
```

> **Note**: `s3-ultra` does **not** use authentication (any `AWS_*` credentials
> you set are ignored by the server). The `--no-sign-request` flag must be
> used with the AWS CLI when creating buckets against s3-ultra.

---

## Environment Variables

Create a `.env` file in the working directory (the directory you run
`uv run mlpstorage` from, typically `mlp-storage/`):

```dotenv
# .env — object storage configuration for mlp-storage

# S3 endpoint URL (required for non-AWS targets)
AWS_ENDPOINT_URL=http://127.0.0.1:9101

# Credentials (can be dummy values for s3-ultra / fake servers)
AWS_ACCESS_KEY_ID=testkey
AWS_SECRET_ACCESS_KEY=testsecret

# Region (required by s3dlio; use "us-east-1" for local servers)
AWS_REGION=us-east-1

# Storage library to use inside dlio_benchmark
# Options: s3dlio (recommended), minio, s3torchconnector
STORAGE_LIBRARY=s3dlio

# S3 bucket name (required for --object mode)
BUCKET=mlp-retinanet
```

### Variable reference

| Variable | Required | Description |
|---|---|---|
| `AWS_ENDPOINT_URL` | Yes (for non-AWS) | Full URL of the S3 endpoint, e.g. `http://127.0.0.1:9101` |
| `AWS_ACCESS_KEY_ID` | Yes | Access key (can be `testkey` for fake servers) |
| `AWS_SECRET_ACCESS_KEY` | Yes | Secret key (can be `testsecret` for fake servers) |
| `AWS_REGION` | Recommended | Region string; defaults to `us-east-1` in s3dlio |
| `STORAGE_LIBRARY` | No | Storage backend inside DLIO. Default: `s3dlio` |
| `BUCKET` | Yes | S3 bucket name. Used as `storage.storage_root` in DLIO |
| `S3DLIO_RT_THREADS` | No | Override Tokio runtime threads for s3dlio. Auto-set to `1.5×write_threads` if not set. |

---

## CLI Flags

### `--object`

Enables object storage mode. Triggers `_apply_object_storage_params()` which:
1. Loads `.env` (via python-dotenv)
2. Injects DLIO storage parameters (see next section)
3. Skips local filesystem directory creation

```bash
uv run mlpstorage training datagen \
    --model retinanet \
    --num-processes 4 \
    --open --object \
    --data-dir retinanet \          # S3 key prefix, NOT a filesystem path
    --allow-run-as-root \
    --skip-validation \
    --params dataset.num_files_train=250000
```

### `--data-dir` in object mode

In `--object` mode, `--data-dir` specifies the **S3 key prefix** (folder inside
the bucket), not a local filesystem path. Example: `--data-dir retinanet`
stores objects at `s3://<BUCKET>/retinanet/`.

### `--file`

Enables local filesystem mode. `.env` is still loaded but S3 params are not
injected. `--data-dir` must point to an existing local directory.

---

## Auto-Injected DLIO Parameters

When `--object` is used, the following DLIO `++workload.*` overrides are
automatically injected (you do **not** need to pass them via `--params`):

| DLIO Parameter | Value | Notes |
|---|---|---|
| `storage.storage_type` | `s3` | Tells DLIO to use S3 backend |
| `storage.storage_root` | `$BUCKET` | Bucket name from `.env` |
| `storage.storage_options.storage_library` | `$STORAGE_LIBRARY` | Library (default: `s3dlio`) |
| `storage.s3_force_path_style` | `true` | Required for non-AWS endpoints (path-style URLs) |

> **Note**: These are only injected if not already present in `params_dict`
> (existing `--params` overrides take precedence).

---

## MPI Workaround (single-node)

On a single machine, OpenMPI's default shared-memory transport (`vader` BTL) can
produce **segfaults** during `MPI_Barrier` when running with `-n > 1`.

**Fix**: Add `--mpi-params "--mca btl tcp,self"` to your command:

```bash
uv run mlpstorage training run \
    --model retinanet \
    --num-accelerators 4 --accelerator-type b200 \
    --client-host-memory-in-gb 47 \
    --open --file \
    --data-dir /mnt/nvme_data/retinanet \
    --allow-run-as-root --skip-validation \
    --mpi-params "--mca btl tcp,self" \          # <-- required on single node
    --params dataset.num_files_train=250000
```

This passes `--mca btl tcp,self` to `mpirun`, disabling the VADER BTL and
falling back to TCP loopback transport.

---

## Complete Example Commands

### File storage — datagen

```bash
cd /path/to/mlp-storage

uv run mlpstorage training datagen \
    --model retinanet \
    --num-processes 4 \
    --open --file \
    --data-dir /mnt/nvme_data/retinanet \
    --allow-run-as-root --skip-validation \
    --params dataset.num_files_train=250000
```

### File storage — training run

```bash
uv run mlpstorage training run \
    --model retinanet \
    --num-accelerators 4 --accelerator-type b200 \
    --client-host-memory-in-gb 47 \
    --open --file \
    --data-dir /mnt/nvme_data/retinanet \
    --allow-run-as-root --skip-validation \
    --mpi-params "--mca btl tcp,self" \
    --params dataset.num_files_train=250000
```

### Object storage — datagen

```bash
# Ensure s3-ultra is running and bucket exists:
#   /path/to/s3-ultra serve --port 9101 --db-path /tmp/s3-ultra-mlp-test &
#   aws --endpoint-url http://127.0.0.1:9101 --no-sign-request s3 mb s3://mlp-retinanet

uv run mlpstorage training datagen \
    --model retinanet \
    --num-processes 4 \
    --open --object \
    --data-dir retinanet \
    --allow-run-as-root --skip-validation \
    --params dataset.num_files_train=250000
```

### Object storage — training run

```bash
uv run mlpstorage training run \
    --model retinanet \
    --num-accelerators 4 --accelerator-type b200 \
    --client-host-memory-in-gb 47 \
    --open --object \
    --data-dir retinanet \
    --allow-run-as-root --skip-validation \
    --mpi-params "--mca btl tcp,self" \
    --params dataset.num_files_train=250000
```

---

## Verified Run Log

| Date | Mode | Command | Outcome |
|---|---|---|---|
| 2026-04-26 | file datagen | NP=4, 250k files → `/mnt/nvme_data/retinanet` | ✅ Exit 0, 67s |
| 2026-04-26 | file training | NP=4, b200, 47GB RAM, `--mca btl tcp,self` | ✅ Exit 0 (see below) |
| 2026-04-26 | object datagen | NP=4, 250k files → `s3://mlp-retinanet/retinanet` | (pending) |
| 2026-04-26 | object training | NP=4, b200, 47GB RAM, `--mca btl tcp,self` | (pending) |

> **First attempt at file training** (without `--mca btl tcp,self`) crashed with
> SIGSEGV in `mca_btl_vader_poll_handle_frag` on rank 3. Fixed by adding
> `--mpi-params "--mca btl tcp,self"`.

---

## Troubleshooting

### `BUCKET environment variable is required for --object mode`
The `.env` file was not found or `BUCKET` is not set. Ensure `.env` exists in
the current working directory and contains `BUCKET=<your-bucket-name>`.

### `NotImplemented: This service has no authentication provider` (s3-ultra)
s3-ultra does not support authentication. Use `--no-sign-request` with the AWS
CLI when creating buckets. Credentials in `.env` (`testkey`/`testsecret`) are
passed to s3dlio which sends them in request headers — s3-ultra ignores them
without error during normal operations.

### Segfault in `mca_btl_vader` (SIGSEGV on MPI_Barrier)
OpenMPI's shared-memory transport crashes on some single-node configurations.
Add `--mpi-params "--mca btl tcp,self"` to all `training run` commands.

### `Insufficient number of training files (Expected: >= 781958, Actual: 250000)`
This is an expected **INVALID** warning for non-standard file counts. The
benchmark still runs successfully. The warning only means the results cannot be
used for official MLPerf Storage submission. Use `--skip-validation` to
suppress the hard stop.

### `storage_options` shows S3 credentials even in `--file` mode
The retinanet workload YAML config includes S3 storage_options for portability.
They are harmless when `storage_type = local_fs` — DLIO ignores them.
