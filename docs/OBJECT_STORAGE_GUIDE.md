# Object Storage Guide

This is the primary guide for running `mlpstorage` training workloads against S3-compatible object storage. It covers the supported storage libraries, required environment variables, current CLI shape, and a small MinIO-based quick start.

For load balancing across multiple endpoints, see [MULTI_ENDPOINT_GUIDE.md](MULTI_ENDPOINT_GUIDE.md). For smoke tests and validation commands, see [OBJECT_STORAGE_TESTING.md](OBJECT_STORAGE_TESTING.md).

## Supported Libraries And Selection

`mlpstorage` uses DLIO for training I/O. Object storage support is selected at runtime and currently supports:

| Library | Best Starting Point | Protocols | Frameworks | Multi-Endpoint | Notes |
|---|---|---|---|---|---|
| `s3dlio` | Recommended default | S3-compatible, Azure, GCS, `file://`, `direct://` | PyTorch, TensorFlow | Native load balancing | Rust/Tokio implementation; best first choice for datagen, training, checkpoints, byte-range reads, and multi-protocol tests. |
| `minio` | S3-compatible SDK comparison | S3-compatible | PyTorch, TensorFlow | MPI rank selection | Uses the MinIO Python SDK; useful for validating MinIO-specific behavior and comparing against `s3dlio`. |
| `s3torchconnector` | AWS PyTorch connector comparison | S3-compatible | PyTorch only | MPI rank selection | AWS connector path; useful for PyTorch-only comparison runs. |

Use `s3dlio` first unless you specifically need to compare library behavior.

### Capability Matrix

| Capability | `s3dlio` | `minio` | `s3torchconnector` |
|---|---|---|---|
| Training reads | Yes | Yes | PyTorch only |
| Datagen writes | Yes | Usually yes | Use `s3dlio` if datagen is unsupported in your workload |
| Checkpoint read/write | Yes | Yes | Yes, but avoid single objects above the AWS CRT size limit |
| Byte-range reads | Yes | Yes | Yes |
| Azure/GCS/file/direct URI schemes | Yes | No | No |
| Native multi-endpoint load balancing | Yes | No | No |
| MPI rank-based endpoint selection | Yes | Yes | Yes |

### Runtime Library Selection

Switch libraries with `STORAGE_LIBRARY`:

```bash
export STORAGE_LIBRARY=s3dlio
export STORAGE_LIBRARY=minio
export STORAGE_LIBRARY=s3torchconnector
```

The same `mlpstorage ... object` command line is used for all three libraries. Runtime credentials, endpoint, bucket, URI scheme, and multi-endpoint settings come from environment variables or `.env`.

TLS handling differs by library. `s3dlio` and `minio` can use `AWS_CA_BUNDLE`; `s3torchconnector` reads the system certificate store instead. See [OBJECT_STORAGE_TESTING.md](OBJECT_STORAGE_TESTING.md) and [tests/object-store/README.md](../tests/object-store/README.md) before running HTTPS tests against a custom endpoint.

## Install Dependencies

From the repository root:

```bash
uv sync
```

Optional object storage libraries are installed from the project dependency set. If you are developing against a local dependency, update `pyproject.toml` and regenerate `uv.lock` with `uv lock`; do not use `pip` directly inside this project.

## Start A Local MinIO Endpoint

MinIO is a portable local S3-compatible service that works well for examples and developer smoke tests.

```bash
docker run --rm \
  --name mlps-minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  quay.io/minio/minio server /data --console-address ":9001"
```

In another terminal, create a bucket:

```bash
aws --endpoint-url http://127.0.0.1:9000 \
  s3 mb s3://mlperf-storage-bench
```

If your `aws` CLI does not already have credentials, export the same MinIO credentials shown below before creating the bucket.

## Environment Variables

Create `.env` in the repository root, or export these variables in the shell before running `mlpstorage`:

```bash
AWS_ENDPOINT_URL=http://127.0.0.1:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
AWS_REGION=us-east-1

# Bucket or top-level container name. Do not include s3:// here.
BUCKET=mlperf-storage-bench

# Object library used by dlio_benchmark.
STORAGE_LIBRARY=s3dlio

# URI scheme used by s3dlio when it constructs object URIs.
STORAGE_URI_SCHEME=s3
```

### Variable Reference

| Variable | Required | Example | Purpose |
|---|---:|---|---|
| `AWS_ENDPOINT_URL` | For non-AWS S3 targets | `http://127.0.0.1:9000` | S3 API endpoint. |
| `AWS_ACCESS_KEY_ID` | Yes | `minioadmin` | Access key. |
| `AWS_SECRET_ACCESS_KEY` | Yes | `minioadmin` | Secret key. |
| `AWS_REGION` | Usually | `us-east-1` | Region for S3 clients. |
| `BUCKET` | Yes for `object` mode | `mlperf-storage-bench` | Bucket/container name passed as DLIO `storage.storage_root`. |
| `STORAGE_LIBRARY` | No | `s3dlio` | One of `s3dlio`, `minio`, `s3torchconnector`. Defaults to `s3dlio`. |
| `STORAGE_URI_SCHEME` | No | `s3` | URI scheme for `s3dlio`; examples include `s3`, `az`, `gs`, `file`, `direct`. Defaults to `s3`. |

## CLI Shape

The current training CLI shape is:

```text
mlpstorage <closed|open|whatif> training <model> <datasize|datagen|run|configview> <file|object> [options]
```

Use the positional `object` selector for S3-compatible storage:

```bash
uv run mlpstorage closed training retinanet datagen object ...
uv run mlpstorage closed training retinanet run object ...
```

Use `file` for local/POSIX filesystem storage.

## `--data-dir` In Object Mode

In `object` mode, `--data-dir` is the dataset prefix inside the bucket. The bucket itself comes from `BUCKET`.

Recommended form:

```bash
--data-dir retinanet
```

This stores and reads objects under:

```text
s3://$BUCKET/retinanet/...
```

A full URI such as `s3://bucket/retinanet` is accepted by the modern `s3dlio` object path, but the prefix form is clearer and avoids mixing bucket selection between CLI flags and environment configuration.

## Object Storage Datagen

Small example suitable for a MinIO smoke test:

```bash
uv run mlpstorage closed training retinanet datagen object \
  --num-processes 4 \
  --data-dir retinanet \
  --results-dir /tmp/mlps-results \
  --allow-run-as-root \
  --skip-validation \
  --params dataset.num_files_train=1024 dataset.num_subfolders_train=16
```

What `mlpstorage` injects for DLIO:

```text
++workload.storage.storage_type=s3
++workload.storage.storage_root=$BUCKET
++workload.storage.storage_options.storage_library=$STORAGE_LIBRARY
++workload.storage.storage_options.uri_scheme=$STORAGE_URI_SCHEME
++workload.dataset.data_folder=retinanet
```

For local S3-compatible endpoints, path-style addressing is enabled automatically when `AWS_ENDPOINT_URL` is set.

## Object Storage Training Run

After datagen:

```bash
uv run mlpstorage closed training retinanet run object \
  --num-accelerators 4 \
  --accelerator-type b200 \
  --client-host-memory-in-gb 64 \
  --data-dir retinanet \
  --results-dir /tmp/mlps-results \
  --allow-run-as-root \
  --skip-validation \
  --params dataset.num_files_train=1024 dataset.num_subfolders_train=16
```

Use the same `--data-dir` prefix for `datagen` and `run`.

## Switching Libraries For Comparison Runs

Use the same object prefix and change only `STORAGE_LIBRARY` when comparing behavior:

```bash
export STORAGE_LIBRARY=s3dlio
# run datagen + run

export STORAGE_LIBRARY=minio
# run the same datagen + run pair

export STORAGE_LIBRARY=s3torchconnector
# run the same training run; datagen can stay on s3dlio if needed
```

Notes:

- Keep `BUCKET`, `AWS_ENDPOINT_URL`, and `--data-dir` unchanged so each library reads the same objects.
- Use `s3dlio` as the baseline because it supports the widest protocol set and native multi-endpoint load balancing.
- Use [OBJECT_STORAGE_TESTING.md](OBJECT_STORAGE_TESTING.md) for small smoke tests before running large benchmark sweeps.

## Multi-Protocol With s3dlio

`s3dlio` can use multiple URI schemes. Set `STORAGE_URI_SCHEME` and `BUCKET` appropriately:

| Target | `STORAGE_URI_SCHEME` | `BUCKET` example |
|---|---|---|
| S3-compatible | `s3` | `mlperf-storage-bench` |
| Azure Blob | `az` | `container-name` |
| Google Cloud Storage | `gs` | `bucket-name` |
| Buffered file path | `file` | `/mnt/data` |
| O_DIRECT file path | `direct` | `/mnt/data` |

For cloud providers, also configure the provider-specific credentials required by `s3dlio`.

## Performance Defaults

For datasets generated by DLIO, `mlpstorage` auto-enables:

```text
++workload.dataset.skip_listing=True
++workload.dataset.listing_validation_interval=<adaptive interval>
```

This avoids expensive full-prefix listings against large object stores while still sampling generated files before training starts.

## Troubleshooting

### `BUCKET environment variable is required for object mode`

Set `BUCKET` in `.env` or export it before running the command. Use only the bucket/container name, not `s3://bucket`.

### Connection refused or timeout

Check that the endpoint is reachable from every MPI rank:

```bash
curl -I http://127.0.0.1:9000/minio/health/live
```

For multi-node runs, do not use `127.0.0.1` unless every rank has its own local endpoint. Use a hostname or IP address reachable from all client nodes.

### Authentication errors

Verify the endpoint, credentials, and bucket:

```bash
aws --endpoint-url "$AWS_ENDPOINT_URL" s3 ls "s3://$BUCKET"
```

### MPI shared-memory failures in containers

Try TCP transport:

```bash
--mpi-btl tcp
```

or pass OpenMPI MCA options through `--mpi-params`.

## See Also

- [MULTI_ENDPOINT_GUIDE.md](MULTI_ENDPOINT_GUIDE.md) for multi-endpoint object storage.
- [OBJECT_STORAGE_TESTING.md](OBJECT_STORAGE_TESTING.md) for smoke tests and validation commands.
- [../tests/object-store/README.md](../tests/object-store/README.md) for maintained object-storage benchmark scripts.
