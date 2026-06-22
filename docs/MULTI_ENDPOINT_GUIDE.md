# Multi-Endpoint Object Storage Guide

This guide explains how to spread object-storage traffic across multiple S3-compatible endpoints when running `mlpstorage` and DLIO workloads.

Start with [OBJECT_STORAGE_GUIDE.md](OBJECT_STORAGE_GUIDE.md) for single-endpoint setup. Use this guide when one endpoint is not enough, when each MPI rank should target a different endpoint, or when `s3dlio` should load-balance requests internally.

## When Multi-Endpoint Helps

Use multiple endpoints when:

- A single S3 endpoint cannot handle the required request rate.
- Storage servers are distributed across several hosts or network paths.
- MPI ranks should be distributed across independent S3 gateways.
- You want to compare endpoint selection strategies.

Multi-endpoint configuration is not data replication. All endpoints must expose the same bucket and object namespace, or the workload will see missing objects.

## Supported Modes

| Mode | Works With | How It Selects Endpoints | Best Use |
|---|---|---|---|
| `s3dlio` native multi-endpoint | `STORAGE_LIBRARY=s3dlio` | Internal round-robin or least-connections | Single process or MPI; highest-performance path. |
| MPI rank-based endpoint selection | `s3dlio`, `minio`, `s3torchconnector` | Rank N uses endpoint `N % endpoint_count` | Simple distributed runs with predictable assignment. |

## Environment Variables

Base object storage settings still come from [OBJECT_STORAGE_GUIDE.md](OBJECT_STORAGE_GUIDE.md):

```bash
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
AWS_REGION=us-east-1
BUCKET=mlperf-storage-bench
STORAGE_LIBRARY=s3dlio
STORAGE_URI_SCHEME=s3
```

Add one of the endpoint list forms below.

### Comma-Separated Endpoint List

```bash
S3_ENDPOINT_URIS=http://minio1:9000,http://minio2:9000,http://minio3:9000
```

### Template Expansion

```bash
S3_ENDPOINT_URIS=http://minio{1..4}:9000
```

This expands to:

```text
http://minio1:9000,http://minio2:9000,http://minio3:9000,http://minio4:9000
```

### Endpoint File

```bash
S3_ENDPOINT_FILE=/path/to/endpoints.txt
```

Example file:

```text
# endpoints.txt
http://minio1:9000
http://minio2:9000
http://minio3:9000
```

Blank lines and lines beginning with `#` are ignored.

### Strategy For s3dlio

```bash
S3_LOAD_BALANCE_STRATEGY=round_robin
```

Supported values depend on the installed `s3dlio` version. Use `round_robin` unless you have confirmed another strategy is available.

## Example: Three MinIO Endpoints

Start three MinIO servers or point to three existing S3-compatible gateways. Each endpoint must expose the same bucket and data.

```bash
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_REGION=us-east-1
export BUCKET=mlperf-storage-bench
export STORAGE_LIBRARY=s3dlio
export STORAGE_URI_SCHEME=s3
export S3_ENDPOINT_URIS=http://minio1:9000,http://minio2:9000,http://minio3:9000
export S3_LOAD_BALANCE_STRATEGY=round_robin
```

Generate data:

```bash
uv run mlpstorage closed training retinanet datagen object \
  --num-processes 6 \
  --data-dir retinanet \
  --results-dir /tmp/mlps-results \
  --allow-run-as-root \
  --skip-validation \
  --params dataset.num_files_train=3000 dataset.num_subfolders_train=30
```

Run training:

```bash
uv run mlpstorage closed training retinanet run object \
  --num-accelerators 6 \
  --accelerator-type b200 \
  --client-host-memory-in-gb 64 \
  --data-dir retinanet \
  --results-dir /tmp/mlps-results \
  --allow-run-as-root \
  --skip-validation \
  --params dataset.num_files_train=3000 dataset.num_subfolders_train=30
```

## MPI Rank-Based Assignment

When MPI is used, non-`s3dlio` libraries can still distribute traffic by selecting one endpoint per MPI rank:

```text
rank 0 -> endpoint 0
rank 1 -> endpoint 1
rank 2 -> endpoint 2
rank 3 -> endpoint 0
```

This behavior is deterministic and easy to reason about. It does not balance per-request load inside a rank.

Use this mode when testing `minio` or `s3torchconnector`, or when you want rank-to-endpoint affinity.

## s3dlio Native Load Balancing

When `STORAGE_LIBRARY=s3dlio`, the library can use all configured endpoints internally. This is the preferred mode when one rank should drive traffic to multiple endpoints.

Typical setup:

```bash
export STORAGE_LIBRARY=s3dlio
export S3_ENDPOINT_URIS=http://minio1:9000,http://minio2:9000,http://minio3:9000
export S3_LOAD_BALANCE_STRATEGY=round_robin
```

In this mode, each object operation can be assigned by the s3dlio load-balancing strategy.

## Data Placement Requirements

Multi-endpoint does not copy objects between endpoints.

Before training, make sure every endpoint can serve the same object namespace. Common patterns:

- All endpoints are gateways in front of the same object store.
- Data was generated through the same multi-endpoint configuration.
- Data was replicated externally by the storage system.

If each endpoint points at a separate, non-replicated bucket, training may fail when a rank requests an object that only exists behind another endpoint.

## Validation Commands

Check each endpoint independently:

```bash
for endpoint in http://minio1:9000 http://minio2:9000 http://minio3:9000; do
  AWS_ENDPOINT_URL=$endpoint aws --endpoint-url "$endpoint" s3 ls "s3://$BUCKET" >/dev/null && \
    echo "OK $endpoint"
done
```

Check that `mlpstorage` sees the endpoint list:

```bash
env | grep -E 'S3_ENDPOINT_URIS|S3_ENDPOINT_FILE|S3_LOAD_BALANCE_STRATEGY'
```

For a tiny functional test, see [OBJECT_STORAGE_TESTING.md](OBJECT_STORAGE_TESTING.md).

## Tuning Guidelines

- Start with one endpoint per storage server or S3 gateway.
- Keep MPI process count at or below the total client CPU and memory capacity.
- Increase `dataset.num_subfolders_train` for very large file counts.
- Keep `dataset.skip_listing=True` for DLIO-generated datasets unless you need full listing validation.
- For MinIO or other S3-compatible endpoints, verify whether path-style addressing is required. `mlpstorage` enables path-style addressing automatically when `AWS_ENDPOINT_URL` is set.

## Troubleshooting

### All ranks hit one endpoint

Check that `S3_ENDPOINT_URIS` is exported in the parent shell before `mpirun` starts. MPI child processes inherit environment variables from the launcher.

### Some ranks report missing objects

The endpoints probably do not expose identical object namespaces. Verify the same bucket and key exists through every endpoint.

### Endpoint works locally but fails under MPI

Use hostnames or IP addresses reachable from all client hosts. `127.0.0.1` means each MPI rank's own host, not the launcher host.

### Poor distribution

For `s3dlio`, use `S3_LOAD_BALANCE_STRATEGY=round_robin` as the baseline. For MPI rank-based assignment, make sure the MPI rank count is at least the endpoint count.

## See Also

- [OBJECT_STORAGE_GUIDE.md](OBJECT_STORAGE_GUIDE.md) for single-endpoint setup.
- [OBJECT_STORAGE_TESTING.md](OBJECT_STORAGE_TESTING.md) for smoke tests.
- [OBJECT_STORAGE_GUIDE.md](OBJECT_STORAGE_GUIDE.md#supported-libraries-and-selection) for library comparison details.
