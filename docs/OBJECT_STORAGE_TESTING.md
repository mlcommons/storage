# Object Storage Testing Guide

This guide contains small, repeatable tests for validating object-storage support in `mlpstorage`. Use it after completing [OBJECT_STORAGE_GUIDE.md](OBJECT_STORAGE_GUIDE.md).

The examples use MinIO because it is widely available and S3-compatible. Replace the endpoint, credentials, and bucket with your own storage service when testing a real system.

## Test Setup

Start or identify an S3-compatible endpoint, then export:

```bash
export AWS_ENDPOINT_URL=http://127.0.0.1:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_REGION=us-east-1
export BUCKET=mlperf-storage-bench
export STORAGE_LIBRARY=s3dlio
export STORAGE_URI_SCHEME=s3
```

Create the bucket if needed:

```bash
aws --endpoint-url "$AWS_ENDPOINT_URL" s3 mb "s3://$BUCKET"
```

Verify credentials:

```bash
aws --endpoint-url "$AWS_ENDPOINT_URL" s3 ls "s3://$BUCKET"
```

## Python Library Smoke Tests

### s3dlio

```bash
uv run python - <<'PY'
import os
import s3dlio
uri = f"s3://{os.environ['BUCKET']}/smoke/s3dlio.bin"
s3dlio.put_bytes(uri, b"mlp-storage smoke test")
data = s3dlio.get_bytes(uri)
assert data == b"mlp-storage smoke test"
print("s3dlio OK", uri)
PY
```

### minio

```bash
uv run python - <<'PY'
import io
import os
from urllib.parse import urlparse
from minio import Minio

endpoint = urlparse(os.environ['AWS_ENDPOINT_URL'])
client = Minio(
    endpoint.netloc,
    access_key=os.environ['AWS_ACCESS_KEY_ID'],
    secret_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    secure=endpoint.scheme == 'https',
)
payload = b"mlp-storage smoke test"
client.put_object(os.environ['BUCKET'], 'smoke/minio.bin', io.BytesIO(payload), len(payload))
obj = client.get_object(os.environ['BUCKET'], 'smoke/minio.bin')
try:
    assert obj.read() == payload
finally:
    obj.close()
    obj.release_conn()
print("minio OK")
PY
```

### s3torchconnector

`s3torchconnector` is PyTorch-oriented. Use the training smoke test below after setting:

```bash
export STORAGE_LIBRARY=s3torchconnector
```

## mlpstorage Parser Smoke Test

This verifies the current CLI grammar without running DLIO:

```bash
uv run python - <<'PY'
from unittest.mock import patch
from mlpstorage_py.cli_parser import parse_arguments

argv = [
    'mlpstorage', 'closed', 'training', 'retinanet', 'datagen', 'object',
    '--num-processes', '2',
    '--data-dir', 'retinanet',
    '--results-dir', '/tmp/mlps-results',
    '--skip-validation',
]
with patch('sys.argv', argv):
    args = parse_arguments()
assert args.data_access_protocol == 'object'
assert args.data_dir == 'retinanet'
print('parser OK')
PY
```

## Small End-To-End Datagen Test

Generate a tiny object-storage dataset:

```bash
uv run mlpstorage closed training retinanet datagen object \
  --num-processes 2 \
  --data-dir retinanet-smoke \
  --results-dir /tmp/mlps-results \
  --allow-run-as-root \
  --skip-validation \
  --params dataset.num_files_train=32 dataset.num_subfolders_train=4
```

Check objects were written:

```bash
aws --endpoint-url "$AWS_ENDPOINT_URL" \
  s3 ls "s3://$BUCKET/retinanet-smoke/retinanet/train/" --recursive | head
```

## Small Training Run Test

Run against the same prefix:

```bash
uv run mlpstorage closed training retinanet run object \
  --num-accelerators 2 \
  --accelerator-type b200 \
  --client-host-memory-in-gb 64 \
  --data-dir retinanet-smoke \
  --results-dir /tmp/mlps-results \
  --allow-run-as-root \
  --skip-validation \
  --params dataset.num_files_train=32 dataset.num_subfolders_train=4 train.epochs=1
```

Use the same `--data-dir` value for datagen and run.

## Test Each Storage Library

Run the same small datagen/run pair with different libraries:

```bash
export STORAGE_LIBRARY=s3dlio
# run datagen + run

export STORAGE_LIBRARY=minio
# run datagen + run

export STORAGE_LIBRARY=s3torchconnector
# run training run; datagen can stay on s3dlio if needed
```

Notes:

- `s3dlio` is the recommended default for both datagen and run.
- `s3torchconnector` is PyTorch-only.
- If a library does not support datagen in your configuration, generate data with `s3dlio`, then switch libraries for `run`.

## Multi-Endpoint Smoke Test

For three endpoints exposing the same bucket:

```bash
export S3_ENDPOINT_URIS=http://minio1:9000,http://minio2:9000,http://minio3:9000
export S3_LOAD_BALANCE_STRATEGY=round_robin
```

Then repeat the small datagen/run test above. See [MULTI_ENDPOINT_GUIDE.md](MULTI_ENDPOINT_GUIDE.md) for endpoint selection details and limitations.

## Unit Tests Useful For Object Storage Changes

Parser and object-storage parameter tests:

```bash
uv run python -m pytest \
  tests/unit/test_cli.py \
  tests/unit/test_cli_parser.py \
  tests/unit/test_dlio_object_storage.py \
  tests/unit/test_datagen_command_generation.py \
  -q
```

DLIO fast CI tests live in the `dlio_benchmark` repository when validating changes there:

```bash
cd ../dlio_benchmark
uv run python -m pytest tests/test_fast_ci.py -q
```

## Troubleshooting

### Bucket does not exist

Create it with the AWS CLI or your object-store console:

```bash
aws --endpoint-url "$AWS_ENDPOINT_URL" s3 mb "s3://$BUCKET"
```

### Objects written to an unexpected prefix

Remember that `mlpstorage` appends the model name if `--data-dir` does not already end with the model name. For example:

```text
--data-dir retinanet-smoke
```

becomes:

```text
retinanet-smoke/retinanet
```

### Local directory creation appears in object mode

Object mode should skip local data directory creation. Confirm the command ends with the positional `object`, not `file`.

### MPI cannot reach the endpoint

Use an endpoint hostname or IP address reachable from every MPI client host. Avoid `127.0.0.1` for multi-node runs unless every node runs its own local endpoint.
