# DLIO Benchmark Storage Patches

This directory contains modified files from the `dlio_benchmark` package to support multi-library S3 storage.

## Overview

These patches enable DLIO to use multiple S3 client libraries (s3torchconnector, minio, s3dlio) through a unified URI-based interface.

## Modified Files

### 1. storage_factory.py
**Changes**: Added implementation selector via config parameter
- Reads `storage.storage_options.storage_library` from YAML config
- Routes to MLP (multi-library) or dpsi (bucket+key) storage handlers
- Default: MLP implementation
- Debug output shows which implementation is selected

### 2. storage_handler.py
**Changes**: Added logger attribute for dpsi compatibility
- Line 28: Added `self.logger = self._args.logger`
- Allows storage handlers to access logger from args
- Required for dpsi implementation compatibility

### 3. s3_torch_storage.py (MLP Implementation - 380 lines)
**Architecture**: URI-based with multi-library support

**Key Features**:
- **URI-based**: Uses full `s3://bucket/path` URIs (not bucket+key separation)
- **Multi-library**: s3torchconnector, minio, s3dlio via config parameter
- **s3dlio integration**: Native API (put_bytes, get_bytes, list)
- **Zero-dependency fallback**: Uses s3torchconnector if others unavailable
- **Configuration**: `storage.storage_options.storage_library` in YAML

**Modified Methods**:
- Lines 173-178: s3dlio client initialization
- Lines 252-263: `get_uri()` - Constructs full s3://bucket/path URIs
- Lines 318-334: `put_data()` - Conditional on storage_library selection
- Lines 336-353: `get_data()` - Direct s3dlio.get_bytes() calls
- Lines 356-395: `list_objects()` - Native s3dlio.list() API

## Installation

These patches are applied to a local editable installation of dlio_benchmark:

```bash
# From mlp-storage directory
cd /home/eval/Documents/Code/mlp-storage
source .venv/bin/activate

# Clone dlio_benchmark (if not already done)
git clone https://github.com/russfellows/dlio_benchmark.git
cd dlio_benchmark
pip install -e .

# Apply patches
cd /home/eval/Documents/Code/mlp-storage
cp patches/storage_factory.py dlio_benchmark/dlio_benchmark/storage/
cp patches/storage_handler.py dlio_benchmark/dlio_benchmark/storage/
cp patches/s3_torch_storage.py dlio_benchmark/dlio_benchmark/storage/
```

## Configuration

Example YAML config:

```yaml
storage:
  storage_type: s3_torch
  storage_root: s3://your-bucket
  storage_options:
    storage_library: s3dlio  # or minio, or s3torchconnector
```

## Testing

See [../tests/README.md](../tests/README.md) for test scripts validating all three storage libraries:
- `test_mlp_s3torch.sh` - s3torchconnector (AWS reference)
- `test_mlp_minio.sh` - minio Python client
- `test_mlp_s3dlio.sh` - s3dlio high-performance library

## Performance (Latest Results)

All tests with MinIO endpoint, 3 files × 5 samples, 65KB records:
- mlp-s3torch: ~30 seconds
- mlp-minio: ~15 seconds (fastest)
- mlp-s3dlio: ~31 seconds

## Related Changes

- **PR #232 fix**: [../mlpstorage/benchmarks/dlio.py](../mlpstorage/benchmarks/dlio.py) line 147
  - Added `and self.args.data_dir` check for empty data_dir handling
- **s3dlio compat layer**: Fixed in s3dlio v0.9.40 (`put_bytes` instead of `put`)

## dpsi Implementation (Reference)

The dpsi implementation uses bucket+key separation and is maintained separately for comparison:
- Location: `/home/eval/Documents/Code/mlp-storage-dpsi`
- Files: `s3_storage_dpsi.py`, `s3_torch_storage_dpsi.py`
- Lines: 145 (vs 380 for MLP)
- Libraries: s3torchconnector only

## Future Options

These patches support the current approach (separate dlio_benchmark repo with manual patching). Future alternatives being considered:
- Git submodule for dlio_benchmark
- Full fork of dlio_benchmark with integrated changes
- Upstream PR to dlio_benchmark project
