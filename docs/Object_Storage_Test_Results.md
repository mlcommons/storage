# Object Storage Library Test Results

This file records measured test results for each object storage library supported
by mlp-storage. For instructions on how to run the tests, see
[Object_Storage_Test_Guide.md](Object_Storage_Test_Guide.md).

---

## Test Matrix

| Library | PyTorch + NPZ | TensorFlow + TFRecord | S3 protocol | Azure | GCS | Local (`file://`) |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| **s3dlio** | ✅ Tested | ✅ Tested | — pending | — pending | — pending | ✅ Tested |
| **minio** | — pending | — pending | — pending | n/a | n/a | n/a |
| **s3torchconnector** | — pending | n/a (PyTorch only) | — pending | n/a | n/a | n/a |

---

## s3dlio — Local Filesystem Tests (February 7, 2026)

**Test environment:**
- Protocol: `file://` (local filesystem)
- Backend: s3dlio via `storage_type: s3dlio`, `storage_root: file://...`
- Platform: Single node
- Test scope: Data generation → data reading (complete round-trip)

### Test 1: PyTorch + s3dlio + NPZ

**Phase 1: Data Generation**

```bash
mlpstorage training datagen \
  --model unet3d \
  --num-processes 1 \
  --data-dir /mnt/scratch/unet3d-test \
  --params dataset.num_files_train=10 \
  --params dataset.num_samples_per_file=1 \
  --params dataset.record_length_bytes=10485760
```

Results:
- **Status**: ✅ SUCCESS
- **Duration**: 3.5 seconds
- **Files created**: 10 NPZ files
- **Total size**: 369 MB (files vary from 3.6 KB to 178 MB due to record_length stdev)
- **Location**: `/mnt/scratch/unet3d-test/unet3d/train/`

File listing:
```
img_00_of_10.npz  178M
img_01_of_10.npz  3.6K
img_02_of_10.npz   11K
img_03_of_10.npz   26M
img_04_of_10.npz  4.4M
img_05_of_10.npz  119M
img_06_of_10.npz   15K
img_07_of_10.npz   43M
img_08_of_10.npz  5.1K
img_09_of_10.npz   19K
```

**Phase 2: Data Reading (PyTorch + s3dlio)**

```bash
mlpstorage training run \
  --model unet3d \
  --accelerator-type h100 \
  --num-accelerators 1 \
  --client-host-memory-in-gb 16 \
  --data-dir /mnt/scratch/unet3d-test \
  --params reader.data_loader=pytorch \
  --params reader.storage_library=s3dlio \
  --params reader.storage_root=file:///mnt/scratch/unet3d-test/unet3d \
  --params dataset.num_files_train=10 \
  --params dataset.num_samples_per_file=1 \
  --params reader.batch_size=2 \
  --params train.epochs=1 \
  --params train.computation_time=0.001
```

Configuration overrides confirmed in results:
```yaml
# /tmp/mlperf_storage_results/.../overrides.yaml
- ++workload.reader.data_loader=pytorch
- ++workload.reader.storage_library=s3dlio
- ++workload.reader.storage_root=file:///mnt/scratch/unet3d-test/unet3d
```

Results:
- **Status**: ✅ SUCCESS
- **Duration**: 0.46 seconds (1 epoch)
- **Steps**: 5 (10 files × 1 sample ÷ batch_size 2)
- **Data loader**: PyTorch
- **Protocol**: `file://`

Epoch statistics:
```json
{
  "start": "2026-02-07T18:35:46.195151",
  "end":   "2026-02-07T18:35:46.663193",
  "duration": "0.46"
}
```

---

### Test 2: TensorFlow + s3dlio + TFRecord

**Phase 1: Data Generation**

```bash
mlpstorage training datagen \
  --model resnet50 \
  --num-processes 1 \
  --data-dir /mnt/scratch/tensorflow-s3dlio-test \
  --params dataset.num_files_train=10 \
  --params dataset.num_samples_per_file=5 \
  --params dataset.record_length_bytes=102400
```

Results:
- **Status**: ✅ SUCCESS
- **Duration**: 0.03 seconds
- **Files created**: 10 TFRecord files
- **Size**: ~501 KB each (~5 MB total)
- **Location**: `/mnt/scratch/tensorflow-s3dlio-test/resnet50/train/`

**Phase 2: Data Reading (TensorFlow + s3dlio)**

```bash
mlpstorage training run \
  --model resnet50 \
  --accelerator-type h100 \
  --num-accelerators 1 \
  --client-host-memory-in-gb 16 \
  --data-dir /mnt/scratch/tensorflow-s3dlio-test \
  --params reader.data_loader=tensorflow \
  --params reader.storage_library=s3dlio \
  --params reader.storage_root=file:///mnt/scratch/tensorflow-s3dlio-test/resnet50 \
  --params dataset.num_files_train=10 \
  --params dataset.num_samples_per_file=5 \
  --params reader.batch_size=4 \
  --params train.epochs=1 \
  --params train.computation_time=0.001
```

Configuration overrides confirmed in results:
```yaml
# /tmp/mlperf_storage_results/.../overrides.yaml
- ++workload.reader.data_loader=tensorflow
- ++workload.reader.storage_library=s3dlio
- ++workload.reader.storage_root=file:///mnt/scratch/tensorflow-s3dlio-test/resnet50
```

Results:
- **Status**: ✅ SUCCESS
- **Duration**: 0.06 seconds (1 epoch)
- **Steps**: 12 (10 files × 5 samples ÷ batch_size 4 = 12.5 → 12)
- **Data loader**: TensorFlow
- **Protocol**: `file://`

---

### s3dlio Local Test Summary

| Test | Framework | Format | Protocol | Status | Duration |
|------|-----------|--------|----------|--------|----------|
| Test 1: unet3d | PyTorch | NPZ | `file://` | ✅ | 0.46 s |
| Test 2: resnet50 | TensorFlow | TFRecord | `file://` | ✅ | 0.06 s |

**Key finding**: s3dlio is framework-agnostic — it works with both PyTorch and
TensorFlow. This differs from s3torchconnector, which is PyTorch only.

---

## s3dlio — Cloud Protocol Tests

S3, Azure, and GCS protocol tests for s3dlio have not yet been measured. Commands
to run them are in [Object_Storage_Test_Guide.md](Object_Storage_Test_Guide.md)
and the test scripts listed below.

---

## minio — Test Results

minio functional and performance tests have not yet been captured. Run tests with:

```bash
# End-to-end DLIO cycle
bash tests/object-store/dlio_minio_cycle.sh

# GET throughput benchmark
python3 tests/object-store/test_s3lib_get_bench.py --library minio

# Checkpoint test
python3 tests/object-store/test_minio_checkpoint.py
```

Record results in this file following the same format as the s3dlio section above.

---

## s3torchconnector — Test Results

s3torchconnector functional and performance tests have not yet been captured. Run
tests with:

```bash
# End-to-end DLIO cycle
bash tests/object-store/dlio_s3torch_cycle.sh

# GET throughput benchmark
python3 tests/object-store/test_s3lib_get_bench.py --library s3torchconnector

# Checkpoint test
python3 tests/object-store/test_s3torch_checkpoint.py
```

Note: s3torchconnector supports PyTorch only. Use `data_loader: pytorch` in all
configurations.

Record results in this file following the same format as the s3dlio section above.

---

## Cross-Library Comparison

Once results are collected for all three object storage libraries, a comparison
table will be added here. The GET throughput benchmark script
(`tests/object-store/test_s3lib_get_bench.py`) runs all three libraries in
sequence and outputs a side-by-side table.

---

## See Also

- [Object_Storage_Test_Guide.md](Object_Storage_Test_Guide.md) — How to run tests
- [Object_Storage_Library_Setup.md](Object_Storage_Library_Setup.md) — Installation and configuration
- [STORAGE_LIBRARIES.md](STORAGE_LIBRARIES.md) — Library capability comparison
