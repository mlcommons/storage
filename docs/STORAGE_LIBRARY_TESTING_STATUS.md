# Storage Library Testing Status

## Overview
This document tracks testing status for the 4 new storage libraries integrated with MLPerf Storage benchmarks.

**Test Date**: February 7, 2026  
**Focus**: Validating new storage libraries (NOT default framework I/O)

---

## The 4 New Storage Libraries

### 1. s3dlio ✅ TESTED
**Status**: ✅ WORKING with both PyTorch and TensorFlow

**Framework Support**:
- ✅ PyTorch + s3dlio + NPZ format (unet3d)
- ✅ TensorFlow + s3dlio + TFRecord format (resnet50)

**Protocols Tested**:
- ✅ `file://` - Local filesystem via s3dlio

**Protocols NOT Tested**:
- ❌ `s3://` - S3-compatible storage
- ❌ `az://` - Azure Blob Storage
- ❌ `gs://` - Google Cloud Storage

**Performance**:
- PyTorch test: 5 steps in 0.46s (complete round-trip: generate NPZ → read with s3dlio)
- TensorFlow test: 12 steps in 0.06s (complete round-trip: generate TFRecord → read with s3dlio)

**Documentation**: [docs/S3DLIO_TEST_RECORD.md](S3DLIO_TEST_RECORD.md)

---

### 2. minio ❌ NOT TESTED
**Status**: Not tested yet

**Expected Support**:
- PyTorch + minio
- TensorFlow + minio
- S3-compatible protocol only

**Next Steps**:
- Test with MinIO server (S3-compatible)
- Validate credentials and authentication
- Compare performance against s3dlio

---

### 3. s3torchconnector ❌ NOT TESTED
**Status**: Not tested yet

**Expected Support**:
- ✅ PyTorch + s3torchconnector (PyTorch-only library)
- ❌ TensorFlow + s3torchconnector (NOT compatible)
- S3-compatible protocol only

**Next Steps**:
- Test with PyTorch workflows
- Validate S3 authentication
- Compare performance against s3dlio + PyTorch

---

### 4. azstoragetorch ❌ NOT TESTED
**Status**: Not tested yet

**Expected Support**:
- ✅ PyTorch + azstoragetorch (PyTorch-only library)
- ❌ TensorFlow + azstoragetorch (NOT compatible)
- Azure Blob Storage protocol only (`az://`)

**Next Steps**:
- Test with Azure Blob Storage
- Validate Azure authentication (account key, connection string, managed identity)
- Compare performance against s3dlio + PyTorch + Azure

---

## Summary

### Tested Libraries
| Library | Framework Support | Protocols Tested | Status |
|---------|------------------|------------------|--------|
| **s3dlio** | PyTorch ✅, TensorFlow ✅ | file:// ✅ | ✅ WORKING |
| **minio** | PyTorch ❓, TensorFlow ❓ | None | ❌ NOT TESTED |
| **s3torchconnector** | PyTorch only | None | ❌ NOT TESTED |
| **azstoragetorch** | PyTorch only | None | ❌ NOT TESTED |

### Testing Priority
1. **s3dlio with cloud protocols** (s3://, az://, gs://) - Highest priority since library already validated
2. **minio** - Test S3-compatible storage with dedicated MinIO library
3. **s3torchconnector** - PyTorch-specific S3 library
4. **azstoragetorch** - PyTorch-specific Azure library

### Key Findings
1. ✅ **s3dlio is framework-agnostic** - Works with BOTH PyTorch and TensorFlow
2. ✅ **Complete round-trips validated** - Generate → Read cycle works for both frameworks
3. ✅ **Command-line overrides work** - Can specify storage_library via --params
4. ✅ **file:// protocol works** - Local testing validated before cloud testing
5. ⚠️ **PyTorch requires NPZ format** - TFRecord not supported by PyTorch in DLIO
6. ⚠️ **TensorFlow can use TFRecord or NPZ** - Both formats work with TensorFlow

---

## Next Steps

### Immediate: Test s3dlio with Cloud Storage
Since s3dlio is validated with `file://`, test cloud protocols next:

```bash
# s3dlio + PyTorch + S3
mlpstorage training run \
  --model unet3d \
  --params reader.storage_library=s3dlio \
  --params reader.storage_root=s3://bucket-name/unet3d \
  ...

# s3dlio + TensorFlow + Azure
mlpstorage training run \
  --model resnet50 \
  --params reader.storage_library=s3dlio \
  --params reader.storage_root=az://container/resnet50 \
  ...
```

### Then: Test Other Libraries
Once s3dlio cloud testing is complete, test the other 3 libraries with their respective protocols.
