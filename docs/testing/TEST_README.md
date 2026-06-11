# S3 Storage Implementation Tests

Each test script is independent and can be run separately.

## Test Scripts

### 1. MLP + s3torchconnector
```bash
cd /home/eval/Documents/Code/mlp-storage
./test_mlp_s3torch.sh
```
- **Bucket**: mlp-s3torch
- **Library**: s3torchconnector (AWS official connector)
- **Expected**: ✅ PASS

### 2. MLP + minio
```bash
cd /home/eval/Documents/Code/mlp-storage
./test_mlp_minio.sh
```
- **Bucket**: mlp-minio
- **Library**: minio (MinIO native SDK)
- **Expected**: ✅ PASS

### 3. dpsi + s3torchconnector (BASELINE)
```bash
cd /home/eval/Documents/Code/mlp-storage-dpsi
./test_dpsi_s3torch.sh
```
- **Bucket**: dpsi-s3torch
- **Library**: s3torchconnector (bucket+key architecture from PR #232)
- **Expected**: ✅ PASS
- **Note**: This is the reference implementation. MLP should match or exceed this.

### 4. MLP + s3dlio
```bash
cd /home/eval/Documents/Code/mlp-storage
./test_mlp_s3dlio.sh
```
- **Bucket**: mlp-s3dlio
- **Library**: s3dlio (our high-performance library)
- **Expected**: ❌ FAIL (known bug in compat layer line 571)

## What Each Test Does

1. **Clean bucket** - Removes all existing objects
2. **Verify empty** - Confirms bucket is clean
3. **Run datagen** - Generates 3 NPZ files (unet3d dataset)
4. **Verify train files** - Lists train directory objects
5. **Complete listing** - Shows full bucket contents

## Expected Output

Each test should create 3 files in the train directory:
- `test-run/unet3d/train/img_0_of_3.npz`
- `test-run/unet3d/train/img_1_of_3.npz`
- `test-run/unet3d/train/img_2_of_3.npz`

Plus empty directories for valid/ and test/

## Next Steps

After confirming tests 1-3 work:
- Fix s3dlio bug in `/home/eval/Documents/Code/s3dlio/python/s3dlio/compat/s3torchconnector.py` line 571
- Re-run test 4 to verify fix
