# Quick Testing Reference

## Test Each PR Before Pushing to GitHub

### PR#1: Multi-Library Storage
```bash
git checkout feature/multi-library-storage
./test_pr1_multilib.sh
```
**Tests**: Data generation + training with s3torchconnector, minio, s3dlio  
**Expected**: All 6 tests pass (2 tests × 3 libraries)

---

### PR#2: Checkpoint Optimization  
```bash
git checkout feature/checkpoint-dgen-optimization
./test_pr2_checkpoint.sh
```
**Tests**: Local file checkpoint with dgen-py optimization  
**Expected**: Local tests pass, S3 tests skip (requires PR#1)

---

### Integration: Both PRs Together
```bash
./test_integration_pr1_pr2.sh
```
**Tests**: Full workflow (generate + train + checkpoint) with all 3 libraries  
**Expected**: All 9 tests pass (3 tests × 3 libraries)

---

## Prerequisites

All test scripts automatically handle:
- ✅ Activating virtual environment (`.venv`)
- ✅ Loading credentials (`.env`)
- ✅ Verifying environment is ready

Just make sure:
- `.env` file exists in repository root
- Virtual environment is set up (`.venv/` directory exists)
- MinIO endpoint at `172.16.1.40:9000` is accessible

---

## Quick Validation Commands

Before running tests, verify environment:

```bash
# Check virtual environment exists
ls -la .venv/

# Check credentials file
cat .env

# Check endpoint connectivity
curl http://172.16.1.40:9000
```

---

## What Gets Tested

### PR#1
- Data generation to S3 with 3 different libraries
- Training (reading from S3) with 3 different libraries
- Library selection via `storage_library` parameter

### PR#2
- Checkpoint data generation with dgen-py (155x faster)
- Memory efficiency (99.8% reduction)
- Local file checkpointing

### Integration
- Everything from PR#1 AND PR#2 together
- S3 checkpointing with all 3 libraries
- dgen-py optimization + multi-library storage

---

## Expected Runtimes

- **PR#1 Test**: ~5-10 minutes (small dataset: 5 files × 5 samples)
- **PR#2 Test**: ~2-5 minutes (local files only)
- **Integration Test**: ~10-15 minutes (full workflow × 3 libraries)

---

## Success = Push to GitHub

Once all tests pass:
```bash
git push origin feature/multi-library-storage
git push origin feature/checkpoint-dgen-optimization
```

Then create PRs on GitHub!
