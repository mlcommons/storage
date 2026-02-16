# PR Readiness Action Plan

## Current State Analysis

### TF_ObjectStorage Branch (Current)
- ✅ 2 commits ahead of origin (multi-library work)
- ⚠️ Untracked files:
  - `dlio_benchmark/` - Modified checkpoint files (needs to go to Feature #2)
  - `tests/checkpointing/compare_methods.py` - Recovered from streaming-checkpoint-poc
  - Various benchmark scripts
  - New strategy doc

### Issues to Resolve:
1. **dlio_benchmark/ modifications** are on wrong branch (TF_ObjectStorage vs checkpoint branch)
2. **Untracked files** need to be committed to appropriate branches
3. **Feature branches** haven't been created yet

---

## 📋 STEP-BY-STEP ACTION PLAN

### Phase 1: Clean Up Current Branch State (TF_ObjectStorage)

**Goal**: Commit only multi-library work to TF_ObjectStorage

```bash
cd /home/eval/Documents/Code/mlp-storage

# Add strategy document and setup script (useful for all branches)
git add docs/TF_ObjectBranch-Strategy.md
git add tests/feature_branch_setup.sh
git commit -m "docs: Add branch strategy and feature branch setup script"

# Add benchmark scripts that belong to multi-library work
git add tests/scripts/benchmark_libraries_v8.py
git add tests/scripts/benchmark_datagen_v2.py
git add tests/scripts/benchmark_storage_libraries.py
git commit -m "test: Add multi-library benchmark scripts"

# Push to origin (optional - can wait)
# git push origin TF_ObjectStorage
```

**DON'T commit yet:**
- `dlio_benchmark/` (belongs to checkpoint feature)
- `tests/checkpointing/` (belongs to checkpoint feature)

---

### Phase 2: Create Feature Branch #1 (Multi-Library Storage)

**Goal**: Clean feature branch for PR #1

```bash
# Create feature branch from current TF_ObjectStorage
git checkout TF_ObjectStorage
git checkout -b feature/multi-library-storage

# This branch now has:
# - All multi-library storage changes
# - Benchmark scripts (v8)
# - Strategy document

# Verify clean state
git status
git log --oneline -5

# Ready for PR!
```

**PR #1 Checklist:**
- [ ] Branch created: `feature/multi-library-storage`
- [ ] Contains multi-library adapter code
- [ ] Contains benchmark scripts
- [ ] No checkpoint/dgen-py code mixed in
- [ ] Passes basic smoke tests

---

### Phase 3: Handle dlio_benchmark Modifications for Checkpoint Feature

**Issue**: We modified `dlio_benchmark/dlio_benchmark/checkpointing/pytorch_checkpointing.py` 
and `tf_checkpointing.py` on TF_ObjectStorage, but they should be on the checkpoint branch.

**Solution Options:**

#### Option A: Stash and Apply (Recommended)
```bash
# Save the dlio_benchmark changes
git checkout TF_ObjectStorage
git add dlio_benchmark/
git stash  # Temporarily save changes

# Switch to checkpoint branch
git checkout streaming-checkpoint-poc

# Apply the changes
git stash pop

# Verify they applied correctly
git status
git diff dlio_benchmark/dlio_benchmark/checkpointing/pytorch_checkpointing.py

# Commit on checkpoint branch
git add dlio_benchmark/
git commit -m "feat: Integrate dgen-py into PyTorch and TensorFlow checkpointing"

# Also add recovered test
git add tests/checkpointing/
git commit -m "test: Add checkpoint comparison test suite"
```

#### Option B: Manual Copy (If stash fails)
```bash
# Back up the changes
cp -r dlio_benchmark/ /tmp/dlio_benchmark_backup/

# Switch to checkpoint branch
git checkout streaming-checkpoint-poc

# Copy over
cp -r /tmp/dlio_benchmark_backup/ dlio_benchmark/

# Commit
git add dlio_benchmark/
git commit -m "feat: Integrate dgen-py into PyTorch and TensorFlow checkpointing"
```

---

### Phase 4: Create Feature Branch #2 (Checkpoint Optimization)

**Goal**: Clean feature branch for PR #2

```bash
# Make sure we're on checkpoint branch with new changes
git checkout streaming-checkpoint-poc

# Create feature branch
git checkout -b feature/checkpoint-dgen-optimization

# This branch now has:
# - StreamingCheckpointing class
# - dgen-py integration in checkpointing
# - gen_random_tensor() optimization
# - compare_methods.py test suite

# Verify
git status
git log --oneline -10

# Ready for PR!
```

**PR #2 Checklist:**
- [ ] Branch created: `feature/checkpoint-dgen-optimization`
- [ ] Contains dgen-py integration
- [ ] Contains StreamingCheckpointing
- [ ] Contains updated checkpointing files
- [ ] Contains test suite (compare_methods.py)
- [ ] Passes checkpoint benchmarks

---

### Phase 5: Test Each Feature Independently

#### Test Feature #1 (Multi-Library)
```bash
git checkout feature/multi-library-storage

# Activate virtual environment
source .venv/bin/activate

# Test s3dlio
export STORAGE_LIBRARY=s3dlio
python tests/scripts/benchmark_libraries_v8.py --target fast --num-objects 100 --quick --libraries s3dlio

# Test minio
export STORAGE_LIBRARY=minio
python tests/scripts/benchmark_libraries_v8.py --target fast --num-objects 100 --quick --libraries minio

# Test s3torchconnector (default)
unset STORAGE_LIBRARY
python tests/scripts/benchmark_libraries_v8.py --target fast --num-objects 100 --quick --libraries s3torchconnectorclient

# ✅ Expected: All 3 libraries work
```

#### Test Feature #2 (Checkpoint + dgen-py)
```bash
git checkout feature/checkpoint-dgen-optimization

# Test dgen-py integration
export DLIO_DATA_GEN=dgen
python -c "from dlio_benchmark.utils.utility import gen_random_tensor; import numpy as np; arr = gen_random_tensor((1000,), np.float32); print('✅ dgen-py works')"

# Test checkpoint generation
python tests/checkpointing/compare_methods.py

# Test with dlio_benchmark (if you have a config)
# dlio_benchmark --config configs/checkpoint_test.yaml

# ✅ Expected: 155x speedup in data generation
```

---

### Phase 6: Integration Testing

**Goal**: Verify both features work together

```bash
# Merge both into TF_ObjectStorage for integration test
git checkout TF_ObjectStorage

# Merge feature 1
git merge feature/multi-library-storage
# (Should be fast-forward, no conflicts)

# Merge feature 2
git merge feature/checkpoint-dgen-optimization
# (May have conflicts - see resolution strategy below)

# If conflicts, resolve and test
git status
# ... resolve conflicts ...
git add <resolved-files>
git commit -m "merge: Integrate multi-library and checkpoint features"

# Test integration
export DLIO_DATA_GEN=dgen
export STORAGE_LIBRARY=s3dlio
python tests/scripts/benchmark_libraries_v8.py --target fast --num-objects 100 --libraries s3dlio

# ✅ Expected: s3dlio + dgen-py = maximum performance
```

---

### Phase 7: Push and Create PRs

```bash
# Push feature branches to GitHub
git push origin feature/multi-library-storage
git push origin feature/checkpoint-dgen-optimization

# On GitHub, create two PRs:
# PR #1: feature/multi-library-storage → origin/TF_ObjectStorage (or main)
#   Title: "feat: Add multi-library S3 storage support (s3dlio, minio, s3torchconnector)"
#   Description: See PR #1 template below

# PR #2: feature/checkpoint-dgen-optimization → origin/TF_ObjectStorage (or main)  
#   Title: "feat: Optimize checkpoint data generation with dgen-py (155x speedup)"
#   Description: See PR #2 template below
```

---

## 📝 PR Description Templates

### PR #1: Multi-Library Storage Support

```markdown
## Summary
Adds support for 3 S3-compatible storage libraries in DLIO Benchmark:
- s3dlio (zero-copy, multi-protocol)
- AWS s3torchconnector (existing default)
- MinIO native SDK

## Motivation
- Enable performance comparison between storage libraries
- Leverage s3dlio's zero-copy optimization (2-3x better write performance)
- Support MinIO-specific deployments

## Changes
- Modified `patches/s3_torch_storage.py` with multi-library adapter pattern
- Added `storage_library` configuration parameter
- Added `STORAGE_LIBRARY` environment variable support
- Added comprehensive benchmark suite (`benchmark_libraries_v8.py`)

## Performance Results
Tested on VAST storage (10 GB/s capable):
- **s3dlio**: 2.88 GB/s PUT, 7.07 GB/s GET ⭐ Best overall
- **minio**: 0.70 GB/s PUT, 6.77 GB/s GET (excellent reads)
- **s3torchconnector**: 1.89 GB/s PUT, 2.39 GB/s GET (baseline)

## Testing
- [x] All 3 libraries tested with 3000 objects × 16 MB
- [x] Backward compatibility verified (defaults to s3torchconnector)
- [x] Integration with existing DLIO configs

## Configuration Example
```yaml
reader:
  storage_library: s3dlio  # or 'minio', 's3torchconnector'
```

## Related Issues
Addresses performance optimization for large-scale checkpointing workloads.
```

### PR #2: Checkpoint & Data Generation Optimization

```markdown
## Summary
Optimizes DLIO Benchmark data generation with dgen-py (Rust-based RNG), achieving **155x speedup** over NumPy.

## Motivation
- Checkpoint generation for large models (70B+ parameters) was bottlenecked by NumPy RNG
- 100 GB checkpoint took 65 seconds just to generate random data
- Real storage I/O was faster than data generation

## Changes
- Added `gen_random_tensor()` with dgen-py support in `utils/utility.py`
- Modified `pytorch_checkpointing.py` to use dgen-py (replaces `torch.rand()`)
- Modified `tf_checkpointing.py` to use dgen-py (replaces `tf.random.uniform()`)
- Added `DLIO_DATA_GEN` environment variable control
- Added `dataset.data_gen_method` YAML configuration
- Added test suite: `tests/checkpointing/compare_methods.py`

## Performance Results
- **Data generation**: 1.54 GB/s → **239 GB/s** (155x faster)
- **100 GB checkpoint**: 65s → **0.4s** generation time
- **Bottleneck**: Now network/storage (as it should be), not data generation

## Usage
```bash
# Enable dgen-py optimization (auto-detect if installed)
export DLIO_DATA_GEN=dgen
dlio_benchmark --config checkpoint_config.yaml

# Or in YAML:
dataset:
  data_gen_method: dgen  # or 'numpy' for legacy
```

## Backward Compatibility
- Automatic fallback to NumPy if dgen-py not installed
- Default behavior unchanged (auto-detect)
- User can force NumPy with `DLIO_DATA_GEN=numpy`

## Testing
- [x] PyTorch checkpoint generation with dgen-py
- [x] TensorFlow checkpoint generation with dgen-py  
- [x] Fallback to NumPy verified
- [x] compare_methods.py benchmark suite passes

## Dependencies
- Optional: `pip install dgen-py` (155x speedup)
- Works without dgen-py (NumPy fallback)
```

---

## ⚠️ Potential Conflicts

When merging both features into TF_ObjectStorage:

**Expected conflicts:**
- `patches/s3_torch_storage.py` - Both features modify this file
- `docs/` - Multiple new docs added

**Resolution:**
1. Keep both features' changes
2. Test that s3dlio + dgen-py work together
3. Verify no functionality lost

---

## 🎯 Success Criteria

### Feature #1 (Multi-Library) Ready When:
- [ ] Branch created and pushed
- [ ] 3 libraries tested and working
- [ ] Benchmark results documented
- [ ] PR description written
- [ ] No merge conflicts with origin

### Feature #2 (Checkpoint) Ready When:
- [ ] Branch created and pushed  
- [ ] dgen-py integration tested
- [ ] 155x speedup verified
- [ ] compare_methods.py passes
- [ ] PR description written
- [ ] No merge conflicts with origin

### Integration Ready When:
- [ ] Both features merged into TF_ObjectStorage
- [ ] Combined testing passes (s3dlio + dgen-py)
- [ ] No regressions in either feature
- [ ] Documentation updated

---

## 📅 Timeline Estimate

- **Phase 1-2** (Feature #1 branch): 15 minutes
- **Phase 3-4** (Feature #2 branch): 30 minutes  
- **Phase 5** (Independent testing): 30 minutes
- **Phase 6** (Integration testing): 30 minutes
- **Phase 7** (Push and create PRs): 15 minutes

**Total: ~2 hours** (assuming no major issues)

---

## 🆘 Troubleshooting

### If dlio_benchmark/ won't stash:
- Use Option B (manual copy)
- Or commit to temp branch, cherry-pick to checkpoint branch

### If merge conflicts are complex:
- Create clean branches from origin/main
- Cherry-pick specific commits
- Manual merge of conflict files

### If tests fail:
- Check virtual environment activated
- Verify dgen-py installed: `pip list | grep dgen`
- Check environment variables: `env | grep DLIO`

---

**Ready to proceed?** Start with Phase 1!
